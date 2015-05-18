//last date of modification : 31-08-12 @Sahil
#include <math.h>
#include <iostream>
#include <vector>

int nDescripSize = 128;

cv::Scalar c_white(255, 255, 255);
cv::Scalar c_gray(127, 127, 127);
cv::Scalar c_black(0, 0, 0);
cv::Scalar c_red(0, 0, 255);
cv::Scalar c_green(0, 255, 0);
cv::Scalar c_blue(255, 0, 0);
cv::Scalar c_cyan(255, 255, 0);
cv::Scalar c_magenta(255, 0, 255);
cv::Scalar c_lemon(32, 255, 255);
cv::Scalar c_orange(0, 127, 255);
cv::Scalar c_pink(127, 0, 255);
cv::Scalar c_violet(255, 0, 127);

/////Color Code
//Not tracked : Blue
//    Tracked from previous frame: Green (Green lines)
//    Tracked from older frames: Purple (no lines)
//Rejected : red lines

//cvScalar c_blue = cvScalar(255,0,0);



/* Calculating distance between two trajectories
 *
 * Input: trajectories
 */
double CalTrajectoryDistance(std::vector<double> descriptor1, float descriptor2[]) {
    double distance = 0;
    for(int i = 0; i < nDescripSize ; i++)
        distance += pow((descriptor1[i] - descriptor2[i]), 2);

    //printf("Returning Distance : %10.9f %10.9f\n" ,distance);
    return distance;
}

/* Similiar to previous function
 */
double CalTrajectoryDistance(std::vector<double> descriptor1, std::vector<double> descriptor2) {
    double distance = 0;
    for(int i = 0; i < nDescripSize ; i++)
        distance += pow((descriptor1[i] - descriptor2[i]), 2);

    //printf("Returning Distance : %10.9f %10.9f\n" ,distance);
    return distance;
}




/* Calculating distances between first two elements of two trajectories
 */
double CalcDispDistance(std::vector<int> descriptor1, std::vector<double> descriptor2) {
    double distance = 0;
    for(int i = 0; i < 2; i++)
        distance += pow((double(descriptor1[i])-descriptor2[i]), 2);

    //printf("Distance between %d , %d & %g , %g is : %g\n" ,descriptor1[0],descriptor1[1],descriptor2[0],descriptor2[1],distance);
    distance = sqrt(distance);
    return distance;
}



/* Getting SIFT keypoints from input image (computed by libsiftfast)
 *
 * Input:
 * input - input image
 * nSiftScales, nSiftInitSigma, nSiftPeakThrs - SIFT parameter
 * x,y,width,height - coordinates of input area
 *
 * Output:
 * keypts - SIFT keypoints (format Keypoint from Libsiftfast)
 */
void GetSiftKeypoints(IplImage *input, int nSiftScales, double nSiftInitSigma, double nSiftPeakThrs, int x, int y, int width, int height, Keypoint &keypts) {

    Image img_sift = CreateImage(width, height);

    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < height; ++j) {
            CvScalar s = cvGet2D(input, j + y, i + x);
            img_sift->pixels[i * img_sift->stride + j] = (s.val[0] + s.val[1] + s.val[2]) / (3.*255);
        }
    }
    SiftParameters dparm = GetSiftParameters();
    dparm.Scales = nSiftScales;
    dparm.InitSigma = nSiftInitSigma;
    dparm.PeakThresh = nSiftPeakThrs;
    SetSiftParameters(dparm);

    keypts = GetKeypoints(img_sift);
    DestroyAllResources();
}

void GetSiftKeypoints(cv::Mat input, int nSiftScales, double nSiftInitSigma, double nSiftPeakThrs, int x, int y, int width, int height, Keypoint &keypts) {

    Image img_sift = CreateImage(width, height);

    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < height; ++j) {
            cv::Vec3b s = input.at<cv::Vec3b>(j+y, i+x);
            img_sift->pixels[i * img_sift->stride + j] = (s[0] + s[1] + s[2]) / (3.*255);
        }
    }
    SiftParameters dparm = GetSiftParameters();
    dparm.Scales = nSiftScales;
    dparm.InitSigma = nSiftInitSigma;
    dparm.PeakThresh = nSiftPeakThrs;
    SetSiftParameters(dparm);

    keypts = GetKeypoints(img_sift);
    DestroyAllResources();
}




/* Creating trajectories from keypoints
 *
 * Input:
 * cvm_org - input image
 * nRx,nRy,nRw,nRh - coordinates of area of interest
 * nFrameNo - number of current frame
 * nTotalFrames - count of total frames
 * nDistThres - distance threshold
 * nSiftScales, nSiftInitSigma, nSiftPeakThres - sift parameters
 * nXDelta, nYDelta - displacement of keypoints coordinates
 * nXPrev, nYPrev - keypoints coordinate in previous frame
 * nDispThresh - threshold for nXDelta, nYDelta
 * nFeatureMode -
 * sFname -
 * nMatchThres - threshold for matching between keypoints and trajectories
 * nReinforce -
 * nKeyPosThres -
 * nTrjLBlank - size of blank for trajectory (where the trajectory is not connected)
 *
 * Output:
 * vnTracker - created trajectories
 * bEndofRotation - flag which indicates if the end of processing was reached
 * bEndofStitching - flag which indicates if the end of stitching was reached
 * nMatchOriginFirst -
 * vFrameMatches -
 * nFrameNrMax -
 * vFirstFrameKeypts - keypoints at first frame
 * vnKeyPrev - keypoints at previous frame
 * vnMatchedKeyCnt -
 * cvm_match -
 */
void Trajectory (cv::Mat cvm_org, int nRx, int nRy, int nRw, int nRh, int nFrameNo, int nTotalFrames,
                 std::vector<std::vector<std::vector<double> > >& vnTracker, double nDistThres,int nSiftScales,double nSiftInitSigma,
                 double nSiftPeakThresh,bool& bEndofRotation,bool& bEndofStitching,int& nMatchOriginFirst,std::vector<int>& vFrameMatches,
                 int &nFrameNrMax, int nXDelta, int nYDelta, int nXPrev,int nYPrev,int nDispThresh, int nFeatureMode, std::string sFname, std::vector<std::vector <double> >& vFirstFrameKeypts,
                 double nMatchThres, int nReinforce, int nKeyPosThres, int nTrjLBlank,
                 std::vector<std::vector<double> > &vnKeyPrev, std::vector<int> &vnMatchedKeyCnt, cv::Mat &cvm_match)
{
    //printf("ssss %f\n", nMatchThres);

    int trackedold=0;
    int trackedprevious=0;
    int nottracked=0;
    int tracked=0;
    int nKeyRemoved=0;

    ////// Get SIFT keypoints /////////////////
    Keypoint keypts, keypts_tmp;
    GetSiftKeypoints(cvm_org, nSiftScales, nSiftInitSigma, nSiftPeakThresh, nRx, nRy, nRw, nRh, keypts);
    keypts_tmp = keypts;

    ///////////////////////////////////////////Keypoints Calculated Now Tracking
    int nMatchOrigin = 0;
    //Creating Trajectories for the first frame

    ///////** If it is the first frame, we create new trajectories for all key points **////////////////
    if (nFrameNo == 0) {
        int nKeyCnt = 0;
        while (keypts) {

            int nTrackerNo = vnTracker.size();
            vnTracker.resize(nTrackerNo + 1);
            vnTracker[nTrackerNo].resize(1);
            vnTracker[nTrackerNo][0].resize(139,0);

            vFirstFrameKeypts.resize(nKeyCnt+1);
            vFirstFrameKeypts[nKeyCnt].resize(129);
            vnKeyPrev.resize(nKeyCnt+1);
            vnKeyPrev[nKeyCnt].resize(133);
            for(int i = 0; i < nDescripSize; i++) {
                vnTracker[nTrackerNo][0][i] = keypts->descrip[i];
                vFirstFrameKeypts[nKeyCnt][i] = keypts->descrip[i];
                vnKeyPrev[nKeyCnt][i] = keypts->descrip[i];
            }

            vnTracker[nTrackerNo][0][128] = keypts->row;
            vnTracker[nTrackerNo][0][129] = keypts->col;
            vnTracker[nTrackerNo][0][130] = keypts->scale;
            vnTracker[nTrackerNo][0][131] = keypts->ori;
            vnTracker[nTrackerNo][0][132] = keypts->fpyramidscale;
            vnTracker[nTrackerNo][0][133] = 0;          //the trajectory update counter. so each time a trajectory is accessed this is no accessed we go ++, and each time it is accessed we go 0
            vnTracker[nTrackerNo][0][134] = 1;          //no 134 denotes whether the trajectory is good or not i.e. 1: valid, 0: invalid
            vnTracker[nTrackerNo][0][135] = 0;          //this indicates whether a particular trajectory has been allotted a keypoint in a frame 0->not processed 1->processed
            vnTracker[nTrackerNo][0][136] = 1;          //Indicating that this trajectory started in the beginning
            vnTracker[nTrackerNo][0][137] = 0;          //This denotes the flag for the NN Matching with the first sift
            vnTracker[nTrackerNo][0][138] = nFrameNo;   //This records the frame number for each keypoint that is added to a trajectory

            vnKeyPrev[nKeyCnt][128] = keypts->row;
            vnKeyPrev[nKeyCnt][129] = keypts->col;
            vnKeyPrev[nKeyCnt][130] = keypts->scale;
            vnKeyPrev[nKeyCnt][131] = keypts->ori;
            vnKeyPrev[nKeyCnt][132] = keypts->fpyramidscale;

            cv::rectangle(cvm_match, cv::Point(keypts->row + nXDelta-2, keypts->col + nYDelta-2), cv::Point(keypts->row + nXDelta+2, keypts->col + nYDelta+2), c_blue, 1);  /////blue for not tracked
            cv::rectangle(cvm_org, cv::Point(keypts->row + nRx-2, keypts->col + nRy-2), cv::Point(keypts->row + nRx+2, keypts->col + nRy+2), c_blue, 2);
            keypts = keypts->next;
            nKeyCnt++;

        }
        printf("\nKeypoints in first frame: %d\n", nKeyCnt);

        vnMatchedKeyCnt[nFrameNo] = nKeyCnt;
    }
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////Trajectories created for first frame



    ///////** If it is not the first frame and the end of rotation has not been spotted, we have to find the nearest matches **////////////////
    else if (bEndofRotation == false) {

        int nTmpMatchCnt = 0;
        while (keypts) {
            float nTmpMaxD = 999999;
            float nTmpD;
            for(size_t i = 0; i < vFirstFrameKeypts.size(); i++) {
                nTmpD = CalTrajectoryDistance(vFirstFrameKeypts[i], keypts->descrip);

                if(nTmpD < nTmpMaxD) nTmpMaxD = nTmpD;
            }
            if(nTmpMaxD < nDistThres) nTmpMatchCnt++;

            keypts = keypts->next;
        }
        vnMatchedKeyCnt[nFrameNo] = nTmpMatchCnt;


        /////Concept to Remove outliers: Add as usual but store the displacement vector as well as the allotted trajectory for each tracked keypoint. use the trajectory index to delete the keypoint from this if necessary
        keypts = keypts_tmp;
        std::vector<std::vector <int> > vnDisplacement (2000,std::vector<int> (3,0)); //0: delta(x), 1: delta(y), 2: trajectory index

        //for each keypoint we try to find a match. If matchh is found, the keypt is added to that trajectory. Else we create a new trajectory for that keypt.
        while (keypts) {

            double nDistMin = 999999;
            int nMatchKey = -1;
            for (size_t i = 0; i < vnKeyPrev.size(); i++) {
                double nKeyDistance = CalTrajectoryDistance(vnKeyPrev[i], keypts->descrip);
                double nDispX = keypts->row - vnKeyPrev[i][128];
                double nDispY = fabs(vnKeyPrev[i][129] - keypts->col);

                if(nKeyDistance < nDistMin) {
                    if (nReinforce == 1)
                        if (nDispX < 3 || nDispY > nKeyPosThres) continue;

                    nDistMin = nKeyDistance;
                    nMatchKey = i;
                }
            }

            if (nDistMin < nDistThres) {
                if (nMatchKey == -1) {printf("ERROR............530\n"); continue;}
            }

            if (nMatchKey >= 0 && nDistMin < nDistThres) {
            //if (true) {

                cv::rectangle(cvm_org, cv::Point(keypts->row + nRx-2, keypts->col + nRy-2), cv::Point(keypts->row + nRx+2, keypts->col + nRy+2), c_blue, 2);

                /////////** Find the Closest trajectory for the current keypoint (?????????If already allotted l) **//////////
                int nMatchedTraj = -1; //indicates the index no of the best matched trajectory
                nDistMin = 999999;
                for (int i = 0; i < int(vnTracker.size()); i++) {
                    //if that trajectory is valid and has not been allotted a keypoint in this particular frame
                    if(vnTracker[i][0][134] == 1 && vnTracker[i][0][135] == 0) {
                        int nLastFeatureIdx = vnTracker[i].size() - 1;

                        //if (nFrameNo < 30)
                        if (nFrameNo - vnTracker[i][nLastFeatureIdx][138] > nTrjLBlank) continue;

                        double nKeyDistance = CalTrajectoryDistance(vnTracker[i][nLastFeatureIdx], keypts->descrip);
                        double nDispX = vnTracker[i][nLastFeatureIdx][128] - keypts->row;
                        double nDispY = fabs(vnTracker[i][nLastFeatureIdx][129] - keypts->col);

                        if(nKeyDistance < nDistMin) {
                            if (nReinforce == 1)
                                if (nDispY > nKeyPosThres) continue;

                            nDistMin = nKeyDistance;
                            nMatchedTraj = i;
                        }
                    }
                }//end of tracker loop


                //If the closest trajectory satisfies the threshold, it means we have tracked the keypoint so we must add it to the appropriate trajectory
                if(nDistMin < nDistThres) {
                    if(nMatchedTraj == -1) {printf("ERROR...................630   %f %f %d\n", nDistMin, nDistThres, nMatchedTraj); continue;}

                    int tmpsize = vnTracker[nMatchedTraj].size();
                    vnTracker[nMatchedTraj].resize(tmpsize+1);
                    vnTracker[nMatchedTraj][tmpsize].resize(139,0);
                    for(int i = 0; i < nDescripSize; i++)
                        vnTracker[nMatchedTraj][tmpsize][i] = keypts->descrip[i];

                    vnTracker[nMatchedTraj][tmpsize][128] = keypts->row;
                    vnTracker[nMatchedTraj][tmpsize][129] = keypts->col;
                    vnTracker[nMatchedTraj][tmpsize][130] = keypts->scale;
                    vnTracker[nMatchedTraj][tmpsize][131] = keypts->ori;
                    vnTracker[nMatchedTraj][tmpsize][132] = keypts->fpyramidscale;
                    vnTracker[nMatchedTraj][0][135] = 1;  //indicating that it has been allotted a keypoint in this frame
                    vnTracker[nMatchedTraj][tmpsize][138] = nFrameNo;
                    vnDisplacement[tracked][0] = vnTracker[nMatchedTraj][tmpsize][128] - vnTracker[nMatchedTraj][tmpsize-1][128];
                    vnDisplacement[tracked][1] = vnTracker[nMatchedTraj][tmpsize][129] - vnTracker[nMatchedTraj][tmpsize-1][129];
                    vnDisplacement[tracked][2] = nMatchedTraj;
                    tracked++;


                    if(vnTracker[nMatchedTraj][tmpsize-1][138] == nFrameNo - 1) {
                        cv::rectangle(cvm_match, cv::Point(keypts->row + nXDelta-2 ,keypts->col + nYDelta-2), cv::Point(keypts->row + nXDelta+2 ,keypts->col + nYDelta+2), c_green, 2);

                        double oldx = vnTracker[nMatchedTraj][tmpsize-1][128] + nXPrev;
                        double oldy = vnTracker[nMatchedTraj][tmpsize-1][129] + nYPrev;
                        if (fabs(keypts->col - vnTracker[nMatchedTraj][tmpsize-1][129]) > nKeyPosThres) printf ("%f %f %f %d\n", keypts->row - vnTracker[nMatchedTraj][tmpsize-1][129], keypts->row, vnTracker[nMatchedTraj][tmpsize-1][129], nKeyPosThres);
                        cv::rectangle(cvm_match, cv::Point(oldx-2 ,oldy-2), cv::Point(oldx+2 ,oldy+2), c_green, 2);
                        cv::line(cvm_match, cv::Point(keypts->row + nXDelta, keypts->col + nYDelta), cv::Point(oldx, oldy), c_green,1);
                    }
                    else {
                        ///if tracked from older frames, draw only keypoint
                        cv::rectangle(cvm_match, cv::Point(keypts->row + nXDelta-2, keypts->col + nYDelta-2), cv::Point(keypts->row + nXDelta+2 ,keypts->col + nYDelta+2), c_pink, 2);
                        trackedold++;
                    }
                }
                //If the closest trajectory does not satisfy the threshold, it means no matching trajectory found. So create New Trajectory for that keypoint
                else if(nDistMin >= nDistThres) {
                    int nTrackerNo = vnTracker.size();
                    vnTracker.resize(nTrackerNo + 1);
                    vnTracker[nTrackerNo].resize(1);
                    vnTracker[nTrackerNo][0].resize(139,0);
                    for(int i = 0; i < nDescripSize; i++) vnTracker[nTrackerNo][0][i] = keypts->descrip[i];

                    vnTracker[nTrackerNo][0][128] = keypts->row;
                    vnTracker[nTrackerNo][0][129] = keypts->col;
                    vnTracker[nTrackerNo][0][130] = keypts->scale;
                    vnTracker[nTrackerNo][0][131] = keypts->ori;
                    vnTracker[nTrackerNo][0][132] = keypts->fpyramidscale;
                    vnTracker[nTrackerNo][0][133] = 0;      //the trajectory update counter. so each time a trajectory is accessed this is no accessed we go ++, and each time it is accessed we go 0
                    vnTracker[nTrackerNo][0][134] = 1;      //no 134 denotes whether the trajectory is good or not i.e. 1: valid, 0: invalid
                    vnTracker[nTrackerNo][0][135] = 1;      //this indicates whether a particular trajectory has been allotted a keypoint in a frame 0->not processed 1->processed
                    vnTracker[nTrackerNo][0][136] = 0;      //Indicating that this trajectory started in the beginning
                    vnTracker[nTrackerNo][0][137] = 0;      //This denotes the flag for the NN Matching with the first sift
                    vnTracker[nTrackerNo][0][138] = nFrameNo;

                    cv::rectangle(cvm_match, cv::Point(keypts->row + nXDelta-2 ,keypts->col + nYDelta-2), cv::Point(keypts->row + nXDelta+2 ,keypts->col + nYDelta+2), c_blue, 1); ///blue for untracked
                    nottracked++;
                    //    printf("No matching trajectoy found for a keypoint of frame %d\n",nFrameNo);
                }


                //Computing the nearest neighbour matches with the first frame sift keypoints only if the nFrameNo>5 i.e. ignoring the first 5 keyframes
                float maxdistance = 1000000;
                float currentdistance;
                int nNearestTrajIndex = -1;
                for(size_t x = 0; x < vFirstFrameKeypts.size(); x++) {
                    // this trajectory started in the beginning and is yet to be assigned a hypothetical keypoint
                    if(vFirstFrameKeypts[x][128] == 0) {
                        currentdistance = CalTrajectoryDistance(vFirstFrameKeypts[x], keypts->descrip);

                        if(currentdistance < maxdistance) {
                            // printf("Distance : %g \n",currentdistance);
                            maxdistance = currentdistance;
                            nNearestTrajIndex = x;
                        }
                    }
                }
                //That it found a nearest neighbour && put threshhold condition
                if(nNearestTrajIndex != -1 &&  maxdistance < nDistThres) {
                    vFirstFrameKeypts[nNearestTrajIndex][128] = 1;
                    nMatchOrigin++;
                }

            }

            keypts = keypts->next;
        }//end of processing of keypoints for that loop i.e. time to update the counters of the trajectories and invalidate the trajectories(if req.) before the beginning of the next frame



        ////// Recording current keypoints /////////
        int nKeyCnt = 0;
        keypts = keypts_tmp;
        while (keypts) {
            vnKeyPrev.resize(nKeyCnt+1);
            vnKeyPrev[nKeyCnt].resize(133);
            for(int i = 0; i < nDescripSize; i++) vnKeyPrev[nKeyCnt][i] = keypts->descrip[i];

            vnKeyPrev[nKeyCnt][128] = keypts->row;
            vnKeyPrev[nKeyCnt][129] = keypts->col;
            vnKeyPrev[nKeyCnt][130] = keypts->scale;
            vnKeyPrev[nKeyCnt][131] = keypts->ori;
            vnKeyPrev[nKeyCnt][132] = keypts->fpyramidscale;

            nKeyCnt++;
            keypts = keypts->next;
        }



        //////////////////////////////////////////////////////Detecting and Removing Outliers using Displacement Vector
        vnDisplacement.resize(tracked);

        int nFont = CV_FONT_HERSHEY_SIMPLEX;
        double nFontSize = 0.4;
        char sText[128];
        sprintf(sText, "%d", nKeyCnt);
        cv::putText(cvm_match, "Total Keypoints:", cv::Point(nRw+nXDelta+100, 30), nFont, nFontSize, c_lemon, 1);
        cv::putText(cvm_match, sText , cv::Point(nRw+nXDelta+260, 30), nFont, nFontSize, c_lemon, 1);
        sprintf(sText, "%d", trackedold);
        cv::putText(cvm_match, "Tracked from old:" , cv::Point(nRw+nXDelta+100, 90), nFont, nFontSize, c_lemon, 1);
        cv::putText(cvm_match, sText , cv::Point(nRw+nXDelta+260, 90), nFont, nFontSize, c_lemon, 1);
        sprintf(sText, "%d", trackedprevious);
        cv::putText(cvm_match, "Tracked from previous:" , cv::Point(nRw+nXDelta+100, 60), nFont, nFontSize, c_lemon, 1);
        cv::putText(cvm_match, sText , cv::Point(nRw+nXDelta+260, 60), nFont, nFontSize, c_lemon, 1);
        sprintf(sText, "%d", nottracked);
        cv::putText(cvm_match, "Not Tracked:" , cv::Point(nRw+nXDelta+100, 120), nFont, nFontSize, c_lemon, 1);
        cv::putText(cvm_match, sText , cv::Point(nRw+nXDelta+260, 120), nFont, nFontSize, c_lemon, 1);
        sprintf(sText, "%d", nKeyRemoved);
        cv::putText(cvm_match, "Rejected:" , cv::Point(nRw+nXDelta+100, 150), nFont, nFontSize, c_lemon, 1);
        cv::putText(cvm_match, sText , cv::Point(nRw+nXDelta+260, 150), nFont, nFontSize, c_lemon, 1);
        sprintf(sText, "%d", (trackedold+trackedprevious)-nKeyRemoved);
        cv::putText(cvm_match, "Actually Tracked:" , cv::Point(nRw+nXDelta+100,180), nFont, nFontSize, c_lemon, 1);
        cv::putText(cvm_match, sText , cv::Point(nRw+nXDelta+260, 180), nFont, nFontSize, c_lemon, 1);
        sprintf(sText, "%g",(double((trackedold+trackedprevious)-nKeyRemoved)*100)/double(nKeyCnt));
        cv::putText(cvm_match, sText , cv::Point(nRw+nXDelta+150, 210), nFont, nFontSize, c_lemon, 1);
        cv::putText(cvm_match, "%" , cv::Point(nRw+nXDelta+260, 210), nFont, nFontSize, c_lemon, 1);


        //////////To generate the curve that will find the end of rotation
        if (nFrameNo == 1) {nMatchOriginFirst = nMatchOrigin; printf("Second with first: %d  File name %s\n",nMatchOrigin, sFname.data());}
        vFrameMatches.push_back(nMatchOrigin);


        /////////////////Updating flags for each trajectory
        for(size_t k = 0; k < vFirstFrameKeypts.size(); k++)
            vFirstFrameKeypts[k][128] = 0;

        for (int x = 0; x < int(vnTracker.size()); x++) {
            if(vnTracker[x][0][134] == 1) {     //valid
                vnTracker[x][0][137] = 0;
                if(vnTracker[x][0][135] == 1) {vnTracker[x][0][133] = 0; vnTracker[x][0][135] = 0;}
                else if(vnTracker[x][0][135] == 0) {vnTracker[x][0][133] += 1; if(vnTracker[x][0][133] > 9) vnTracker[x][0][134] = 0;}
            }
        }

        /////////////////Deciding when you need to stop the loop
        if(nFrameNo == nTotalFrames-1) {
            nFrameNrMax = -1;
            int nMaxMatches = 0;
            int nFrameMatchesSize = vFrameMatches.size();
            if (nFrameMatchesSize < 5){printf("ERROR!!!!\n");}
            for(int o = 0; o < nFrameMatchesSize; o++) {
                if (o < nFrameMatchesSize/2) continue;
                if (vFrameMatches[o] > nMaxMatches) {nMaxMatches = vFrameMatches[o]; nFrameNrMax = o + 2;}
            }
            if(nFrameNrMax != -1) {
                printf("Rotation ends at frame number %d\n",nFrameNrMax);
                bEndofRotation = true;
                printf("First Peak %d Second Peak%d\n", nMatchOriginFirst, nMaxMatches);
            }
        }


        if(bEndofRotation == true && bEndofStitching == false) {
            bEndofStitching = true;
        }
        ////////////////////End of Stitchingx
    }//end of frame loop which exectues for all frames as long as the end of rotation is not detected
    //////////////////////////////////////////////////////END OF TRACKING

    FreeKeypoints(keypts);
    DestroyAllResources();

}



/* Concatenate trajectories at both end of trajectory graph
 *
 * Input:
 * vnTracker - input trajectories
 * nTrjLBlank - size of blank of trajectory
 * nTrjLRes - size of reserve of trajectory (reserve is opposite of blank)
 * nInterval -
 * nTrjLMax, nTrjLMin - min and max of trajectory length
 * nTrjStitchDist - trajectory distance for stitching
 *
 * Output:
 * nFrameNrMax -
 * vnTrajectory - stitched trajectories
 */
void Stitching (std::vector<std::vector<std::vector<double> > > vnTracker, int nTrjLBlank, int nTrjLRes, int nInterval, int nTrjLMax, int nTrjLMin, double nTrjStitchDist,
                int &nFrameNrMax, std::vector<std::vector<std::vector<double> > > &vnTrajectory) {

    vnTrajectory = vnTracker;
    vnTrajectory[0][0][134] = 1;
    vnTrajectory[0][0][136] = 0;

    if (int(vnTrajectory[0].size()) > nTrjLMax) vnTrajectory[0][0][134] = 0;

    for (size_t ref = 1; ref < vnTracker.size(); ref++) {

        ////////* Reference Trajectory */////////////////
        if (!vnTrajectory[ref].size()) continue;                                                    // Does reference trajectory exsist?

        vnTrajectory[ref][0][134] = 0;                                                              // Set the reference trajectory as invalid at first
        vnTrajectory[ref][0][136] = 0;

        int nRefSize = vnTrajectory[ref].size();

        if (nRefSize < nTrjLRes) continue;                                                          // Too short Ref. trajectories are invalid
        if (vnTrajectory[ref][nRefSize-1][138] - vnTrajectory[ref][0][138] > nTrjLMax) continue;    // Too long Ref. trajectories are invalid

        vnTrajectory[ref][0][134] = 1;                                                              // The reference trajectory is valid

        for (size_t j = 1; j < vnTrajectory[ref].size(); j++) {
            if (vnTrajectory[ref][j][138] - vnTrajectory[ref][j-1][138] > nTrjLBlank) {
                vnTrajectory[ref][0][134] = 0;
                break;
            }
        }

        if (!vnTrajectory[ref][0][134]) continue;                                                   // The ref. trajectory must be valid
        if (vnTrajectory[ref][nRefSize-1][138] < nFrameNrMax-1 - nTrjLRes) continue;                // The ref. trajectory must reach until the end of rotation


        //bool flag = false;
        double mindist = 999;
        int nCandMatched = -1;
        for (size_t cand = 0; cand < ref; cand++) {
            ////////* Candidate Trajectory */////////////////
            if (!vnTrajectory[cand].size()) continue;                                               // Candidate trajectory must exsist
            int nCandSize = vnTrajectory[cand].size();

            if (vnTrajectory[cand][0][136] == 1) continue;                                          // Candidate trajectory is already stitched
            if (nCandSize < nTrjLRes) continue;                                                     // Too short candidate trajectories are neglected
            if (vnTrajectory[cand][nCandSize-1][138] - vnTrajectory[cand][0][138] > nTrjLMax) continue;   // Too long candidate trajectories are neglected
            if (nCandSize < nTrjLMin && vnTrajectory[cand][0][138] > nTrjLRes) continue;            // Cand. trajectories must start from eraly frame

            //if (vnTrajectory[ref][0][138] <= vnTrajectory[cand][nCandSize-1][138]) continue;
            if (vnTrajectory[ref][0][138] < vnTrajectory[cand][nCandSize-1][138] + nInterval) continue; // 1. ref. KeyPt must be after an interval from the last of cand. Key

            double nDist = CalTrajectoryDistance(vnTrajectory[ref][nRefSize-1], vnTrajectory[cand][0]);

            if (nDist > nTrjStitchDist) continue;
            if (nRefSize + nCandSize > nTrjLMax) continue;    // Sum of the ref. & cand. trajectory should not be too long

            if (nDist < mindist) {
                mindist = nDist;
                nCandMatched = cand;
            }

            //if (!flag) {printf("%5d: ", (int)ref); flag = true;}

            //printf("%5.3f (%4d, %2.0f, %2.0f) ", nDist, cand, vnTrajectory[ref][0][138], vnTrajectory[cand][nCandLast][138]);
        }

        if (nCandMatched != -1) {
//            vnTrajectory[nCandMatched].insert(vnTrajectory[nCandMatched].end(), vnTrajectory[ref].begin(), vnTrajectory[ref].end());
//            vnTrajectory[ref][0][134] = 0;
//            vnTrajectory[ref][0][136] = 1;
//            vnTrajectory[nCandMatched][0][134] = 1;
//            vnTrajectory[nCandMatched][0][136] = 0;

            vnTrajectory[ref].insert(vnTrajectory[ref].end(), vnTrajectory[nCandMatched].begin(), vnTrajectory[nCandMatched].end());
            vnTrajectory[nCandMatched][0][134] = 0;
            vnTrajectory[nCandMatched][0][136] = 1;
            vnTrajectory[ref][0][134] = 1;
            vnTrajectory[ref][0][136] = 0;

            for (size_t i = nRefSize; i < vnTrajectory[ref].size(); i++) {
                vnTrajectory[ref][i][138] += nFrameNrMax;
            }

        }
    }
}
