/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2013, NI Group, TU-Berlin.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 * * $Id: makelib.cpp 2013-11-30 00:00:00 Sahil Narang, Fritjof Wolf and JongHan Park
 *
 */

#include <iostream>

// ROS core
#include <ros/ros.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "opencv2/imgproc/imgproc_c.h"


#include "siftfast/siftfast.h"

#include "terminal_tools/parse.h"

#include "libmake_func.cpp"




//// 2D
int nCvCurrentkey = -1;


// --------------------
// -----Parameters-----
// --------------------
std::string sFileName;
std::string sPath;
std::string sExten;
std::string sOutputPath;
std::string sOutputFileName;
int nFrameCnt = 0;


//////// for the 2D SIFT Keypoint detection /////////////////////
int nSiftScales = 0;
double nSiftInitSigma = 0;
double nSiftPeakThresh = 0;
int nSiftImgOffset_x, nSiftImgOffset_y;
int nSiftImgWidth, nSiftImgHeight;

int nThresh = 0, nMinRect = 0;
int nFocusX = 0, nFocusY = 0, nFocusWidth = 0, nFocusHeight = 0;

bool bFlagModeSift = false;
double nDistThres;
// Global data
IplImage *cv_image_camera = NULL;
CvRect Sift_rect;
int nFeatureMode;
double nDispThresh;
int nSiftLibMode = 0;
int nBriefFlag = 0;




double nMatchThres = 0;
int nReinforce = 0;
int nKeyPosThres = 0;
int nTrjLBlank = 0;
int nTrjLRes = 0;
int nInterval = 0;
int nTrjLMin = 0;
int nTrjLMin2 = 0;
int nTrjLMax = 0;
double nTrjStitchDist = 0;

int nTrjGraphHeight = 0;



// initialize parameters
void ParameterInit(int argc, char** argv) {
    terminal_tools::parse_argument (argc, argv, "-path", sPath);
    terminal_tools::parse_argument (argc, argv, "-file", sFileName);
    terminal_tools::parse_argument (argc, argv, "-ext", sExten);
    terminal_tools::parse_argument (argc, argv, "-oppath", sOutputPath);
    terminal_tools::parse_argument (argc, argv, "-opfile", sOutputFileName);
    terminal_tools::parse_argument (argc, argv, "-siftsc", nSiftScales); if(nSiftScales == 0) nSiftScales = 3;
    terminal_tools::parse_argument (argc, argv, "-siftis", nSiftInitSigma); if(nSiftInitSigma == 0) nSiftInitSigma = 1.6;
    terminal_tools::parse_argument (argc, argv, "-siftpt", nSiftPeakThresh); if(nSiftPeakThresh == 0) nSiftPeakThresh = 0.04;

    terminal_tools::parse_argument (argc, argv, "-thr", nThresh); if(nThresh == 0) nThresh = 100;
    terminal_tools::parse_argument (argc, argv, "-mrect", nMinRect); if(nMinRect == 0) nMinRect = 100;
    terminal_tools::parse_argument (argc, argv, "-fox", nFocusX);
    terminal_tools::parse_argument (argc, argv, "-foy", nFocusY);
    terminal_tools::parse_argument (argc, argv, "-fow", nFocusWidth); if(nFocusWidth == 0) nFocusWidth = 300;
    terminal_tools::parse_argument (argc, argv, "-foh", nFocusHeight); if(nFocusHeight == 0) nFocusHeight = 300;
    terminal_tools::parse_argument (argc, argv, "-tgh", nTrjGraphHeight); if(nTrjGraphHeight == 0) nTrjGraphHeight = 900;

    terminal_tools::parse_argument (argc, argv, "-dthres", nDistThres); if(nDistThres == 0) nDistThres = 0.5;
    terminal_tools::parse_argument (argc, argv, "-mthres", nMatchThres); if(nMatchThres == 0) nMatchThres = 0.4;
    terminal_tools::parse_argument (argc, argv, "-tlblank", nTrjLBlank); if(nTrjLBlank == 0) nTrjLBlank = 7;
    terminal_tools::parse_argument (argc, argv, "-tlres", nTrjLRes); if(nTrjLRes == 0) nTrjLRes = 3;
    terminal_tools::parse_argument (argc, argv, "-tlinterv", nInterval); if(nInterval == 0) nInterval = 20;
    terminal_tools::parse_argument (argc, argv, "-tlmax", nTrjLMax); if(nTrjLMax == 0) nTrjLMax = 25;
    terminal_tools::parse_argument (argc, argv, "-tlmin", nTrjLMin); if(nTrjLMin == 0) nTrjLMin = 15;
    terminal_tools::parse_argument (argc, argv, "-tlmin2", nTrjLMin2); if(nTrjLMin2 == 0) nTrjLMin2 = 5;
    terminal_tools::parse_argument (argc, argv, "-tstdist", nTrjStitchDist); if(nTrjStitchDist == 0) nTrjStitchDist = 0.2;
    terminal_tools::parse_argument (argc, argv, "-featuremode", nFeatureMode); if(nFeatureMode == 0) nFeatureMode = 0;

    terminal_tools::parse_argument (argc, argv, "-reinf", nReinforce);
    terminal_tools::parse_argument (argc, argv, "-pthres", nKeyPosThres); if(nKeyPosThres == 0) nKeyPosThres = 50;

    terminal_tools::parse_argument (argc, argv, "-siftlibmode", nSiftLibMode); if(nSiftLibMode == 0) nSiftLibMode = 0;
    terminal_tools::parse_argument (argc, argv, "-dispthresh", nDispThresh); if(nDispThresh == 0) nDispThresh = 10;
    terminal_tools::parse_argument (argc, argv, "-brief", nBriefFlag);

    printf ("\n\n");
    printf ("===============================================================\n");
    printf ("** Parameter setting **\n");
    printf ("===============================================================\n");
    switch (nFeatureMode) {
    case 0: {
        printf ("Feature Mode: .............................. SIFT\n");
        printf ("2D SIFT - Scales: .......................... %5d\n", nSiftScales);
        printf ("2D SIFT - Initial Sigma: ................... %10.4f\n", nSiftInitSigma);
        printf ("2D SIFT - Peak thresh: ..................... %10.4f\n", nSiftPeakThresh);

        switch (nSiftLibMode) {
        case 0: {printf ("2D SIFT - Feature Set: ..................... Average of trajectory\n"); break;}
        case 1: {printf ("2D SIFT - Feature Set: ..................... A special member of trajectory\n"); break;}
        }
        break;}
    case 1: {
        printf ("Feature Mode: .............................. RGB Color Histogram\n");
        break;}
    case 2: {
        printf ("Feature Mode: ... No feature Calculation-Opponent Color Space\n");
        break;}
    case 3: {
        printf ("Feature Mode: .............................. Transformed Color Space Histogram\n");
        break;}
    case 4: {
        printf ("Feature Mode: .............................. Hue Sift features\n");
        break;}
    }

    printf ("===============================================================\n\n\n");
}


void MakeObjectMask (cv::Mat cvm_org, cv::Mat &cvm_mask) {
    cv::Mat cvm_gray(cvm_org.size(), CV_8UC1, cv::Scalar(0,0,0));
    cv::Mat cvm_binary(cvm_org.size(), CV_8UC1, cv::Scalar(0,0,0));

    cv::cvtColor(cvm_org, cvm_gray, CV_BGR2GRAY);

    cv::Mat out, tmp;
    tmp = cvm_gray(cvRect(nFocusX, nFocusY, nFocusWidth, nFocusHeight));
    out = cvm_binary(cvRect(nFocusX, nFocusY, nFocusWidth, nFocusHeight));
    cv::threshold(tmp, out, nThresh, 255, CV_THRESH_BINARY );

    tmp = cvm_binary(cvRect(nFocusX, nFocusY, nFocusWidth, nFocusHeight));
    out = cvm_mask(cvRect(nFocusX-1, nFocusY-1, nFocusWidth, nFocusHeight));
    cv::add(out, tmp, out);
    tmp = cvm_binary(cvRect(nFocusX, nFocusY, nFocusWidth, nFocusHeight));
    out = cvm_mask(cvRect(nFocusX, nFocusY-1, nFocusWidth, nFocusHeight));
    cv::add(out, tmp, out);
    tmp = cvm_binary(cvRect(nFocusX, nFocusY, nFocusWidth, nFocusHeight));
    out = cvm_mask(cvRect(nFocusX+1, nFocusY-1, nFocusWidth, nFocusHeight));
    cv::add(out, tmp, out);
    tmp = cvm_binary(cvRect(nFocusX, nFocusY, nFocusWidth, nFocusHeight));
    out = cvm_mask(cvRect(nFocusX-1, nFocusY, nFocusWidth, nFocusHeight));
    cv::add(out, tmp, out);
    tmp = cvm_binary(cvRect(nFocusX, nFocusY, nFocusWidth, nFocusHeight));
    out = cvm_mask(cvRect(nFocusX, nFocusY, nFocusWidth, nFocusHeight));
    cv::add(out, tmp, out);
    tmp = cvm_binary(cvRect(nFocusX, nFocusY, nFocusWidth, nFocusHeight));
    out = cvm_mask(cvRect(nFocusX+1, nFocusY, nFocusWidth, nFocusHeight));
    cv::add(out, tmp, out);
    tmp = cvm_binary(cvRect(nFocusX, nFocusY, nFocusWidth, nFocusHeight));
    out = cvm_mask(cvRect(nFocusX-1, nFocusY+1, nFocusWidth, nFocusHeight));
    cv::add(out, tmp, out);
    tmp = cvm_binary(cvRect(nFocusX, nFocusY, nFocusWidth, nFocusHeight));
    out = cvm_mask(cvRect(nFocusX, nFocusY+1, nFocusWidth, nFocusHeight));
    cv::add(out, tmp, out);
    tmp = cvm_binary(cvRect(nFocusX, nFocusY, nFocusWidth, nFocusHeight));
    out = cvm_mask(cvRect(nFocusX+1, nFocusY+1, nFocusWidth, nFocusHeight));
    cv::add(out, tmp, out);

    out.release(); tmp.release();

    //cv::imshow("Test", cvm_binary);
    //cv::imshow("Gray Image", cvm_gray);
    cvm_gray.release();
    cvm_binary.release();
}


void DrawRotationGraph (std::vector<int> vnMatchedKeyCnt, int nFrameNrMax, cv::Mat &graph) {

    int nWidth = graph.cols;
    int nHeight = graph.rows;
    int nOffsetX = 60;
    int nOffsetY = 10;
    graph = c_white;

    int nFont = CV_FONT_HERSHEY_SIMPLEX;

    char tmpNum[10];
    int currpt;

    //X axis
    for (int i = 0; i <= nFrameNrMax + 10; i = i+20) {
        currpt = nOffsetX+20 + i*(nWidth - nOffsetX)/(nFrameNrMax + 10);
        sprintf(tmpNum,"%d", i);
        cv::putText(graph, tmpNum , cv::Point(currpt, nHeight-10), nFont, 0.4, c_black, 1);
    }
    sprintf(tmpNum,"%d", nFrameNrMax); currpt = nOffsetX+20 + nFrameNrMax*(nWidth - nOffsetX)/(nFrameNrMax+10);
    cv::putText(graph, tmpNum , cv::Point(currpt, nHeight-10), nFont, 0.4, c_black, 1);
    cv::line(graph, cv::Point(currpt, 20), cv::Point(currpt, nHeight-40), c_violet, 1); //horizontal line

    //Y axis
    for (int j = 0; j <= nHeight-40; j = j+30) {
        currpt = nHeight-30 - j;
        sprintf(tmpNum, "%d", j);
        cv::putText(graph, tmpNum , cv::Point(10, currpt), nFont, 0.4, c_black, 1);
    }

    cv::Point oldpt;
    cv::Point newpt;
    oldpt.x = nOffsetX;
    oldpt.y = nHeight-40;
    for (size_t i = 0; i < vnMatchedKeyCnt.size(); i++) {
        newpt.x = nOffsetX+20 + i*(nWidth - nOffsetX)/(nFrameNrMax+10);
        newpt.y = nHeight-40 - vnMatchedKeyCnt[i];
        cv::line(graph, oldpt, newpt, c_blue, 1);
        oldpt.x = newpt.x;
        oldpt.y = newpt.y;
    }
    cv::line(graph, cv::Point(nOffsetX , nHeight-40),cv::Point(nWidth-10, nHeight-40), c_black, 2); //horizontal axe
    cv::line(graph, cv::Point(nOffsetX , nHeight-40),cv::Point(nOffsetX, nOffsetY), c_black, 2);  //Vertical axe
}


void DrawTrajectory (int nValidCnt, int nFrameNrMax, int nMaxStitchedFrame, std::vector<std::vector<std::vector<double> > > vnTrajectory) {
    int nThickness = nTrjGraphHeight/nValidCnt;
    if (!nThickness) nThickness = 1;
    else if (nThickness > 2) nThickness = 2;

    int nOffsetX = 60; int nOffsetY = 10;
    int nXInterval = 10; int nYInterval = 10;
    cv::Mat cvm_traj (vnTrajectory.size() + nOffsetY + 20, nMaxStitchedFrame*nXInterval + nOffsetX + 20, CV_8UC3);
    cv::Mat cvm_trajv (nValidCnt*nThickness + nOffsetY + 30, nMaxStitchedFrame*nXInterval + nOffsetX + 60, CV_8UC3);
    cvm_traj = cv::Scalar(255,255,255);
    cvm_trajv = cv::Scalar(255,255,255);


    int nCnt = 0;
    for (size_t i = 0; i < vnTrajectory.size(); i++) {
        int yy = i + nOffsetY;
        //cv::line(cvm_traj, cv::Point(nOffsetX, yy), cv::Point(nMaxStitchedFrame*nXInterval + nOffsetX, yy), cv::Scalar(192,192,192), 1); //horizontal axe
        if (vnTrajectory[i].size() && vnTrajectory[i][0][134] == 1) {
            if (vnTrajectory[i].size() == 1) continue;

            yy = nCnt*nThickness + nThickness/2 + nOffsetY;
            //cv::line(cvm_trajv, cv::Point(nOffsetX, yy), cv::Point(nMaxStitchedFrame*nXInterval + nOffsetX, yy), cv::Scalar(192,192,192), 1); //horizontal axe
            nCnt++;
        }
    }
    for (int j = 1; j < nMaxStitchedFrame+1; j++) {
        int xx = j*nXInterval + nOffsetX;
        cv::line(cvm_traj, cv::Point(xx, vnTrajectory.size() + nOffsetY), cv::Point(xx, nOffsetY), cv::Scalar(192,192,192), 1);  //Vertical axe
        cv::line(cvm_trajv, cv::Point(xx, nValidCnt*nThickness + nOffsetY), cv::Point(xx, nOffsetY), cv::Scalar(192,192,192), 1);  //Vertical axe
    }
    cv::line(cvm_traj, cv::Point(nOffsetX, vnTrajectory.size() + nOffsetY), cv::Point(nOffsetX, nOffsetY), c_black, 2);  //Vertical axe
    cv::line(cvm_traj, cv::Point(nOffsetX, vnTrajectory.size() + nOffsetY), cv::Point(nMaxStitchedFrame*nXInterval + nOffsetX, vnTrajectory.size() + nOffsetY), c_black, 2); //horizontal axe
    cv::line(cvm_trajv, cv::Point((nFrameNrMax-1)*nXInterval + nOffsetX, nValidCnt*nThickness + nOffsetY), cv::Point((nFrameNrMax-1)*nXInterval + nOffsetX, nOffsetY), c_violet, 1);  //Vertical axe
    cv::line(cvm_trajv, cv::Point(nOffsetX, nValidCnt*nThickness + nOffsetY), cv::Point(nOffsetX, nOffsetY), c_black, 2);  //Vertical axe
    cv::line(cvm_trajv, cv::Point(nOffsetX, nValidCnt*nThickness + nOffsetY), cv::Point((nMaxStitchedFrame+1)*nXInterval + nOffsetX, nValidCnt*nThickness + nOffsetY), c_black, 2); //horizontal axe

    char sText[128];
    int nFont = CV_FONT_HERSHEY_SIMPLEX;
    sprintf(sText,"End of rotation");
    cv::putText(cvm_trajv, sText, cv::Point((nFrameNrMax-1)*nXInterval + nOffsetX, (nValidCnt/5)*nThickness + nOffsetY), nFont, 0.4, c_violet, 1);

    sprintf(sText,"Frame of rotation");
    cv::putText(cvm_trajv, sText, cv::Point((nMaxStitchedFrame)*nXInterval + nOffsetX - 70, nValidCnt*nThickness + nOffsetY + 15), nFont, 0.4, c_black, 1);

    for (int i = 0; i < nMaxStitchedFrame; i += 10) {
        sprintf(sText,"%2d", i);
        cv::putText(cvm_trajv, sText, cv::Point(i*nXInterval + nOffsetX - 8, nValidCnt*nThickness + nOffsetY + 15), nFont, 0.4, c_black, 1);
    }



    nCnt = 0;
    for (size_t i = 0; i < vnTrajectory.size(); i++) {
        int yy = i + nOffsetY;
        for (size_t j = 0; j < vnTrajectory[i].size(); j++) {
            int xx = j*nXInterval + nOffsetX;
            if (vnTrajectory[i][j].size()) {
                xx = vnTrajectory[i][j][138]*nXInterval + nOffsetX;
                cv::line(cvm_traj, cv::Point(xx, yy), cv::Point(xx + nXInterval-1, yy), c_magenta, 1);
            }
        }
        if (vnTrajectory[i].size() && vnTrajectory[i][0][134] == 1) {
            if (vnTrajectory[i].size() == 1) continue;

            yy = nCnt*nThickness + nThickness/2 + nOffsetY;
            for (size_t j = 0; j < vnTrajectory[i].size(); j++) {
                int xx = j*nXInterval + nOffsetX;
                if (vnTrajectory[i][j].size()) {
                    xx = vnTrajectory[i][j][138]*nXInterval + nOffsetX;
                    cv::rectangle(cvm_trajv, cv::Point(xx-1, yy-1), cv::Point(xx+1, yy+1), c_black, 1);
                    cv::rectangle(cvm_trajv, cv::Point(xx, yy), cv::Point(xx, yy), c_white, 1);

                    if (j > 0 && vnTrajectory[i][j-1][138] == vnTrajectory[i][j][138] - 1) {
                        cv::line(cvm_trajv, cv::Point(xx-nXInterval+2, yy), cv::Point(xx-2, yy), c_blue, 1);
                    }
                }
            }
            nCnt++;
        }
    }

    //for (size_t i = 0; i < vnTrajectory[0].size(); i++) printf("%3.0f ", vnTrajectory[0][i][138]); printf("\n");

    cv::imshow("All Trajectories", cvm_traj);
    cv::imshow("Final Trajectories", cvm_trajv);

    nCvCurrentkey = cv::waitKey();
    if (nCvCurrentkey == 115) cv::imwrite("Trajectory.bmp", cvm_trajv);
}


void DrawSiftMatching (cv::Mat input, cv::Mat &output, int tlx, int tly, int brx, int bry, int nXDelta, int nYDelta)
{
    cv::Mat cvm_clone;
    output.copyTo(cvm_clone);
    output = c_black;

    /////copy previous image from pos(nXDelta,nYDelta to 0,0)
    int nY_curr = nYDelta;
    for(int j = nYDelta; j < nYDelta + (bry-tly); j++) {
        int nX_curr = 0;
        for(int i = nXDelta; i < nXDelta + (brx-tlx); i++) {
            output.at<cv::Vec3b>(nY_curr, nX_curr) = cvm_clone.at<cv::Vec3b>(j,i);
            nX_curr++;
        }
        nY_curr++;
    }

    /////copy current image
    nY_curr = nYDelta;
    for(int j = tly; j < bry; j++) {
        int nX_curr = nXDelta;
        for(int i = tlx; i < brx; i++) {
            output.at<cv::Vec3b>(nY_curr, nX_curr) = input.at<cv::Vec3b>(j,i);
            nX_curr++;
        }
        nY_curr++;
    }
}




int
main (int argc, char** argv)
{
    ///////////Sahil
    std::vector<std::vector<std::vector<double> > > vnTracker;
    std::vector<std::vector<std::vector<double> > > vnTrajectory;
    std::vector<std::vector <double> > vFirstFrameKeypts;
    bool bEndofStitching=false;
    bool bEndofRotation=false;
    int NNMatchesFirst;
    std::vector<int> vFrameMatches;
    int nFrameNrMax;



    std::vector<int> vnMatchedKeyCnt;
    std::vector<std::vector<double> > vnKeyPrev;


    //////////// Set parameters as default ///////////////////////////////
    ParameterInit(argc, argv);
    if(argc<2){
        printf("Usage: main <image-file-name>\n\7");
        exit(0);
    }

    printf ("Making a library of the object %s\n\n", sFileName.data());
    printf ("Getting the rectangle..");
    std::string sFname;
    sExten = "." + sExten;
    char sFNum[10];

    cv::Mat cvm_org;
    int nFrameLimit = 10000;
    int nRx1 = 2000, nRy1 = 2000, nRx2 = 0, nRy2 = 0;

    //cv::namedWindow("Gray Image"); cvMoveWindow("Gray Image", 200, 200);
    //cv::namedWindow("test"); cvMoveWindow("test", 300, 300);
    cv::namedWindow("Image"); cvMoveWindow("Image", 50, 50);
    cv::namedWindow("Test Mask"); cvMoveWindow("Test Mask", 100, 100);





//    while (dirp) {
//        if ((dp = readdir(dirp)) != NULL) {
//            string sFile = dp->d_name;
//            size_t pos = sFile.find(".");
//            string sExt = sFile.substr(pos+1);

//            if(sExt.compare("bmp")==0 || sExt.compare("jpeg")==0 || sExt.compare("jpg")==0 || sExt.compare("tiff")==0 || sExt.compare("tif")==0 || sExt.compare("png")==0) {
//                string sFileName = sInputPath + sFile;
//                cout << "Opening File : " << sFileName << "\n\n";
//                const char *c = sFileName.c_str();
//                IplImage *cv_input = cvLoadImage(c);


    ////////////////////////////////////////////////////////////////////////////////
    ///////////////** 1. rotation: to get a bounding rectangle **///////////////////
    ////////////////////////////////////////////////////////////////////////////////
    bool bFlagFirst = true;
    for (int i = 0; i < nFrameLimit; i++) {
        sprintf(sFNum, "%06i", i);
        sFname = sPath + sFileName + "_" + sFNum + sExten;

        cvm_org = cv::imread(sFname.data(), CV_LOAD_IMAGE_COLOR);
        if(cvm_org.data) {

            if (bFlagFirst) {
                if (nFocusX+nFocusWidth > cvm_org.cols-4 || nFocusY+nFocusHeight > cvm_org.rows-4) {
                    nFocusX = 2; nFocusY = 2; nFocusWidth = cvm_org.cols-4; nFocusHeight = cvm_org.rows-4;
                }
                bFlagFirst = false;
            }

            printf (".");
            nFrameCnt++;


            cv::Mat cvm_mask(cvm_org.size(), CV_8UC1, cv::Scalar(0,0,0));
            MakeObjectMask(cvm_org, cvm_mask);
            cv::imshow("Test Mask", cvm_mask);


            std::vector<std::vector<cv::Point> > contours;
            std::vector<cv::Vec4i> hierarchy;
            cv::findContours(cvm_mask, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point(0,0));
            cvm_mask.release();

            int res = 0;
            cv::Rect rect;
            for (size_t i = 0; i < contours.size(); i++) {
                rect = cv::boundingRect(contours[i]);
                int area = rect.width*rect.height;
                if (area < nMinRect)
                    continue;

                int x2 = rect.x + rect.width, y2 = rect.y + rect.height;
                if (rect.x < nRx1) nRx1 = rect.x;
                if (rect.y < nRy1) nRy1 = rect.y;
                if (x2 > nRx2) nRx2 = x2;
                if (y2 > nRy2) nRy2 = y2;
                cv::rectangle(cvm_org, cv::Point(nRx1 - res, nRy1 - res), cv::Point(nRx2 + res, nRy2 + res + 10), c_red, 1);
                //cv::rectangle(cvm_org, cv::Point(nRx1 - res, nRy1 - res), cv::Point(nRx2 + res, nRy2 + res + 10), c_red, 1);
            }//for

            cv::imshow("Image", cvm_org);

            nCvCurrentkey = cv::waitKey(3);
        }
    }//for

    nCvCurrentkey = cv::waitKey();
    cv::destroyWindow("Test Mask");
    //cv::destroyWindow("Gray Image");
    //cv::destroyWindow("Test");
    printf (" O.K.\n\n");



    ////////////////////////////////////////////////////////////////////////////////
    ///////////////** 2. rotation: to get the period **/////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    int nFrameNo = 0;
//    std::vector<std::vector<double> > vnKeyFirst;
//    for (int i = 0; i < nFrameLimit; i++) {
//        sprintf(sFNum, "%06i", i);
//    sFname = sPath + sFileName + "_" + sFNum + sExten;

//        cvm_org = cv::imread(sFname.data(), CV_LOAD_IMAGE_COLOR);

//        if(cvm_org.data) {
//            Keypoint keypts;
//            GetSiftKeypoints(cvm_org, nSiftScales, nSiftInitSigma, nSiftPeakThresh, nRx1, nRy1, nRx2-nRx1+1, nRy2-nRy1+1, keypts);

//            int nKeyCnt = 0;
//            while (keypts) {
//                if (nFrameNo == 0) {
//                    vnKeyFirst.resize(vnKeyFirst.size()+1);
//                    vnKeyFirst[nKeyCnt].resize(nDescripSize);
//                    for (int ds = 0; ds < nDescripSize; ds++) vnKeyFirst[nKeyCnt][ds] = keypts->descrip[ds];
//                }
//                cv::rectangle(cvm_org, cv::Point(keypts->row + nRx1-2 ,keypts->col + nRy1-2), cv::Point(keypts->row + nRx1+2 ,keypts->col + nRy1+2), c_blue, 1);

//                nKeyCnt++;
//                keypts = keypts->next;
//            }
//            nFrameNo++;

//            cv::imshow("Image", cvm_org);
//            nCvCurrentkey = cv::waitKey(3);
//        }

//    }
//    nCvCurrentkey = cv::waitKey();




    //////////////////////////////////////////////////////////////////////////////////////
    ///////////////** 3. rotation: to generate key-point trajectories **//////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    cv::Mat cvm_match_org(cvSize(900, nRy2-nRy1 + 60), CV_8UC3);
    cvm_match_org = c_black;
    cv::namedWindow("Matching SIFT"); cvMoveWindow("Matching SIFT", 100, 200);

    nFrameNo = 0;
    printf ("Extracting features of the object %s.....", sFileName.data());
    std::string sOPExt=".yaml";
    sOutputFileName = sOutputPath + sOutputFileName + sOPExt;

    for (int i = 0; i < nFrameLimit; i++) {

        int nFont = CV_FONT_HERSHEY_SIMPLEX;

        sprintf(sFNum, "%06i", i);
        sFname = sPath + sFileName + "_" + sFNum + sExten;

        cvm_org = cv::imread(sFname.data(), CV_LOAD_IMAGE_COLOR);
        if(cvm_org.data) {
            /////Testing SIFT Matching

            int nXDelta = nRx2-nRx1+1 + 200;
            DrawSiftMatching(cvm_org, cvm_match_org, nRx1, nRy1, nRx2, nRy2, nXDelta, 0);
            cv::putText(cvm_match_org, "Previous Frame", cv::Point(0, (nRy2-nRy1+1)+30), nFont, 0.8, c_lemon, 1);
            cv::putText(cvm_match_org, "Current Frame", cv::Point(nXDelta, (nRy2-nRy1+1)+30), nFont, 0.8, c_lemon, 1);
            cv::Mat cvm_match;
            cvm_match_org.copyTo(cvm_match);
            vnMatchedKeyCnt.resize(nFrameNo + 1);

            //Sift Trajectory Calculation
            if(nFeatureMode == 0) {
                Trajectory(cvm_org, nRx1, nRy1, nRx2-nRx1+1, nRy2-nRy1+1, nFrameNo, nFrameCnt, vnTracker, nDistThres, nSiftScales, nSiftInitSigma,
                           nSiftPeakThresh, bEndofRotation, bEndofStitching, NNMatchesFirst, vFrameMatches, nFrameNrMax,
                           nXDelta, 0, 0, 0, nDispThresh, nFeatureMode, sFname, vFirstFrameKeypts,
                           nMatchThres, nReinforce, nKeyPosThres, nTrjLBlank, vnKeyPrev, vnMatchedKeyCnt, cvm_match);
                nFrameNo++;
            }

            cv::imshow("Matching SIFT", cvm_match);
            if(nBriefFlag!=0) nCvCurrentkey = cv::waitKey();
            ///////////////////////////////////////////////End of Feature Calculation
            if(nFeatureMode==0)
                cv::imshow("Image", cvm_org);

            nCvCurrentkey = cv::waitKey(10);
            cvm_match.release();
        }
        cvm_org.release();
    }





    nFrameNrMax = 42;

    //////////////////////////////////////////////////////////////////////////////////////
    ///////////////** Extract valid trajectories and save it **///////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////

    //////////////Saving Library File for SIFT Features

    Stitching(vnTracker, nTrjLBlank, nTrjLRes, nInterval, nTrjLMax, nTrjLMin, nTrjStitchDist, nFrameNrMax, vnTrajectory);


    //////////// Extracting valid trajectories //////////////////////////////////////
    for (size_t i = 0; i < vnTrajectory.size(); i++) {
        if (!vnTrajectory[i].size()) continue;                                            // Does the trajectory exsist?
        if (int(vnTrajectory[i].size()) < nTrjLMin2) vnTrajectory[i][0][134] = 0;         // Short trajectory is invalid
        if (int(vnTrajectory[i].size()) > nTrjLMax) vnTrajectory[i][0][134] = 0;          // Too long trajectory is invalid
        if (vnTrajectory[i][0][136] == 1) vnTrajectory[i][0][134] = 0;                    // Stitched trajectory is invalid
    }

    int nValidCnt = 0, nMaxStitchedFrame = 0;
    int nTotalSift = 0;
    for (size_t i = 0; i < vnTrajectory.size(); i++) {
        if(vnTrajectory[i].size() && vnTrajectory[i][0][134] == 1) {
            nValidCnt++;
            if (vnTrajectory[i][vnTrajectory[i].size()-1][138] > nMaxStitchedFrame) nMaxStitchedFrame = vnTrajectory[i][vnTrajectory[i].size()-1][138];
            for (size_t j=0; j < vnTrajectory[i].size();j++) nTotalSift++;
        }
    }


    printf("Total Number of valid trajectories at the end %d\n", nValidCnt);
    printf("Total Frames %d\n ", nFrameNrMax);

    printf("\nSaving Trajectories to disk\n\n");




    cv::Mat mFeatureSet(nValidCnt, 133, CV_32FC1);
    int n = 0;
    for(size_t i =0; i < vnTrajectory.size(); i++) {
        if (!vnTrajectory[i].size() || vnTrajectory[i][0][134] != 1) continue;

        switch (nSiftLibMode) {
        case 0: {       // Save average key-point-values of trajectories
            for(int k = 0; k < 133; k++) {
                float nFeatureVal = 0;
                int nFeatureCnt = 0;
                for(size_t j = 0; j < vnTrajectory[i].size(); j++) {
                    nFeatureVal += vnTrajectory[i][j][k];
                    nFeatureCnt++;
                }
                mFeatureSet.at<float>(n,k) = nFeatureVal / float(nFeatureCnt);
            }
            break;
        }
        case 1: {       // Save first key-points of trajectories
            for(int k = 0; k < 128; k++) {
                int num = 0;
                mFeatureSet.at<float>(n,k) = vnTrajectory[i][num][k];
            }
            break;
        }
        case 2: {       // Save key-points in the middle from trajectories
            for(int k = 0; k < 128; k++) {
                int num = vnTrajectory[i].size()/2;
                mFeatureSet.at<float>(n,k) = vnTrajectory[i][num][k];
            }
            break;
        }
        }
        n++;
    }

    printf("Rotation ends at frame number %d\t", nFrameNrMax);
    cv::FileStorage fs(sOutputFileName, cv::FileStorage::WRITE);
    fs << "TestObjectFeatureVectors" << mFeatureSet;
    fs.release();
    printf("\nTotal Features Computed : %d Total Trajectories Saved : %d %d\n", nTotalSift, nValidCnt, int(vnTrajectory.size()));

    cv::destroyWindow("Image");
    //Sift library file */saved

    printf (" \nO.K.\n\n\n");




    int nMaxKeyCnt = 0;
    for (int i = 0; i < nFrameNrMax; i++) {
        if (vnMatchedKeyCnt[i] > nMaxKeyCnt) nMaxKeyCnt = vnMatchedKeyCnt[i];
    }
    cv::Mat cvm_rgraph(cvSize(640,nMaxKeyCnt + 70), CV_8UC3);
    DrawRotationGraph(vnMatchedKeyCnt, nFrameNrMax, cvm_rgraph);
    //DrawRotationGraph(vFrameMatches, nFrameNrMax, cvm_rgraph);
    DrawTrajectory (nValidCnt, nFrameNrMax, nMaxStitchedFrame, vnTrajectory);

    cv::namedWindow("Rotation Graph"); cvMoveWindow("Rotation Graph", 300, 300);
    cv::imshow("Rotation Graph", cvm_rgraph);


    cv::destroyWindow("Matching SIFT");
    cv::destroyWindow("All Trajectories");
    cv::destroyWindow("Final Trajectories");
    cv::destroyWindow("Rotation Graph");

    cvm_org.release();
    return (0);
} //end of main


