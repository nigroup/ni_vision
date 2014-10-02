/*
 * Function for attention and recognition
 */



// Fast SIFT Library
//#include "siftfast/siftfast.h"

///////////
//#include "flann/flann.h"
#include "func_recognition_flann.hpp"

#define PI 3.1415926536


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




/* Calculating delta scale orientation (by Sahil)
 *
 * Input:
 * input_idx - indices of input SIFT keypoints
 * vnDeltaScale, vnDeltaOri - delta scale, delta orientation of input keypoints
 * nDeltaBinNo - number of bins
 * nMaxDeltaOri, nMinDeltaOri - range of delta orientation
 * T_orient, T_scale -
 *
 * Output:
 * nDeltaScale - output scale (think of an "average value")
 * nTruePositives - count of true-positive keypoints
 * output_flag - flag for true-positive keypoints
 */
void CalcDeltaScaleOri(std::vector<int> input_idx, std::vector<double> vnDeltaScale, std::vector<double> vnDeltaOri, double& nDeltaScale,
                       int nDeltaBinNo, double nMaxDeltaOri, double nMinDeltaOri, double T_orient, double T_scale, int& nTruePositives, std::vector<bool>& output_flag)
{
    //Put nDeltaBinNo = 12
    //int keycount = 0;
    std::vector <int>  vnHist(nDeltaBinNo,0);
    double nBinWidth1D = (2*PI) / double(nDeltaBinNo);
    double nLow1D = (-PI);
    int tmpmax = 0;
    double tmpmaxhigh = 0;
    for(size_t currkey = 0; currkey < vnDeltaOri.size(); currkey++)
    {
        double currHigh1D = nLow1D + nBinWidth1D;
        bool breakflag1D = false;
        for(int currBin1D = 0; currBin1D < nDeltaBinNo; currBin1D++)
        {
            if(vnDeltaOri[currkey] <= currHigh1D )
            {
                vnHist[currBin1D]++;
                //if(currBin1D>0) vnHist[currBin1D-1]++;
                breakflag1D = true;
                if(vnHist[currBin1D] > tmpmax)
                {
                    tmpmax = vnHist[currBin1D];
                    tmpmaxhigh = currHigh1D;
                }
                //                if(currBin1D>0 && vnHist[currBin1D-1]>tmpmax)
                //                {
                //                    tmpmax=vnHist[currBin1D-1];
                //                    tmpmaxhigh=currHigh1D-nBinWidth1D;
                //                }

            }
            if(breakflag1D == true) break;
            currHigh1D += nBinWidth1D;
        } //end of 1D
    } //end of current key
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////Histogram Created, Now printing values
    double tmpmaxlow=tmpmaxhigh-nBinWidth1D;

    ////////////////////Delta Ori Range found
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////Finding TruePositives
    std::vector<double> vnRectDeltaScale(vnDeltaScale.size(),0);
    int index_rect=0;
    std::vector<int> output_idx(vnDeltaOri.size(), 0);
    for(size_t k=0; k<vnDeltaOri.size(); k++)
    {
        if(vnDeltaOri[k]>= (tmpmaxlow-T_orient) && vnDeltaOri[k] <= (tmpmaxhigh + T_orient))  //for each keypoint that satisifies the delta ori range,1) store its delta scale in vnRectDeltaScale 2) mark it as true in keypts list 3) store index in outputidx
        {
            vnRectDeltaScale[index_rect]=vnDeltaScale[k];
            output_flag[input_idx[k]] = true;
            output_idx[index_rect++] = input_idx[k];
        }
    }
    vnRectDeltaScale.resize(index_rect);
    std::vector<double> vnRectDeltaScale_tmp = vnRectDeltaScale;
    std::sort(vnRectDeltaScale_tmp.begin(), vnRectDeltaScale_tmp.end());
    //    for (int i = 0; i < index_rect; i++) {
    //        printf("%9.3f", vnRectDeltaScale[i]);
    //    }
    //    printf("\n");
    //    for (int i = 0; i < index_rect; i++) {
    //        printf("%9.3f", vnRectDeltaScale_tmp[i]);
    //    }
    //    printf("\n");
    ///////Find Median Delta Scale of Qualified keypts
    if(index_rect) {
        if(index_rect % 2 == 0)
            nDeltaScale = double((vnRectDeltaScale_tmp[(index_rect/2)-1] + vnRectDeltaScale_tmp[index_rect/2])/2);
        else
            nDeltaScale = vnRectDeltaScale_tmp[index_rect/2];
    }


    for(size_t index = 0; index < vnRectDeltaScale.size(); index++)
    {
        if(vnRectDeltaScale[index]>= (nDeltaScale-T_scale) && vnRectDeltaScale[index]<= (nDeltaScale+T_scale)) nTruePositives++;
        else output_flag[output_idx[index]] = false;
    }

//    printf("\t\tMedian Scale: %g Rectified Scale Range [ %g , %g ]\n",nDeltaScale,nDeltaScale-T_scale,nDeltaScale+T_scale);
//    printf("\t\tOriginal Matches :%d  Orientation matches: %d  Scale Matches(final matches) :%d\n", (int)vnDeltaScale.size(), (int)vnRectDeltaScale.size(),nTruePositives);
}





/* Swap object surface in the memory and all properties which belong to the objects
 *
 * Output:
 * vbProtoCand - candidate flag for object surfaces (true, then candidate for attention)
 * vnProtoIdx - indices for object surfaces
 * vnProtoPtsCnt - count of pixels for object surfaces
 * mnProtoPtsIdx - indices of pixels for object surfaces
 * stProto - properties of object surfaces (ProtoProp is self-defined struct)
 * vnProtoFound - found flag for objects surfaces
 * nProtoCnt - count of object surfaces
 * vnTmpProtoDiff - color differences of objects surfaces (to target object)
 */
void SwapMemory (std::vector<bool> &vbProtoCand, std::vector<int> &vnProtoIdx, std::vector<int> &vnProtoPtsCnt, std::vector<std::vector<int> > &mnProtoPtsIdx, ProtoProp &stProto,
                 std::vector<int> &vnProtoFound, int nProtoCnt, std::vector<std::vector<float> > &vnTmpProtoDiff)
{
    std::vector<std::vector<float> > vnTmpProtoDiff2 = vnTmpProtoDiff;
    std::sort(vnTmpProtoDiff[0].begin(), vnTmpProtoDiff[0].end());
    std::vector<int> vTmpIdx(nProtoCnt, 0);
    for (int i = 0; i < nProtoCnt; i++) {
        for (int j = 0; j < nProtoCnt; j++) {
            if (vnTmpProtoDiff[0][i] == vnTmpProtoDiff2[0][j]) vTmpIdx[i] = j;
        }
    }


    std::vector<bool> vbAaaProtoCand = vbProtoCand;
    std::vector<int> vnAaaProtoIdx = vnProtoIdx;
    std::vector<int> vnAaaProtoPtsCnt = vnProtoPtsCnt;
    std::vector<std::vector<int> > mnAaaProtoPtsIdx = mnProtoPtsIdx;
    std::vector<std::vector<int> > mnAaaProtoRect = stProto.mnRect;
    std::vector<std::vector<int> > mnAaaProtoRCenter = stProto.mnRCenter;
    std::vector<std::vector<float> > mnAaaProtoCubic = stProto.mnCubic;
    std::vector<std::vector<float> > mnAaaProtoCCenter = stProto.mnCCenter;
    std::vector<std::vector<float> > mnAaaProtoColorHist = stProto.mnColorHist;
    std::vector<int> vnAaaProtoLength = stProto.vnLength;
    std::vector<int> vnAaaProtoMemoryCnt = stProto.vnMemoryCnt;
    std::vector<int> vnAaaProtoStableCnt = stProto.vnStableCnt;
    std::vector<int> vnAaaProtoDisapCnt = stProto.vnDisapCnt;
    std::vector<int> vnAaaProtoFound = vnProtoFound;

    for (int i = 0; i < nProtoCnt; i++) {
        vbProtoCand[i] = vbAaaProtoCand[vTmpIdx[i]];
        vnProtoIdx[i] = vnAaaProtoIdx[vTmpIdx[i]];
        vnProtoPtsCnt[i] = vnAaaProtoPtsCnt[vTmpIdx[i]];
        mnProtoPtsIdx[i] = mnAaaProtoPtsIdx[vTmpIdx[i]];
        stProto.mnRect[i] = mnAaaProtoRect[vTmpIdx[i]];
        stProto.mnRCenter[i] = mnAaaProtoRCenter[vTmpIdx[i]];
        stProto.mnCubic[i] = mnAaaProtoCubic[vTmpIdx[i]];
        stProto.mnCCenter[i] = mnAaaProtoCCenter[vTmpIdx[i]];
        stProto.mnColorHist[i] = mnAaaProtoColorHist[vTmpIdx[i]];
        stProto.vnLength[i] = vnAaaProtoLength[vTmpIdx[i]];
        stProto.vnMemoryCnt[i] = vnAaaProtoMemoryCnt[vTmpIdx[i]];
        stProto.vnStableCnt[i] = vnAaaProtoStableCnt[vTmpIdx[i]];
        stProto.vnDisapCnt[i] = vnAaaProtoDisapCnt[vTmpIdx[i]];
        vnProtoFound[i] = vnAaaProtoFound[vTmpIdx[i]];
    }
}




/* Resetting properties of object surfaces
 *
 * Input:
 * nObjsNrLimit - maximum number of object surfaces that can be processed
 * nTrackHistoBin_max - (number of bins)^number of channels
 * nRecogRtNr -
 *
 * Output:
 * vnProtoIdx - indices for object surfaces
 * vnProtoPtsCnt - count of pixels for object surfaces
 * mnProtoPtsIdx - indices of pixels for object surfaces
 * stProto - properties of object surfaces (ProtoProp is self-defined struct)
 * vnProtoFound - found flag for objects surfaces
 * nProtoCnt - count of object surfaces
 * nProtoNr - number of current object surface
 * nFoundCnt - count of found object surfaces
 * nFoundNr - number of found object surface
 * vnRecogRating - recognition rating (like a priority) of object surfaces
 */
void ResetMemory (int nObjsNrLimit, int nTrackHistoBin_max, int nRecogRtNr,
                  std::vector<int> &vnProtoIdx, std::vector<int> &vnProtoPtsCnt, std::vector<std::vector<int> > &mnProtoPtsIdx, ProtoProp &stProto,
                  std::vector<int> &vnProtoFound, int &nProtoCnt, int &nProtoNr, int &nFoundCnt, int &nFoundNr, std::vector<int> &vnRecogRating)
{
    vnProtoIdx.assign(nObjsNrLimit, 0);
    vnProtoPtsCnt.assign(nObjsNrLimit, 0);
    mnProtoPtsIdx.assign(nObjsNrLimit, std::vector<int>(0, 0));
    stProto.mnRCenter.assign(nObjsNrLimit, std::vector<int>(2, 0));
    stProto.mnCCenter.assign(nObjsNrLimit, std::vector<float>(3, 0));
    stProto.mnRect.assign(nObjsNrLimit, std::vector<int>(4, 0));
    stProto.mnCubic.assign(nObjsNrLimit, std::vector<float>(6, 0));
    stProto.mnColorHist.assign(nObjsNrLimit, std::vector<float>(nTrackHistoBin_max, 0));
    stProto.vnLength.assign(nObjsNrLimit, 0);
    stProto.vnMemoryCnt.assign(nObjsNrLimit, 0);
    stProto.vnStableCnt.assign(nObjsNrLimit, 0);
    stProto.vnDisapCnt.assign(nObjsNrLimit, 0);
    vnProtoFound.assign(nObjsNrLimit, 0);

    nProtoCnt = 0;
    nProtoNr = 0;
    nFoundCnt = 0;
    nFoundNr = 0;
    vnRecogRating.assign(nRecogRtNr, 0);
}




/* Selection of a candidate object surface
 *
 * Input:
 * nProtoCnt - count of candidate object surfaces
 * vbProtoCand - flag for candidate object surfaces
 *
 * Output:
 * vProtoFound - flag for found candidate object surfaces
 * nCandID - ID of selected candidate object surface
 */
void SelRecognition_Pre (int nProtoCnt, std::vector<bool> vbProtoCand, std::vector<int> &vProtoFound, int &nCandID) {

    for (int i = 0; i < nProtoCnt; i++) {
        if (!vbProtoCand[i]) continue;

        // vProtoFound.... 0: not matched, 1:not matched, 2: matched, 3: matched
        if (vProtoFound[i] < 2) {
            nCandID = i;
            break;
        }
    }
}




/* Extraction of pixel indices for the selected candidate object surface from original image
 * (before that the indices are from the downsampled image)
 *
 * Input:
 * nCandID - ID of selected candidate object surface
 * nImgScale - ratio between original and downsampled image
 * nDsWidth - width of downsampled image
 * vnProtoPtsCnt - count of pixel which belong to the object surface (of downsampled image)
 * cvm_rgb_org - original image
 * mnProtoPtsIdx - indices of pixel which belong to the object surfae (of downsampled image)
 *
 * Output:
 * cvm_cand_tmp - image of candidate object surface
 * vnIdxTmp - indices of pixel which belong to the object surfae (of original image)
 */
void SelRecognition_1 (int nCandID, int nImgScale, int nDsWidth, std::vector<int> vnProtoPtsCnt, cv::Mat cvm_rgb_org,
                       cv::Mat &cvm_cand_tmp, std::vector<std::vector<int> > mnProtoPtsIdx, std::vector<int> &vnIdxTmp) {
    int pts_cnt = 0;
    int xx, yy;
    if (nImgScale > 1) {
        int idx;
        for (int j = 0; j < vnProtoPtsCnt[nCandID]; j++) {
            GetPixelPos(mnProtoPtsIdx[nCandID][j], nDsWidth, xx, yy);
            xx = xx*nImgScale; yy = yy*nImgScale;

            for (int mm = 0; mm < nImgScale; mm++) {
                if (yy-mm < 0) continue;
                for (int nn = 0; nn < nImgScale; nn++) {
                    if (xx-nn < 0) continue;
                    //if (xx-nn < 0xx+nn >= cvm_rgb_org.cols) continue;
                    idx = (yy-mm) * nDsWidth*nImgScale + xx-nn;
                    //vnIdxTmp.resize(pts_cnt++);
                    vnIdxTmp[pts_cnt++] = idx;
                }
            }
        }
    }
    else {vnIdxTmp = mnProtoPtsIdx[nCandID]; pts_cnt = (int)vnProtoPtsCnt[nCandID];}

    //int xxx, yyy;
    for (int i = 0; i < pts_cnt; i++) {
        cvm_cand_tmp.data[vnIdxTmp[i]*3] = cvm_rgb_org.data[vnIdxTmp[i]*3];
        cvm_cand_tmp.data[vnIdxTmp[i]*3+1] = cvm_rgb_org.data[vnIdxTmp[i]*3+1];
        cvm_cand_tmp.data[vnIdxTmp[i]*3+2] = cvm_rgb_org.data[vnIdxTmp[i]*3+2];
    }
}



/* SIFT keypoint matching with FLANN (fast library for approximate nearest neighbor) to find target object
 *
 * Input:
 * tcount -
 * nFlannKnn - parameter of FLANN
 * nFlannLibCols_sift - dimension of library which stores the sift trajectories
 * nFlannMatchFac - merge factor (parameter of FLANN)
 * mnSiftExtraFeatures - matrix which stores sift trajectories (from training)
 * FlannIdx_Sift - FLANN index (type is from flann library)
 * FlannParam - flann parameters (struct from flann lib)
 * keypts - input SIFT keypoints
 *
 * Output:
 * nKeyptsCnt - count of matched SIFT keypoints
 * nFlannIM - count of initial matched keypoints after flann matching
 * vnSiftMatched - matched sift keypoints
 * vnDeltaScale - delta scale of keypoints
 * vnDeltaOri - delta orientation of keypoints
 * nMaxDeltaOri, nMinDeltaOri - max and min delta orientation
 * nMaxDeltaScale, nMinDeltaScale - max and min Scale orientation
 */
void SelRecognition_FlannSift (int tcount, int nFlannKnn, int nFlannLibCols_sift, double nFlannMatchFac, std::vector <std::vector <float> > mnSiftExtraFeatures, flann_index_t FlannIdx_Sift, struct FLANNParameters FLANNParam,
                          Keypoint keypts, int &nKeyptsCnt, int &nFlannIM, std::vector<int> &vnSiftMatched, std::vector<double> &vnDeltaScale, std::vector<double> &vnDeltaOri, double &nMaxDeltaOri, double &nMinDeltaOri, double &nMaxDeltaScale, double &nMinDeltaScale) {
    while (keypts) {
        float *f1;
        float *testset;
        int *result;
        float *dists, *dist0, *dist1;
        result = (int*) malloc(tcount * nFlannKnn * sizeof(int));
        dists = (float*) malloc(tcount * nFlannKnn * sizeof(float));
        testset = (float*) malloc(tcount * nFlannLibCols_sift * sizeof(float));
        f1 = testset;
        for (int q = 0; q < 128; q++) {*f1 = keypts->descrip[q]; f1++;}
        flann_find_nearest_neighbors_index(FlannIdx_Sift, testset, tcount, result, dists, nFlannKnn, &FLANNParam);
        double distsq1, distsq2;
        //distsq1 and distsq2 represent the best match and the second best match respectively
        dist0 = dists;
        dist1 = dists;
        dist1++;
        if(*dist0 <= *dist1) {distsq1 = *dist0; distsq2 = *dist1;}
        else {distsq1 = *dist1; distsq2 = *dist0;}
        //if(10*10*distsq1 < nFlannMatchFac*nFlannMatchFac*distsq2) matchcount++; //Consider the current key a match if the closest sq. distance is less than matchfactor/10 of the second closest distance

        if(distsq1 <= nFlannMatchFac*distsq2) {
            vnSiftMatched.resize(nFlannIM + 1);
            vnDeltaScale.resize(nFlannIM + 1);
            vnDeltaOri.resize(nFlannIM + 1);

            vnSiftMatched[nFlannIM] = nKeyptsCnt;
            vnDeltaScale[nFlannIM] = keypts->scale / mnSiftExtraFeatures[*result][2];
            double nCurrOri = keypts->ori;
            double nModelOri = mnSiftExtraFeatures[*result][3];
            double nDeltaOriTmp = 0;
            double nDeltaOri;
            nDeltaOriTmp = nCurrOri - nModelOri;
            if(nDeltaOriTmp >= -PI && nDeltaOriTmp <= PI) nDeltaOri = nDeltaOriTmp;
            else if(nDeltaOriTmp > PI) nDeltaOri=(-2*PI) + nDeltaOriTmp;
            else if(nDeltaOriTmp < -PI) nDeltaOri = (nDeltaOriTmp + 2*PI);
            vnDeltaOri[nFlannIM] = nDeltaOri;
            if(vnDeltaScale[nFlannIM] > nMaxDeltaScale) nMaxDeltaScale = vnDeltaScale[nFlannIM];
            if(vnDeltaScale[nFlannIM] < nMinDeltaScale) nMinDeltaScale = vnDeltaScale[nFlannIM];
            if(vnDeltaOri[nFlannIM] > nMaxDeltaOri) nMaxDeltaOri = vnDeltaOri[nFlannIM];
            if(vnDeltaOri[nFlannIM] < nMinDeltaOri) nMinDeltaOri = vnDeltaOri[nFlannIM];
            nFlannIM++;
        }

        nKeyptsCnt++;
        //free(f1);
        free(testset);
        free(result);
        free(dists);
        //free(dist0);
        //free(dist1);

        keypts = keypts->next;
    }
}
