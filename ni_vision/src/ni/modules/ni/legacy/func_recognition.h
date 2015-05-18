/** @file Function for attention and recognition
 */
#ifndef _NI_LEGACY_FUNC_RECOGNITION_H_
#define _NI_LEGACY_FUNC_RECOGNITION_H_

#include <boost/bind.hpp>

#include <opencv2/core/core.hpp>

#include "ni/3rdparty/siftfast/siftfast.h"

#include "ni/legacy/func_operations.h"
#include "ni/legacy/func_recognition_flann.h"
#include "ni/legacy/timer.h"
#include "ni/legacy/surfprop.h"
#include "ni/legacy/trackprop.h"
#include "ni/legacy/taskid.h"

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


/* Resetting properties of object surfaces
 *
 * Input:
 * nObjsNrLimit - maximum number of object surfaces that can be processed
 * nTrackHistoBin_max - (number of bins)^number of channels
 * nRecogRtNr -
 *
 * Output:
 * stMems - properties of object surfaces in the Short-Term Memory (SurfProp is self-defined struct)
 * nMemsCnt - count of object surfaces
 * nProtoNr - number of current object surface
 * nFoundCnt - count of found object surfaces
 * nFoundNr - number of found object surface
 * vnRecogRating - recognition rating (like a priority) of object surfaces
 */
void ResetMemory (int nObjsNrLimit, int nTrackHistoBin_max, int nRecogRtNr,
                  SurfProp &stMems, int &nMemsCnt, int &nProtoNr, int &nFoundCnt, int &nFoundNr, std::vector<int> &vnRecogRating)
{
    stMems.vnIdx.assign(nObjsNrLimit, 0);
    stMems.vnPtsCnt.assign(nObjsNrLimit, 0);
    stMems.mnPtsIdx.assign(nObjsNrLimit, std::vector<int>(0, 0));
    stMems.mnRCenter.assign(nObjsNrLimit, std::vector<int>(2, 0));
    stMems.mnCCenter.assign(nObjsNrLimit, std::vector<float>(3, 0));
    stMems.mnRect.assign(nObjsNrLimit, std::vector<int>(4, 0));
    stMems.mnCubic.assign(nObjsNrLimit, std::vector<float>(6, 0));
    stMems.mnColorHist.assign(nObjsNrLimit, std::vector<float>(nTrackHistoBin_max, 0));
    stMems.vnLength.assign(nObjsNrLimit, 0);
    stMems.vnMemCtr.assign(nObjsNrLimit, 0);
    stMems.vnStableCtr.assign(nObjsNrLimit, 0);
    stMems.vnLostCtr.assign(nObjsNrLimit, 0);
    stMems.vnFound.assign(nObjsNrLimit, 0);

    nMemsCnt = 0;
    nProtoNr = 0;
    nFoundCnt = 0;
    nFoundNr = 0;
    vnRecogRating.assign(nRecogRtNr, 0);
}



/* Attention: Top-Down guidance - Reordering all properties of object surfaces in the memory after priority
 *
 * Output:
 * vbProtoCand - candidate flag for object surfaces (true, then candidate for attention)
 * stMems - properties of object surfaces in the Short-Term Memory (SurfProp is self-defined struct)
 * vnMemsFound - found flag for objects surfaces
 * nMemsCnt - count of object surfaces
 * vnCandClrDist - color differences of objects surfaces (to target object)
 */
void Attention_TopDown (std::vector<bool> &vbProtoCand, SurfProp &stMems, int nMemsCnt, std::vector<std::pair<float, int> > &veCandClrDist)
{
    std::sort(veCandClrDist.begin(), veCandClrDist.end(), boost::bind(&std::pair<float,int>::first, _1) < boost::bind(&std::pair<float,int>::first, _2));

    std::vector<bool> vbTmpProtoCand = vbProtoCand;
    SurfProp stTmp = stMems;

    for (int i = 0; i < nMemsCnt; i++) {
        vbProtoCand[i] = vbTmpProtoCand[veCandClrDist[i].second];
        stMems.vnIdx[i] = stTmp.vnIdx[veCandClrDist[i].second];
        stMems.vnPtsCnt[i] = stTmp.vnPtsCnt[veCandClrDist[i].second];
        stMems.mnPtsIdx[i] = stTmp.mnPtsIdx[veCandClrDist[i].second];
        stMems.mnRect[i] = stTmp.mnRect[veCandClrDist[i].second];
        stMems.mnRCenter[i] = stTmp.mnRCenter[veCandClrDist[i].second];
        stMems.mnCubic[i] = stTmp.mnCubic[veCandClrDist[i].second];
        stMems.mnCCenter[i] = stTmp.mnCCenter[veCandClrDist[i].second];
        stMems.mnColorHist[i] = stTmp.mnColorHist[veCandClrDist[i].second];
        stMems.vnLength[i] = stTmp.vnLength[veCandClrDist[i].second];
        stMems.vnMemCtr[i] = stTmp.vnMemCtr[veCandClrDist[i].second];
        stMems.vnStableCtr[i] = stTmp.vnStableCtr[veCandClrDist[i].second];
        stMems.vnLostCtr[i] = stTmp.vnLostCtr[veCandClrDist[i].second];
        stMems.vnFound[i] = stTmp.vnFound[veCandClrDist[i].second];
    }
}





/* Selection of a candidate object surface
 *
 * Input:
 * nMemsCnt - count of candidate object surfaces
 * vbProtoCand - flag for candidate object surfaces
 *
 * Output:
 * vMemsFound - state of candidate; 0: not inspected & not found, 1:not inspected & found, 2: inspected & not found, 3: inspected & found
 * nCandID - ID of selected candidate object surface
 */
void Attention_Selection (int nMemsCnt, std::vector<bool> vbProtoCand, std::vector<int> &vnMemsFound, int &nCandID) {

    for (int i = 0; i < nMemsCnt; i++) {
        if (!vbProtoCand[i]) continue;

        // Selecting the most relevant candidate from not inspected candidate pool
        if (vnMemsFound[i] < 2) {
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
 * vnMemsPtsCnt - count of pixel which belong to the object surface (of downsampled image)
 * cvm_rgb_org - original image
 * mnMemsPtsIdx - indices of pixel which belong to the object surfae (of downsampled image)
 *
 * Output:
 * cvm_cand_tmp - image of candidate object surface
 * vnIdxTmp - indices of pixel which belong to the object surfae (of original image)
 */
void Recognition_Attention (int nCandID, int nImgScale, int nDsWidth, std::vector<int> vnMemsPtsCnt, cv::Mat cvm_rgb_org,
                       cv::Mat &cvm_cand_tmp, std::vector<std::vector<int> > mnMemsPtsIdx, std::vector<int> &vnIdxTmp) {
    int pts_cnt = 0;
    int xx, yy;
    if (nImgScale > 1) {
        int idx;
        for (int j = 0; j < vnMemsPtsCnt[nCandID]; j++) {
            GetPixelPos(mnMemsPtsIdx[nCandID][j], nDsWidth, xx, yy);
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
    else {vnIdxTmp = mnMemsPtsIdx[nCandID]; pts_cnt = (int)vnMemsPtsCnt[nCandID];}

    //int xxx, yyy;
    for (int i = 0; i < pts_cnt; i++) {
        cvm_cand_tmp.data[vnIdxTmp[i]*3] = cvm_rgb_org.data[vnIdxTmp[i]*3];
        cvm_cand_tmp.data[vnIdxTmp[i]*3+1] = cvm_rgb_org.data[vnIdxTmp[i]*3+1];
        cvm_cand_tmp.data[vnIdxTmp[i]*3+2] = cvm_rgb_org.data[vnIdxTmp[i]*3+2];
    }
}



/* FLANN (fast library for approximate nearest neighbor) matching SIFT keypoints to find target object
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
void Recognition_Flann (int tcount, int nFlannKnn, int nFlannLibCols_sift, double nFlannMatchFac, std::vector <std::vector <float> > mnSiftExtraFeatures, flann_index_t FlannIdx_Sift, struct FLANNParameters FLANNParam,
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



/* Recognition process of a selected candidate. Determines if a selected object is the target object
 *
 * Input:
 * nCandID - ID of the candidate in the list candidate object surfaces
 * nImgScale - ratio of original and downsampled image
 * nTimeRatio - ratio of milli- and nanoseconds (i.e. 10⁶)
 * nMemsCnt - number of object surfaces
 * nTrackHistoBin_max - (number of bin)^(number of channels)
 * sTimeDir - path to save time measurement to
 * sImgExt - extension of image file
 * cvm_rgb_org - original rgb image
 * cvm_rgb_ds - downsampled rgb image
 * cvm_rec_org - original image for recognition process
 * cvm_rec_ds - downsampled original image for recognition process
 * nTrackCntLost - threshold for vnMemsLostCnt
 *
 * Output:
 * nCandCnt - number of candidate object surfaces
 * nCandClrDist - buffer of color histogram difference between the current candidate and the target object
 * nFoundCnt - count of found objects in a recognition cycle
 * nFoundNr - number of found object
 * nFoundFrame - number of frame where the object was found
 * nCandKeyCnt - count of keypoints for the current candidate
 * nCandRX, nCandRY, nCandRW, nCandRH - coordinates of the 2D-bounding box for the current candidate
 * t_rec_found_start, t_rec_found_end - variable for time measurement
 * bSwitchRecordTime - flag for time measurement
 * nRecogRtNr - count of frames to record
 * vnRecogRating_tmp - vector of results for time measurement
 * cvm_cand - image of the current candidate
 */
void Recognition (int nCandID, int nImgScale, int nDsWidth, int nTimeRatio, int nMemsCnt, int nTrackHistoBin_max,
                     cv::Mat cvm_rgb_org, cv::Mat &cvm_rec_org, cv::Mat &cvm_rec_ds, SurfProp &stMems,
                     int nTrackCntLost, int &nCandCnt, float nCandClrDist, int &nFoundCnt, int &nFoundNr, int &nFoundFrame,
                     int &nCandKeyCnt, int &nCandRX, int &nCandRY, int &nCandRW, int &nCandRH,
                     struct timespec t_rec_found_start, struct timespec t_rec_found_end, bool bSwitchRecordTime, int nRecogRtNr, std::vector<int> &vnRecogRating_tmp, cv::Mat cvm_cand,
                  bool &bTimeSift,
                  bool &bTimeFlann,
                  const bool bRecogClrMask,
                  const TrackProp &stTrack,
                  const std::vector<std::vector<float > > &mnColorHistY_lib,
                  const int nSiftScales,
                  const double nSiftInitSigma,
                  const double nSiftPeakThrs,
                  double &nTimeSift,
                  const int tcount,
                  const int nFlannKnn,
                  const int nFlannLibCols_sift,
                  const double nFlannMatchFac,
                  const std::vector <std::vector <float> > mnSiftExtraFeatures,
                  const flann_index_t FlannIdx_Sift,
                  const struct FLANNParameters FLANNParam,
                  const int T_numb,
                  const double T_orient,
                  const double T_scale,
                  const int nDeltaBinNo,
                  const int nFlannMatchCnt,
                  const double nRecogDClr,
                  double &nTimeRecFound,
                  const int nCtrFrame,
                  const int nCtrFrame_tmp,
                  const std::vector<bool> &vbFlagTask,
                  const TaskID &stTID,
                  std::vector<std::vector<int> > &mnTimeMeas1,
                  std::vector<std::vector<float> > &mnTimeMeas2,
                  const int nCtrRecCycle,
                  double &nTimeRecFound_max,
                  double &nTimeRecFound_min,
                  double &nTimeFlann,
                  const int nRecordMode,
                  const cv::Scalar &c_red,
                  const cv::Scalar &c_white,
                  const cv::Scalar &c_blue,
                  const cv::Scalar &c_lemon) {

    using std::max;
    using std::min;

    nCandCnt++;
    // Initialize
    if (stMems.vnFound[nCandID] < 1) stMems.vnFound[nCandID] = 2;
    else stMems.vnFound[nCandID] = 3;



    ////////////** Extracting SIFT key-points **////////////////////////////////////////////////////////
    struct timespec t_sift_start, t_sift_end; clock_gettime(CLOCK_MONOTONIC_RAW, &t_sift_start); bTimeSift = true;

    //////*  Extracting object surface from original image  */////////////////////////
    cvm_cand = cv::Scalar(0, 0, 0);
    std::vector<int> vnIdxTmp(stMems.vnPtsCnt[nCandID]*nImgScale*nImgScale, 0);

    Recognition_Attention (nCandID, nImgScale, nDsWidth, stMems.vnPtsCnt, cvm_rgb_org, cvm_cand, stMems.mnPtsIdx, vnIdxTmp);


    //////** 1. Estimating histogram-distance **//////////////////////////////
    float nColorDistMaskOrg = 0;
    if (bRecogClrMask) {
        ///// Calc3DColorHistogram computes the same color models but it is more robust ///////
        std::vector<float> vnHistTmp(nTrackHistoBin_max, 0);
        Calc3DColorHistogram (cvm_rgb_org, vnIdxTmp, stTrack.HistoBin, vnHistTmp);


        for (int i = 0; i< nTrackHistoBin_max; i++) {
            if (mnColorHistY_lib.size() == 1) {
                nColorDistMaskOrg += fabs(mnColorHistY_lib[0][i] - vnHistTmp[i]);
            }
            else nColorDistMaskOrg += fabs(mnColorHistY_lib[stTrack.ClrMode][i] - vnHistTmp[i]);
        }
        nColorDistMaskOrg = nColorDistMaskOrg/2;
    }
    //////////////////////* End of the estimating histogram-distance *////////////////////////


    //////** 2. SIFT extraction from the object surface **//////////////////////////////////////////

    ///////// Calculating rectangle coordinates of the candidate in high-resolution 2D image from the rectangle on downsampled image
    int x_min_org = stMems.mnRect[nCandID][0] * nImgScale;
    int y_min_org = stMems.mnRect[nCandID][1] * nImgScale - nImgScale+1;
    int x_max_org = stMems.mnRect[nCandID][2] * nImgScale + nImgScale-1;
    int y_max_org = stMems.mnRect[nCandID][3] * nImgScale;
    if (x_min_org < 0) x_min_org = 0; if (x_max_org > cvm_rgb_org.cols - 1) x_min_org = cvm_rgb_org.cols;
    if (y_min_org < 0) y_min_org = 0; if (y_max_org > cvm_rgb_org.cols - 1) y_max_org = cvm_rgb_org.cols;

    nCandRX = x_min_org;
    nCandRY = y_min_org;
    nCandRW = x_max_org - x_min_org + 1;
    nCandRH = y_max_org - y_min_org + 1;

    Keypoint keypts, keypts_tmp;
    GetSiftKeypoints(cvm_cand, nSiftScales, nSiftInitSigma, nSiftPeakThrs, nCandRX, nCandRY, nCandRW, nCandRH, keypts);
    keypts_tmp = keypts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t_sift_end); nTimeSift = double(timespecDiff(&t_sift_end, &t_sift_start)/nTimeRatio);
    cvm_cand.release();



    /////** Flann searching **//////////////////////////////
    struct timespec t_flann_start, t_flann_end; clock_gettime(CLOCK_MONOTONIC_RAW, &t_flann_start); bTimeFlann = true;

    std::vector<int> vnSiftMatched; std::vector <double> vnDeltaScale; std::vector <double> vnDeltaOri;
    double nMaxDeltaOri = -999; double nMaxDeltaScale = -999;
    double nMinDeltaOri = 999; //double nMinDeltaScale = 999;
    int nKeyptsCnt = 0; int nFlannIM=0;

    Recognition_Flann (tcount, nFlannKnn, nFlannLibCols_sift, nFlannMatchFac, mnSiftExtraFeatures, FlannIdx_Sift, FLANNParam, keypts, nKeyptsCnt, nFlannIM, vnSiftMatched, vnDeltaScale, vnDeltaOri, nMaxDeltaOri, nMinDeltaOri, nMaxDeltaScale, nMinDeltaOri);
    nCandKeyCnt = nKeyptsCnt;

    /////** Filtering: extracting true-positives from matched keypoints **//////////////////
    double nDeltaScale=0;
    int nFlannTP=0;

    std::vector<bool> vbSiftTP(nKeyptsCnt, 0);
    if(nFlannIM > T_numb) CalcDeltaScaleOri(vnSiftMatched, vnDeltaScale, vnDeltaOri, nDeltaScale, nDeltaBinNo, nMaxDeltaOri, nMinDeltaOri, T_orient, T_scale, nFlannTP, vbSiftTP);
    else nFlannTP=0;



    ////** Setting the object as "found" and Drawing**////////////
    float nColorDist;
    if (bRecogClrMask) nColorDist = nColorDistMaskOrg; else nColorDist = nCandClrDist;
    if(nFlannTP >= nFlannMatchCnt && nColorDist < nRecogDClr) { //If the number of matches for the current object are more than or equal to the threshhold for matches
        clock_gettime(CLOCK_MONOTONIC_RAW, &t_rec_found_end); nTimeRecFound = double(timespecDiff(&t_rec_found_end, &t_rec_found_start)/nTimeRatio);
        printf("%d. object with %3d keypoints (%d %d) at frame %d (%d), Color dist: %4.3f (%4.3f), Object size: %4.0f mm\n", nCandCnt, nFlannTP, nFlannIM, nKeyptsCnt, nCtrFrame, nCtrFrame_tmp, nCandClrDist, nColorDist, stMems.vnLength[nCandID]*1000);

        nFoundCnt++;
        nFoundNr = nCandCnt;
        nFoundFrame = nCtrFrame;
        stMems.vnFound[nCandID] = 3;


        if (vbFlagTask[stTID.nRecTime] && bSwitchRecordTime) {
            if (nFoundNr < nRecogRtNr) vnRecogRating_tmp[nFoundNr]++;
            else vnRecogRating_tmp[0]++;

            int size = mnTimeMeas1.size();
            mnTimeMeas1.resize(size+1);
            mnTimeMeas1[size].assign(9, 0);
            mnTimeMeas1[size][0] = nCtrRecCycle+1;
            mnTimeMeas1[size][1] = nFoundFrame;
            mnTimeMeas1[size][2] = nFoundNr;
            mnTimeMeas1[size][4] = nKeyptsCnt;
            mnTimeMeas1[size][5] = nFlannIM;
            mnTimeMeas1[size][6] = nFlannTP;
            mnTimeMeas1[size][7] = nCandRW;
            mnTimeMeas1[size][8] = nCandRH;

            mnTimeMeas2.resize(size+1);
            mnTimeMeas2[size].assign(2, 0);
            mnTimeMeas2[size][0] = nTimeRecFound;
            mnTimeMeas2[size][1] = nTimeSift;

            if (nTimeRecFound > nTimeRecFound_max) nTimeRecFound_max = nTimeRecFound;
            if (nTimeRecFound < nTimeRecFound_min) nTimeRecFound_min = nTimeRecFound;
        }
    }
    else {if (stMems.vnFound[nCandID] == 1 || stMems.vnFound[nCandID] == 3) stMems.vnFound[nCandID] = 2;}

    clock_gettime(CLOCK_MONOTONIC_RAW, &t_flann_end); nTimeFlann = double(timespecDiff(&t_flann_end, &t_flann_start)/nTimeRatio);



    if (vbFlagTask[stTID.nRecogOrg] || (vbFlagTask[stTID.nRecVideo] && nRecordMode)) {
        cv::rectangle(cvm_rec_org, cv::Point(x_min_org, y_min_org), cv::Point(x_max_org, y_max_org), c_red, 3);

        keypts = keypts_tmp;
        while(keypts) {
            cv::rectangle(cvm_rec_org, cv::Point(keypts->row + x_min_org - 2, keypts->col + y_min_org - 2), cv::Point(keypts->row + x_min_org + 2, keypts->col + y_min_org + 2), c_white, 1);
            keypts= keypts->next;
        }
        keypts = keypts_tmp;
        nKeyptsCnt = 0;
        while(keypts) {
            if(vbSiftTP[nKeyptsCnt]) {
                cv::rectangle(cvm_rec_org, cv::Point(keypts->row + x_min_org - 2, keypts->col + y_min_org - 2), cv::Point(keypts->row + x_min_org + 2, keypts->col + y_min_org + 2), c_white, 1);
                cv::rectangle(cvm_rec_org, cv::Point(keypts->row + x_min_org - 3, keypts->col + y_min_org - 3), cv::Point(keypts->row + x_min_org + 3, keypts->col + y_min_org + 3), c_blue, 1);
                cv::rectangle(cvm_rec_org, cv::Point(keypts->row + x_min_org - 1, keypts->col + y_min_org - 1), cv::Point(keypts->row + x_min_org + 1, keypts->col + y_min_org + 1), c_blue, 1);
            }
            nKeyptsCnt++;
            keypts= keypts->next;
        }
        FreeKeypoints(keypts); FreeKeypoints(keypts_tmp);


        int offset = 2*nImgScale;
        int line_thickness = cvm_rec_org.cols/150;
        for (int j = 0; j < nMemsCnt; j++) {
            if (stMems.vnFound[j] != 1 && stMems.vnFound[j] != 3) continue;

            if (stMems.vnLostCtr[j] > nTrackCntLost) continue;

            int x_min_tmp = stMems.mnRect[j][0] * nImgScale;
            int y_min_tmp = stMems.mnRect[j][1] * nImgScale - nImgScale+1;
            int x_max_tmp = stMems.mnRect[j][2] * nImgScale + nImgScale-1;
            int y_max_tmp = stMems.mnRect[j][3] * nImgScale;
            if (x_min_tmp < 0) x_min_tmp = 0;
            if (x_max_tmp > cvm_rgb_org.cols - 1) x_min_tmp = cvm_rgb_org.cols;
            if (y_min_tmp < 0) y_min_tmp = 0;
            if (y_max_tmp > cvm_rgb_org.cols - 1) y_max_tmp = cvm_rgb_org.cols;

            bool draw = true;
            for (int k = 0; k < j; k++) {
                if (stMems.vnFound[k] != 1 && stMems.vnFound[k] != 3) continue;
                int x_overlapp_min = max(stMems.mnRect[j][0], stMems.mnRect[k][0]);
                int y_overlapp_min = max(stMems.mnRect[j][1], stMems.mnRect[k][1]);
                int x_overlapp_max = min(stMems.mnRect[j][2], stMems.mnRect[k][2]);
                int y_overlapp_max = min(stMems.mnRect[j][3], stMems.mnRect[k][3]);
                int size_overlapp = (x_overlapp_max - x_overlapp_min)*(y_overlapp_max - y_overlapp_min);
                int size_curr = (stMems.mnRect[j][2] - stMems.mnRect[j][0])*(stMems.mnRect[j][3] - stMems.mnRect[j][1]);
                int size_past = (stMems.mnRect[k][2] - stMems.mnRect[k][0])*(stMems.mnRect[k][3] - stMems.mnRect[k][1]);

            }

            if (draw) cv::rectangle(cvm_rec_org, cv::Point(x_min_tmp -offset, y_min_tmp -offset), cv::Point(x_max_tmp +offset, y_max_tmp +offset), c_lemon, line_thickness);
        }
    }

    if (vbFlagTask[stTID.nRecogDs] || ((vbFlagTask[stTID.nRecogOrg] || vbFlagTask[stTID.nRecVideo]) && nRecordMode)) {
        for (int j = 0; j < stMems.vnPtsCnt[nCandID]; j++) {
            cvm_rec_ds.data[stMems.mnPtsIdx[nCandID][j]*3] = 0;
            cvm_rec_ds.data[stMems.mnPtsIdx[nCandID][j]*3+1] = 0;
            cvm_rec_ds.data[stMems.mnPtsIdx[nCandID][j]*3+2] = 255;
        }

        int offset = 2;
        int line_thickness = cvm_rec_ds.cols/128;
        for (int j = 0; j < nMemsCnt; j++) {
            if (stMems.vnFound[j] != 1 && stMems.vnFound[j] != 3) continue;

            if (stMems.vnLostCtr[j] > nTrackCntLost) continue;

            bool draw = true;
            for (int k = 0; k < j; k++) {
                if (stMems.vnFound[k] != 1 && stMems.vnFound[k] != 3) continue;
                int x_overlapp_min = max(stMems.mnRect[j][0], stMems.mnRect[k][0]);
                int y_overlapp_min = max(stMems.mnRect[j][1], stMems.mnRect[k][1]);
                int x_overlapp_max = min(stMems.mnRect[j][2], stMems.mnRect[k][2]);
                int y_overlapp_max = min(stMems.mnRect[j][3], stMems.mnRect[k][3]);
                int size_overlapp = (x_overlapp_max - x_overlapp_min)*(y_overlapp_max - y_overlapp_min);
                int size_curr = (stMems.mnRect[j][2] - stMems.mnRect[j][0])*(stMems.mnRect[j][3] - stMems.mnRect[j][1]);
                int size_past = (stMems.mnRect[k][2] - stMems.mnRect[k][0])*(stMems.mnRect[k][3] - stMems.mnRect[k][1]);

                if (size_overlapp > size_curr*0.6 || size_overlapp > size_past*0.6) draw = false;
            }

            if (draw) cv::rectangle(cvm_rec_ds, cv::Point(stMems.mnRect[j][0] -offset, stMems.mnRect[j][1] -offset), cv::Point(stMems.mnRect[j][2] +offset, stMems.mnRect[j][3] +offset), c_lemon, line_thickness);
        }
    }
}

#endif // _NI_LEGACY_FUNC_RECOGNITION_H_
