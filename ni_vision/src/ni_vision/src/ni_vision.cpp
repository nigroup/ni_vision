#include <ros/ros.h>
#include <boost/thread/mutex.hpp>
#include <boost/thread.hpp>



#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


#include <sensor_msgs/PointCloud2.h>
// PCL specific includes
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/io.h>  //for getFieldIndex()


// Fast SIFT Library
#include "siftfast/siftfast.h"
#include "flann/flann.h"


// Terminal tools
#include "terminal_tools/parse.h"


// etc
#include <sys/stat.h>
#include <time.h>

#include <iostream>
#include <fstream>


// -----------------------------------------
// -----Sub functions-----------------------
// -----------------------------------------
#include "func_header.cpp"

#include "func_init.cpp"
#include "func_operations.cpp"
#include "func_preproc.cpp"
#include "func_segmentation.cpp"
#include "func_segmentation_gb.cpp"
#include "func_recognition.cpp"
#include "func_etc.cpp"

//#include "/opt/ros/electric/stacks/perception_pcl/flann/include/flann/flann.h"


namespace enc = sensor_msgs::image_encodings;


//////// Dimension of Depth Image ////////////////////////////////////////
int nCvWidth;
int nCvHeight;
int nCvSize;


//// 2D
int nCvCurrentkey = -1;



// Global data
sensor_msgs::PointCloud2ConstPtr cloud_, cloud_old_;
boost::mutex m;

pcl::PointCloud<pcl::PointXYZ> cloud_xyz;
pcl::PointCloud<pcl::PointXYZRGB> cloud_xyz_rgb;

cv::Mat cvm_image_camera;

///// temp ////
int nCropX = 1, nCropY = 17, nCropW = 292, nCropH = 220;
int nCalibX = nCropX-1, nCalibY = nCropY-1, nCalibW = nCropW+4, nCalibH = nCropH+2;
std::vector<std::vector<int> > mnTimeMeas1;
std::vector<std::vector<float> > mnTimeMeas2;
std::vector<std::vector<float> > mnTimeMeas3;

std::vector<float> mnTimeMeasGbSegm;
std::vector<std::vector<float> > mnTimeMeasSiftWhole;
bool bRtGbSegm = false, bRtSiftW = true;
int nMeasCounter = 0;



// for histogram_view
static int nStatCounter = 1;




int64_t timespecDiff (struct timespec *timeA_p, struct timespec *timeB_p) {
  return ((timeA_p->tv_sec * 1000000000) + timeA_p->tv_nsec) - ((timeB_p->tv_sec * 1000000000) + timeB_p->tv_nsec);
}


void SelRecognition (int nCandID, int nImgScale, int nTimeRatio, int nProtoCnt, int nTrackHistoBin_max, std::string sTimeDir, std::string sImgExt,
                     cv::Mat cvm_rgb_org, cv::Mat cvm_rgb_ds, cv::Mat &cvm_rec_org, cv::Mat &cvm_rec_ds,
                     std::vector<int> vnProtoPtsCnt, std::vector<std::vector<int> > mnProtoPtsIdx, std::vector<std::vector<int> > mnProtoRect, std::vector<int> &vnProtoFound,
                     std::vector<int> vnProtoLength, std::vector<int> vnProtoDisapCnt, int nTrackCntDisap, int &nCandCnt, std::vector<std::vector<float> > vnTmpProtoDiff, int &nFoundCnt, int &nFoundNr, int &nFoundFrame,
                     int &nCandKeyCnt, int &nCandRX, int &nCandRY, int &nCandRW, int &nCandRH,
                     struct timespec t_rec_found_start, struct timespec t_rec_found_end, bool bSwitchRecordTime, int nRecogRtNr, std::vector<int> &vnRecogRating_tmp, int nModeRecord, cv::Mat cvm_cand) {
                     //std::vector<std::vector<float> > mnProtoClrHist, std::vector<std::vector<int> > mnProtoCenter) {

    nCandCnt++;
    if (vnProtoFound[nCandID] < 1) vnProtoFound[nCandID] = 2;
    else vnProtoFound[nCandID] = 3;



    ////////////** Extracting SIFT key-points **////////////////////////////////////////////////////////
    struct timespec t_sift_start, t_sift_end; clock_gettime(CLOCK_MONOTONIC_RAW, &t_sift_start); bTimeSift = true;

    /////** Extracting object surface from original image **//////////////////////////////
    cvm_cand = cv::Scalar(0, 0, 0);
    std::vector<int> vnIdxTmp(vnProtoPtsCnt[nCandID]*nImgScale*nImgScale, 0);

    SelRecognition_1 (nCandID, nImgScale, nCvWidth, vnProtoPtsCnt, cvm_rgb_org, cvm_cand, mnProtoPtsIdx, vnIdxTmp);

//    std::string sRecordRecDir = "aaa_";
//    std::string sDat;
//    char s_frame[128];
//    sprintf(s_frame, "snap_%06i_08_rec1", nCntFrame); sDat = sRecordRecDir + "/" + s_frame + sImgExt; cv::imwrite(sDat, cvm_cand_tmp);
//    printf(sDat.data());
//    printf("\n");


    /////** Estimating histogram-distance **//////////////////////////////
    float nColorDistMaskOrg = 0;
    if (bRecogClrMask) {
        std::vector<float> vnHistTmp(nTrackHistoBin_max, 0);
        // Calc3DColorHistogram computes the same color models but it is more robust

        Calc3DColorHistogram (cvm_rgb_org, vnIdxTmp, nTrackHistoBin, vnHistTmp);

        //else Calc3DColorHistogram (cvm_rgb_ds, mnProtoPtsIdx[nCandID], nTrackClrMode, nTrackHistoBin, vnHistTmp);


        // Histogram_view code
//        if(nStatCounter < 10) {
//            stringstream ss_puffer;
//            ss_puffer << nStatCounter;
//            stringstream ss_puffer2;
//            ss_puffer2 << nTrackClrMode;
//            cv::FileStorage fs("/home/ni/ros_overlays/Histograms/histo_Mode" + ss_puffer2.str() + "_Obj" +ss_puffer.str() +".yaml", cv::FileStorage::WRITE);
//            cv::Mat cvm_puffer(nTrackHistoBin*nTrackHistoBin*nTrackHistoBin, 1, cv::DataType<float>::type);
//            for(int i = 0; i < nTrackHistoBin*nTrackHistoBin*nTrackHistoBin; i++) {
//                cvm_puffer.at<float>(i,0) = vnHistTmp[i];
//            }
//            fs << "TestObjectFeatureVectors" << cvm_puffer;
//            fs.release();
//        }


        for (int i = 0; i< nTrackHistoBin_max; i++) {
            if (mnColorHistY_lib.size() == 1) nColorDistMaskOrg += fabs(mnColorHistY_lib[0][i] - vnHistTmp[i]);
            else nColorDistMaskOrg += fabs(mnColorHistY_lib[nTrackClrMode][i] - vnHistTmp[i]);
        }
        nColorDistMaskOrg = nColorDistMaskOrg/2;
    }


    /////** SIFT extraction from the object surface **//////////////////////////////
    int x_min_org = mnProtoRect[nCandID][0] * nImgScale;
    int y_min_org = mnProtoRect[nCandID][1] * nImgScale - nImgScale+1;
    int x_max_org = mnProtoRect[nCandID][2] * nImgScale + nImgScale-1;
    int y_max_org = mnProtoRect[nCandID][3] * nImgScale;
    if (x_min_org < 0) x_min_org = 0;
    if (x_max_org > cvm_rgb_org.cols - 1) x_min_org = cvm_rgb_org.cols;
    if (y_min_org < 0) y_min_org = 0;
    if (y_max_org > cvm_rgb_org.cols - 1) y_max_org = cvm_rgb_org.cols;

    nCandRX = x_min_org;
    nCandRY = y_min_org;
    nCandRW = x_max_org - x_min_org + 1;
    nCandRH = y_max_org - y_min_org + 1;

    //cv::imshow("aaaaaaaa", cvm_cand);

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

    SelRecognition_FlannSift (tcount, nFlannKnn, nFlannLibCols_sift, nFlannMatchFac, mnSiftExtraFeatures, FlannIdx_Sift, FLANNParam, keypts, nKeyptsCnt, nFlannIM, vnSiftMatched, vnDeltaScale, vnDeltaOri, nMaxDeltaOri, nMinDeltaOri, nMaxDeltaScale, nMinDeltaOri);
    nCandKeyCnt = nKeyptsCnt;

    /////** Filtering: extracting true-positives from matched keypoints **//////////////////
    double nDeltaScale=0;
    int nFlannTP=0;

    std::vector<bool> vbSiftTP(nKeyptsCnt, 0);
    if(nFlannIM > T_numb) CalcDeltaScaleOri(vnSiftMatched, vnDeltaScale, vnDeltaOri, nDeltaScale, nDeltaBinNo, nMaxDeltaOri, nMinDeltaOri, T_orient, T_scale, nFlannTP, vbSiftTP);
    else nFlannTP=0;



    ////** Setting the object as "found" and Drawing**////////////
    float nColorDist;
    if (bRecogClrMask) nColorDist = nColorDistMaskOrg; else nColorDist = vnTmpProtoDiff[0][nCandID];
    if(nFlannTP >= nFlannMatchCnt && nColorDist < nRecogDClr) { //If the number of matches for the current object are more than or equal to the threshhold for matches
        clock_gettime(CLOCK_MONOTONIC_RAW, &t_rec_found_end); nTimeRecFound = double(timespecDiff(&t_rec_found_end, &t_rec_found_start)/nTimeRatio);
        //printf("%d. object with %3d keypoints (%d %d) at frame %d (%d), Color dist: %4.3f (%4.3f), Object size: %d mm\n", nCandCnt, nFlannTP, nFlannIM, nKeyptsCnt, nCntFrame, nCntFrame_tmp, vnTmpProtoDiff[0][nCandID], nColorDist, vnProtoLength[nCandID]);

        nFoundCnt++;
        nFoundNr = nCandCnt;
        nFoundFrame = nCntFrame;
        vnProtoFound[nCandID] = 3;


        if (vbFlagTask[stTID.nRecTime] && bSwitchRecordTime) {
            if (nFoundNr < nRecogRtNr) vnRecogRating_tmp[nFoundNr]++;
            else vnRecogRating_tmp[0]++;

            int size = mnTimeMeas1.size();
            mnTimeMeas1.resize(size+1);
            mnTimeMeas1[size].assign(9, 0);
            mnTimeMeas1[size][0] = nCntRecCycle+1;
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
    else {if (vnProtoFound[nCandID] == 1 || vnProtoFound[nCandID] == 3) vnProtoFound[nCandID] = 2;}

    clock_gettime(CLOCK_MONOTONIC_RAW, &t_flann_end); nTimeFlann = double(timespecDiff(&t_flann_end, &t_flann_start)/nTimeRatio);



    if (vbFlagTask[stTID.nRecogOrg] || (vbFlagTask[stTID.nRecVideo] && nRecordMode) || nModeRecord == 2 || nModeRecord == 3) {
        cv::rectangle(cvm_rec_org, cv::Point(x_min_org, y_min_org), cv::Point(x_max_org, y_max_org), c_red, 3);

//        if(nStatCounter < 10) {
//            stringstream ss_puffer3;
//            ss_puffer3 << nStatCounter;
//            stringstream ss_puffer4;
//            ss_puffer4 << nTrackClrMode;
//            imwrite( "/home/ni/ros_overlays/Histograms/testimage_Mode" + ss_puffer4.str() + "_Obj" + ss_puffer3.str() + ".jpg", cvm_rec_org );
//            nStatCounter++;
//        }


        keypts = keypts_tmp;
        while(keypts) {
            cv::rectangle(cvm_rec_org, cv::Point(keypts->row + x_min_org - 2, keypts->col + y_min_org - 2), cv::Point(keypts->row + x_min_org + 2, keypts->col + y_min_org + 2), c_white, 1);
            keypts= keypts->next;
        }
        keypts = keypts_tmp;
        nKeyptsCnt = 0;
        while(keypts) {
            if(vbSiftTP[nKeyptsCnt]) {
                cv::rectangle(cvm_rec_org, cv::Point(keypts->row + x_min_org - 3, keypts->col + y_min_org - 3), cv::Point(keypts->row + x_min_org + 3, keypts->col + y_min_org + 3), c_blue, 1);
                cv::rectangle(cvm_rec_org, cv::Point(keypts->row + x_min_org - 1, keypts->col + y_min_org - 1), cv::Point(keypts->row + x_min_org + 1, keypts->col + y_min_org + 1), c_blue, 1);
            }
            nKeyptsCnt++;
            keypts= keypts->next;
        }
        FreeKeypoints(keypts); FreeKeypoints(keypts_tmp);


        int offset = 2*nImgScale;
        int line_thickness = cvm_rec_org.cols/150;
        for (int j = 0; j < nProtoCnt; j++) {
            if (vnProtoFound[j] != 1 && vnProtoFound[j] != 3) continue;

            if (vnProtoDisapCnt[j] > nTrackCntDisap) continue;

            int x_min_tmp = mnProtoRect[j][0] * nImgScale;
            int y_min_tmp = mnProtoRect[j][1] * nImgScale - nImgScale+1;
            int x_max_tmp = mnProtoRect[j][2] * nImgScale + nImgScale-1;
            int y_max_tmp = mnProtoRect[j][3] * nImgScale;
            if (x_min_tmp < 0) x_min_tmp = 0;
            if (x_max_tmp > cvm_rgb_org.cols - 1) x_min_tmp = cvm_rgb_org.cols;
            if (y_min_tmp < 0) y_min_tmp = 0;
            if (y_max_tmp > cvm_rgb_org.cols - 1) y_max_tmp = cvm_rgb_org.cols;

            bool draw = true;
            for (int k = 0; k < j; k++) {
                if (vnProtoFound[k] != 1 && vnProtoFound[k] != 3) continue;
                int x_overlapp_min = max(mnProtoRect[j][0], mnProtoRect[k][0]);
                int y_overlapp_min = max(mnProtoRect[j][1], mnProtoRect[k][1]);
                int x_overlapp_max = min(mnProtoRect[j][2], mnProtoRect[k][2]);
                int y_overlapp_max = min(mnProtoRect[j][3], mnProtoRect[k][3]);
                int size_overlapp = (x_overlapp_max - x_overlapp_min)*(y_overlapp_max - y_overlapp_min);
                int size_curr = (mnProtoRect[j][2] - mnProtoRect[j][0])*(mnProtoRect[j][3] - mnProtoRect[j][1]);
                int size_past = (mnProtoRect[k][2] - mnProtoRect[k][0])*(mnProtoRect[k][3] - mnProtoRect[k][1]);

                //if (size_overlapp > size_curr*0.6 || size_overlapp > size_past*0.6) draw = false;
            }

            if (draw) cv::rectangle(cvm_rec_org, cv::Point(x_min_tmp -offset, y_min_tmp -offset), cv::Point(x_max_tmp +offset, y_max_tmp +offset), c_lemon, line_thickness);
        }
    }

    if (vbFlagTask[stTID.nRecogDs] || ((vbFlagTask[stTID.nRecogOrg] || vbFlagTask[stTID.nRecVideo]) && nRecordMode) || nModeRecord == 2 || nModeRecord == 3) {
        for (int j = 0; j < vnProtoPtsCnt[nCandID]; j++) {
            cvm_rec_ds.data[mnProtoPtsIdx[nCandID][j]*3] = 0;
            cvm_rec_ds.data[mnProtoPtsIdx[nCandID][j]*3+1] = 0;
            cvm_rec_ds.data[mnProtoPtsIdx[nCandID][j]*3+2] = 255;
        }

        //ShowColorHistogramView (nCandID, nTrackHistoBin_max, mnProtoClrHist, mnProtoCenter, vnProtoPtsCnt, cvm_rec_ds);

        int offset = 2;
        int line_thickness = cvm_rec_ds.cols/128;
        for (int j = 0; j < nProtoCnt; j++) {
            if (vnProtoFound[j] != 1 && vnProtoFound[j] != 3) continue;

            if (vnProtoDisapCnt[j] > nTrackCntDisap) continue;

            bool draw = true;
            for (int k = 0; k < j; k++) {
                if (vnProtoFound[k] != 1 && vnProtoFound[k] != 3) continue;
                int x_overlapp_min = max(mnProtoRect[j][0], mnProtoRect[k][0]);
                int y_overlapp_min = max(mnProtoRect[j][1], mnProtoRect[k][1]);
                int x_overlapp_max = min(mnProtoRect[j][2], mnProtoRect[k][2]);
                int y_overlapp_max = min(mnProtoRect[j][3], mnProtoRect[k][3]);
                int size_overlapp = (x_overlapp_max - x_overlapp_min)*(y_overlapp_max - y_overlapp_min);
                int size_curr = (mnProtoRect[j][2] - mnProtoRect[j][0])*(mnProtoRect[j][3] - mnProtoRect[j][1]);
                int size_past = (mnProtoRect[k][2] - mnProtoRect[k][0])*(mnProtoRect[k][3] - mnProtoRect[k][1]);

                if (size_overlapp > size_curr*0.6 || size_overlapp > size_past*0.6) draw = false;
            }

            if (draw) cv::rectangle(cvm_rec_ds, cv::Point(mnProtoRect[j][0] -offset, mnProtoRect[j][1] -offset), cv::Point(mnProtoRect[j][2] +offset, mnProtoRect[j][3] +offset), c_lemon, line_thickness);
        }
    }
}


void updateImage() {

    ros::Duration d (0.001);

    bool bPointRgb;
    //std::vector<sensor_msgs::PointField> fields;

    InitVariables();
    cv::setMouseCallback(sTitle, MouseHandler, NULL);


    std::vector<int> vnBtnProp(nTaskNrMax, 0);
    vnBtnProp[stTID.nSegmentation] = 20; vnBtnProp[stTID.nRecognition] = 20;


    int nPadOver, nPadCol1, nPadCol2, nPadHeight;
    SetPad(nBtnSize, mnBtnPos, nPadOver, nPadCol1, nPadCol2, nPadHeight);

    int nBorderThickness = 2, nTabThickness = 20;
    cv::Size size_org = cvm_image_camera.size();
    cv::Size size_ds = cv::Size(nCvWidth, nCvHeight);
    cv::Size size_total = cv::Size(size_org.width + size_ds.width + 3*nBorderThickness, size_org.height + 2*nBorderThickness);

    int nTimeRatio = 1000000;
    int nObjsNrLimit = 1000;
    int nImgScale = size_org.width / size_ds.width;

    int nSegNr = 0, nCandCnt = 0;

    cv::Mat cvm_rgb_org(size_org, CV_8UC3);
    cv::Mat cvm_rgb_ds(size_ds, CV_8UC3);
    cv::Mat cvm_depth(size_ds, CV_8UC3);
    cv::Mat cvm_depth_smooth(size_ds, CV_8UC3);
    cv::Mat cvm_segment(size_ds, CV_8UC3);
    cv::Mat cvm_segment_tmp(size_ds, CV_8UC3);
    cv::Mat cvm_gbsegm(size_ds, CV_8UC3);
    cv::Mat cvm_track(size_ds, CV_8UC3);
    cv::Mat cvm_rec_org(size_org, CV_8UC3);
    cv::Mat cvm_rec_ds(size_ds, CV_8UC3);
    cv::Mat cvm_att(size_org, CV_8UC3);
    cv::Mat cvm_sift(size_org, CV_8UC3);

    cv::Mat cvm_cand(size_org, CV_8UC3);
    cv::Mat cvm_record_rec(size_total, CV_8UC3);
    cv::Mat cvm_record_ds(nCvHeight, nCvWidth*2 , CV_8UC3);

    cv::Mat cvm_dgradx(size_ds, CV_8UC3);
    cv::Mat cvm_dgrady(size_ds, CV_8UC3);
    cv::Mat cvm_dgrady_blur(size_ds, CV_8UC3);
    cv::Mat cvm_dgrady_smth(size_ds, CV_8UC3);

    cv::Mat cvm_depth_process(nCvHeight*2, nCvWidth*2, CV_8UC3);
    cv::Mat cvm_set_seg(50, 400, CV_8UC3);
    cv::Mat cvm_set_rec(50, 400, CV_8UC3);
    cv::Mat cvm_hist(300, 550, CV_8UC3);


    cv::VideoWriter writer_track;
    cv::VideoWriter writer_rec_org, writer_rec_ds, writer_total;


    int nTabWidth = nPadCol1 + 3*nBtnSize -13 + 360;
    cv::Mat cvm_main(nPadHeight, nTabWidth, CV_8UC3);


    std::vector<cv::Scalar> mnColorTab(50000, cv::Scalar(0, 0, 0));
    mnColorTab[0] = c_blue; mnColorTab[1] = c_green; mnColorTab[2] = c_red;
    mnColorTab[3] = c_cyan; mnColorTab[4] = c_lemon; mnColorTab[5] = c_magenta;
    mnColorTab[6] = c_turkey; mnColorTab[7] = c_kakki; mnColorTab[8] = c_orange;
    mnColorTab[9] = c_darkblue; mnColorTab[10] = c_darkgreen; mnColorTab[11] = c_darkred;
    mnColorTab[12] = c_lightblue; mnColorTab[13] = c_pink; mnColorTab[14] = c_violet;
    for (int i = 15; i < 50000; i++) {
        mnColorTab[i] = cv::Scalar(rand()%255+1, rand()%255+1, rand()%255+1);
        if (mnColorTab[i][0] < 40 && mnColorTab[i][1] < 40 && mnColorTab[i][2] < 40) {if(i) i--; else i = 0;}
    }
    /////////////////////////////////////////////////////////////////////////




    //** for the Tracking **////////////////////////////////////////////////////////////
    int nTrackHistoBin_max = nTrackHistoBin*nTrackHistoBin*nTrackHistoBin;

    std::vector<int> vnProtoIdx(nObjsNrLimit, 0);
    std::vector<int> vnProtoMemoryCnt(nObjsNrLimit, 0);
    std::vector<int> vnProtoStableCnt(nObjsNrLimit, 0);
    std::vector<int> vnProtoDisapCnt(nObjsNrLimit, 0);
    std::vector<int> vnProtoPtsCnt(nObjsNrLimit, 0);
    std::vector<std::vector<int> > mnProtoPtsIdx(nObjsNrLimit, std::vector<int>(0, 0));
    std::vector<std::vector<int> > mnProtoCenter(nObjsNrLimit, std::vector<int>(2, 0));
    std::vector<std::vector<float> > mnProtoRCenter(nObjsNrLimit, std::vector<float>(3, 0));
    std::vector<std::vector<int> > mnProtoRect(nObjsNrLimit, std::vector<int>(4, 0));
    std::vector<std::vector<float> > mnProtoCubic(nObjsNrLimit, std::vector<float>(6, 0));
    std::vector<int> vnProtoLength(nObjsNrLimit, 0);
    std::vector<std::vector<float> > mnProtoClrHist(nObjsNrLimit, std::vector<float>(nTrackHistoBin_max, 0));

    std::vector<int> vnProtoFound(nObjsNrLimit, 0);
    int nProtoCnt = 0;
    ////////////////////////////////////////////////////////////////////////////////////

    int nFoundCnt = 0, nFoundNr = 0, nFoundFrame = 0;
    int nRecogRtNr = 10; std::vector<int> vnRecogRating(nRecogRtNr, 0), vnRecogRating_tmp(nRecogRtNr, 0);

    float nDGradNan = 1000;


    bool bSwitchRecordRec = false;
    std::string sRecordSegDir; std::string sRecordSegDirPref = sDataDir + "/" + "NI_Record Segmemtation_";
    std::string sRecordRecDir; std::string sRecordRecDirPref = sDataDir + "/" + "NI_Record Recognition_";
    std::string sSnapDir; std::string sSnapDirPref = sDataDir + "/" + "NI_Snapshots_";
    std::string sVideoDir; std::string sVideoDirPref = sDataDir + "/" + "NI_Videos_";
    std::string sTimeDir; std::string sTimeDirPref = sDataDir + "/" + "NI_Time Measuring_";
    std::string sTimeFile, sTimeFile_detail;
    std::string sImgExt;
    double nDiagonal = sqrt(nCvWidth*nCvWidth + nCvHeight*nCvHeight);

    bool bSwitchDSegm = false;
    bool bSwitchRecog = false;
    bool bSwitchRecogNewCyc = true;
    bool bSwitchRecordSegm = false;
    bool bSwitchRecordTime = false;
    bool bSwitchRecordVideo = false;

    std::string sTitleDir = "/home/ni/Video/Objektfotos/";
    std::string sTitleFile = sTitleDir + sTarget + "/Title/" + sTarget + "." + "jpg";
    cv::Mat cvm_target_org = cv::imread(sTitleFile.data(), CV_LOAD_IMAGE_COLOR);
    cv::Mat cvm_target;
    if(cvm_target_org.data) {
        double sx = (double)nCvWidth/cvm_target_org.cols;
        double sy = (double)nCvHeight/cvm_target_org.rows;
        cv::resize (cvm_target_org, cvm_target, cv::Size(), min(sx,sy), min(sx,sy), cv::INTER_AREA);
    }


    nTimeRecCycle = 0;

    struct timespec t_reccyc_start, t_reccyc_end;
    struct timespec t_rec_found_start, t_rec_found_end;



    double nTimeTotal_prev = 0;
    double nTimeCurr = 0, nTimePrev = 0;
    struct timespec t_total_origin;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t_total_origin);
    int nTmpAttKeyCnt = 0, nTmpAttWidth = 0, nTmpAttHeight = 0;
    while (true) {

        d.sleep ();
        //if(bFlagEnd) break;         // finish the program
        if(bFlagEnd) printf("%d", 1/0);   // finish the program

        if (!cloud_) continue;
        //if (cloud_old_ == cloud_) continue;

        m.lock ();
        if (pcl::getFieldIndex (*cloud_, "rgb") != -1) {bPointRgb = true; pcl::fromROSMsg (*cloud_, cloud_xyz_rgb);}
        else {bPointRgb = false; pcl::fromROSMsg (*cloud_, cloud_xyz);}
        cloud_old_ = cloud_;
        m.unlock ();


        //////////////////////////////////////////////////////////////////////////////////////////////////
        //              Pre-Process
        ////////////*******************///////////////////////////////////////////////////////////////////

        struct timespec t_total_start, t_total_end; clock_gettime(CLOCK_MONOTONIC_RAW, &t_total_start);
        struct timespec t_pre_start, t_pre_end; clock_gettime(CLOCK_MONOTONIC_RAW, &t_pre_start);


        nTimePre = 0; nTimeDepth = 0; nTimeBlur = 0; nTimeTotal = 0;
        nTimeSegm = 0; nTimeTrack = 0; nTimeAtt = 0; nTimeRec = 0; nTimeSift = 0; nTimeFlann = 0;
        nTimeGbSegm = 0; nTimeSiftWhole = 0; nTimeSiftDrawWhole = 0; nTimeFlannWhole = 0;
        bTimeDepth = false; bTimeBlur = false;
        bTimeSegm = false; bTimeTrack = false; bTimeRec = false; bTimeSift = false; bTimeFlann = false;
        bTimeGbSegm = false; bTimeSiftWhole = false;

        cvm_main = cv::Scalar(0, 0, 0);
        cvm_set_seg = cv::Scalar(0, 0, 0);
        cvm_set_rec = cv::Scalar(0, 0, 0);
        cvm_rgb_org = cv::Scalar(0, 0, 0);
        cvm_rgb_ds = cv::Scalar(0, 0, 0);
        cvm_depth = cv::Scalar(0, 0, 0);
        cvm_depth_smooth = cv::Scalar(0, 0, 0);
        cvm_segment = cv::Scalar(0, 0, 0);
        cvm_segment_tmp = cv::Scalar(0, 0, 0);
        cvm_gbsegm = cv::Scalar(0, 0, 0);
        cvm_track = cv::Scalar(0, 0, 0);
        cvm_att = cv::Scalar(0, 0, 0);
        cvm_rec_org = cv::Scalar(0, 0, 0);
        cvm_rec_ds = cv::Scalar(0, 0, 0);
        cvm_sift = cv::Scalar(0, 0, 0);
        cvm_cand = cv::Scalar(0, 0, 0);
        cvm_record_rec = cv::Scalar(0, 0, 0);
        cvm_depth_process = cv::Scalar(0, 0, 0);
        cvm_dgradx = cv::Scalar(0, 0, 0);
        cvm_dgrady = cv::Scalar(0, 0, 0);
        cvm_dgrady_blur = cv::Scalar(0, 0, 0);
        cvm_dgrady_smth = cv::Scalar(0, 0, 0);

        cvm_record_ds = cv::Scalar(0, 0, 0);

        for (int i = 0; i < nTaskNrMax; i++) if (!vbFlagTask[i]) CloseWindow(i);


        // The value of nTrackHistoBin_tmp depends on the number of channels of the selected color model
        int nTrackHistoBin_tmp;
        switch(nTrackClrMode) {
        case 0: case 1: case 2: nTrackHistoBin_tmp = nTrackHistoBin_max; break;
        case 3: nTrackHistoBin_tmp = nTrackHistoBin*nTrackHistoBin; break;
        case 4: case 5: case 6: nTrackHistoBin_tmp = nTrackHistoBin_max; break;
        case 7: nTrackHistoBin_tmp = nTrackHistoBin*nTrackHistoBin; break;
        }

        ///// Setting recording mode //////////////////
        int nModeRecord = 0;
        if (vbFlagTask[stTID.nRecSegm] && vbFlagTask[stTID.nSegmentation] && !vbFlagTask[stTID.nRecognition]) nModeRecord = 1;
        else if (vbFlagTask[stTID.nRecRecog] && vbFlagTask[stTID.nSegmentation] && vbFlagTask[stTID.nRecognition]) nModeRecord = 2;
        else if (vbFlagTask[stTID.nRecSegm] && vbFlagTask[stTID.nRecRecog] && vbFlagTask[stTID.nSegmentation] && vbFlagTask[stTID.nRecognition]) nModeRecord = 3;

        ///// Setting snapshot format ////////////////
        switch (nSnapFormat) {case 0: sImgExt = ".bmp"; break; case 1: sImgExt = ".tif"; break; case 2: sImgExt = ".jpg"; break;}





        /////////////** Copying original RGB image and generating downsampled image **///////////
        cvm_image_camera.copyTo(cvm_rgb_org);
        for(size_t i = 0; i < nCvSize; i++) {
            int xx, yy;
            GetPixelPos(i, nCvWidth, xx, yy);
            cvm_rgb_ds.at<cv::Vec3b>(yy, xx) = cvm_rgb_org.at<cv::Vec3b>(yy*nImgScale, xx*nImgScale);
        }



        /////////////** copy point cloud **///////////
        pcl::PointCloud<pcl::PointXYZ> cloud_Input;
        pcl::PointCloud<pcl::PointXYZRGB> cloud_Input_rgb;
        //pcl::transformPointCloud (cloud_xyz_rgb, cloud_rotated_rgb_tmp, T_vpt);
        //pcl::copyPointCloud (cloud_rotated_rgb_tmp, cloud_Input_rgb);
        if (bPointRgb) pcl::copyPointCloud (cloud_xyz_rgb, cloud_Input_rgb);
        else pcl::copyPointCloud (cloud_xyz, cloud_Input);


        //// Set an indices for cloud ///////////////
        std::vector<int> vnCloudIdx(nCvSize);
        for (int i = 0; i < nCvSize; ++i) vnCloudIdx[i] = i;

        clock_gettime(CLOCK_MONOTONIC_RAW, &t_pre_end); nTimePre = double(timespecDiff(&t_pre_end, &t_pre_start)/nTimeRatio);



        if (vbFlagTask[stTID.nRgbOrg]) OpenWindow(stTID.nRgbOrg);
        if (vbFlagTask[stTID.nRgbDs]) OpenWindow(stTID.nRgbDs);



        /////////** Create Depth Image **////////////////////////////////////////////////////
        std::vector<float> vnX(nCvSize, 0), vnY(nCvSize, 0), vnZ(nCvSize, 0);
        std::vector<int> vnCloudIdx_d(nCvSize, 0);

        std::vector<float> vnDGradX(nCvSize, nDGradNan);
        std::vector<float> vnDGradY(nCvSize, nDGradNan);
        std::vector<float> vnDGradY_blur(nCvSize, nDGradNan);
        std::vector<float> vnDGradY_smth(nCvSize, nDGradNan);

        float nDRange = 6;
        int nDIdxCntTmp = 0;
        float nDMin = 10, nDMax = 0;
        float nDGradXMin = 99, nDGradXMax = -99;
        float nDGradYMin = 99, nDGradYMax = -99;

        if (vbFlagTask[stTID.nSegmentation] || vbFlagTask[stTID.nDepth]) {
            struct timespec t_depth_start, t_depth_end; clock_gettime(CLOCK_MONOTONIC_RAW, &t_depth_start); bTimeDepth = true;

            /////// Making depth map: store coordinates from point cloud //////////////
            if (bPointRgb) MakeDepthMap (cloud_Input_rgb, bDepthCalib, nCalibX, nCalibY, nCalibW, nCalibH, nCvSize, nCvWidth, nCvHeight, nDMax, nDMin, nDIdxCntTmp, vnCloudIdx_d, vnX, vnY, vnZ);
            else MakeDepthMap (cloud_Input, bDepthCalib, nCalibX, nCalibY, nCalibW, nCalibH, nCvSize, nCvWidth, nCvHeight, nDMax, nDMin, nDIdxCntTmp, vnCloudIdx_d, vnX, vnY, vnZ);
            ////////////////////////////////////////////////////////////////////

            MakeDGradMap (vnZ, vnCloudIdx_d, nDIdxCntTmp, nDGradConst, nDSegmDThres, nDGradNan, nCvWidth, nDGradXMin, nDGradXMax, nDGradYMin, nDGradYMax, vnDGradX, vnDGradY);
            clock_gettime(CLOCK_MONOTONIC_RAW, &t_depth_end); nTimeDepth = double(timespecDiff(&t_depth_end, &t_depth_start)/nTimeRatio);



            /////// Processing depth gradient map ////////////////////////////////////////////////////////////////////////////////////////////////////////////
            struct timespec t_blur_start, t_blur_end; clock_gettime(CLOCK_MONOTONIC_RAW, &t_blur_start); bTimeBlur = true;

            vnDGradY_blur = vnDGradY; vnDGradY_smth = vnDGradY;
            //DSegm_SmoothDepthGrad (vnDGradY, vnCloudIdx_d, size_ds, nDSegmDThres, -nDSegmDThres, nDGradNan, nDGradFilterMode, nDGradFilterSize, nDGradSmthMode, nDGradSmthCenter, nDGradSmthBnd1, nDGradSmthFac, vnDGradY_blur, vnDGradY_smth);
            DSegm_SmoothDepthGrad (vnDGradY, vnCloudIdx_d, size_ds, nDSegmDThres, -nDSegmDThres, nDGradNan, nDGradFilterMode, nDGradFilterSize, nDGradSmthMode, nDGradSmthCenter, nDGradSmthBnd1+1, nDGradSmthBnd2, nDGradSmthFac, vnDGradY_blur, vnDGradY_smth);
            //vnDGradY_smth = vnDGradY_blur;
            //DrawDepthGrad (vnDGradY_blur, vnCloudIdx_d, bDepthDispMode, nDSegmDThres, -nDSegmDThres, nDGradNan, cvm_dgrady_blur);
            //DrawDepthGrad (vnDGradY_smth, vnCloudIdx_d, bDepthDispMode, nDSegmDThres, -nDSegmDThres, nDGradNan, cvm_dgrady_smth);
            for (int i = 0; i < nDIdxCntTmp; i++) {if (vnDGradX[vnCloudIdx_d[i]] > nDGradNan*0.8 || vnDGradY[vnCloudIdx_d[i]] > nDGradNan*0.8) vnDGradY_smth[vnCloudIdx_d[i]] = nDGradNan;}

            /////// Visualizing depth and its variations /////////////////////////
            if (vbFlagTask[stTID.nDepth] || nModeRecord == 1 || nModeRecord == 3) {
                if (vbFlagTask[stTID.nDepth]) {
                    if(!vbFlagWnd[stTID.nDepth]) {
                        vbFlagWnd[stTID.nDepth] = true;
                        cv::namedWindow(vsWndName[stTID.nDepth]);
                        cvMoveWindow(vsWndName[stTID.nDepth].data(), mWndPos[stTID.nDepth][0], mWndPos[stTID.nDepth][1]);
                        int dgc = int(nDGradConst*100);
                        cvCreateTrackbar(vsTrackbarName[81].data(), vsWndName[stTID.nDepth].data(), &nDGradSmthCenter, 255, TrackbarHandler_none);
                        cvCreateTrackbar(vsTrackbarName[82].data(), vsWndName[stTID.nDepth].data(), &nDGradSmthBnd1, 30, TrackbarHandler_none);
                        cvCreateTrackbar(vsTrackbarName[83].data(), vsWndName[stTID.nDepth].data(), &nDGradSmthBnd2, 30, TrackbarHandler_none);
                        cvCreateTrackbar(vsTrackbarName[84].data(), vsWndName[stTID.nDepth].data(), &dgc, 100, TrackbarHandler_DGradC);
                    }
                }
                if (nDMax > nDRange) DrawDepth(vnZ, vnCloudIdx_d, bDepthDispMode, nDRange, nDMin, cvm_depth);
                else DrawDepth(vnZ, vnCloudIdx_d, bDepthDispMode, nDMax, nDMin, cvm_depth);

                DrawDepthGrad (vnDGradY, vnCloudIdx_d, bDepthDispMode, nDSegmDThres, -nDSegmDThres, nDGradNan, cvm_dgrady);
                DrawDepthGrad (vnDGradX, vnCloudIdx_d, bDepthDispMode, nDSegmDThres, -nDSegmDThres, nDGradNan, cvm_dgradx);
            }

            clock_gettime(CLOCK_MONOTONIC_RAW, &t_blur_end); nTimeBlur = double(timespecDiff(&t_blur_end, &t_blur_start)/nTimeRatio);
            /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        }




        nSegNr = 0;
        if (vbFlagTask[stTID.nSegmentation]){

            // Trigger
            if (!bSwitchDSegm) {bSwitchDSegm = true; nTimeTotal_acc = 0; nTimeTotal_avr = 0; nCntFrame_tmp = 0; nTimeFrame_acc = 0; nTimeFrame_avr = 0;}



            ////////////////////** Segmantation **////////////////////////////////////////////////////////////////////////
            struct timespec t_segm_start, t_segm_end; clock_gettime(CLOCK_MONOTONIC_RAW, &t_segm_start); bTimeSegm = true;

            std::vector<int> vnSegmMap(nCvSize, 0);
            std::vector<int> vnSegmMap_tmp = vnSegmMap;
            DSegmentation(vnDGradY_smth, vnCloudIdx_d, nDSegmSizeThres, nDSegmGradDist, nDGradNan, nCvWidth, nCvSize, 0, nCvWidth, 0, nCvHeight, vnSegmMap_tmp, vnSegmMap, nSegNr);
            nSegNr++;

            // Making Segments Point Indices
            std::vector<int> vnTmpPtsCnt(nSegNr, 0);
            std::vector<std::vector<int> > mnTmpPtsIdx(nSegNr, std::vector<int>(nCvSize, 0));
            Map2Objects(vnSegmMap, nSegNr, nCvSize, mnTmpPtsIdx, vnTmpPtsCnt);

            // Show Segmentation
            if (vbFlagTask[stTID.nDSegm] || nModeRecord == 1 || nModeRecord == 3){
                if (vbFlagTask[stTID.nDSegm]) OpenWindow(stTID.nDSegm);
                for (int i = 0; i < nCvSize; i++) {
                    if (vnDGradX[i] > nDGradNan*0.8 || vnDGradY[i] > nDGradNan*0.8) {
                        //AssignColor(i, mnColorTab[0], cvm_dgrady_blur);
                        //cvm_dgrady_blur.data[i*3] = 0; cvm_dgrady_blur.data[i*3+1] = 0; cvm_dgrady_blur.data[i*3+2] = 0;
                        continue;
                    }
                    //AssignColor(i, mnColorTab[vnSegmMap[i]], cvm_segment);
                    cvm_segment.data[i*3] = mnColorTab[vnSegmMap[i]][0]; cvm_segment.data[i*3+1] = mnColorTab[vnSegmMap[i]][1]; cvm_segment.data[i*3+2] = mnColorTab[vnSegmMap[i]][2];
                    //AssignColor(i, mnColorTab[vnSegmMap_tmp[i]], cvm_segment_tmp);
                    cvm_segment_tmp.data[i*3] = mnColorTab[vnSegmMap_tmp[i]][0]; cvm_segment_tmp.data[i*3+1] = mnColorTab[vnSegmMap_tmp[i]][1]; cvm_segment_tmp.data[i*3+2] = mnColorTab[vnSegmMap_tmp[i]][2];
                }
            }

            clock_gettime(CLOCK_MONOTONIC_RAW, &t_segm_end); nTimeSegm = double(timespecDiff(&t_segm_end, &t_segm_start)/nTimeRatio);




            ///////////////////////** Tracking **/////////////////////////////////////////////////////////////////////////////
            struct timespec t_track_start, t_track_end; clock_gettime(CLOCK_MONOTONIC_RAW, &t_track_start); bTimeTrack = true;

            std::vector<std::vector<int> > mnTmpRect(nSegNr, std::vector<int>(4,0));
            std::vector<std::vector<int> > mnTmpRCenter(nSegNr, std::vector<int>(2,0));
            std::vector<std::vector<float> > mnTmpCubic(nSegNr, std::vector<float>(6,0));
            std::vector<std::vector<float> > mnTmpCCenter(nSegNr, std::vector<float>(3,0));
            std::vector<int> vnTmpLength(nSegNr, 0);
            std::vector<std::vector<float> > mnTmpClrHist(nSegNr, std::vector<float> (nTrackHistoBin_max, 0));
            std::vector<int> vnTmpMemoryCnt(nSegNr, 0);
            std::vector<int> vnTmpStableCnt(nSegNr, 0);
            std::vector<int> vnTmpDisapCnt(nSegNr, 0);

            TrackingPre (nDSegmCutSize, nCvWidth, nCvHeight, vnX, vnY, vnZ, cvm_rgb_ds, nTrackClrMode, nTrackHistoBin, nTrackCntMem, nTrackCntStable, nTrackCntDisap,
                         vnTmpPtsCnt, mnTmpPtsIdx, mnTmpRect, mnTmpRCenter, mnTmpCubic, mnTmpCCenter, vnTmpLength, mnTmpClrHist, vnTmpMemoryCnt, vnTmpStableCnt, vnTmpDisapCnt, nSegNr);
            Tracking (nSegNr, nObjsNrLimit, nDiagonal, nTrackMode,
                      nTrackDPos, nTrackDSize, nTrackDClr, nTrackPFac, nTrackSFac, nTrackCFac, nTrackDist, nTrackCntMem, nTrackCntStable, nTrackCntDisap, nTrackHistoBin_tmp,
                      vnTmpPtsCnt, mnTmpPtsIdx, mnTmpRect, mnTmpRCenter, mnTmpCubic, mnTmpCCenter, vnTmpLength, mnTmpClrHist, vnTmpMemoryCnt, vnTmpStableCnt, vnTmpDisapCnt,
                      vnProtoIdx, vnProtoPtsCnt, mnProtoPtsIdx, mnProtoRect, mnProtoCenter, mnProtoCubic, mnProtoRCenter, vnProtoLength, mnProtoClrHist, vnProtoFound, vnProtoMemoryCnt, vnProtoStableCnt, vnProtoDisapCnt, nProtoCnt, vbFlagTask[stTID.nMat]);

            // Show Tracking
            if (vbFlagTask[stTID.nTrack] || ((vbFlagTask[stTID.nRecogOrg] || vbFlagTask[stTID.nRecVideo]) && nRecordMode) || nModeRecord) {
                if(vbFlagTask[stTID.nTrack] && !vbFlagWnd[stTID.nTrack]) SetFlagWnd(stTID.nTrack);

                for (int i = 0; i < nProtoCnt; i++) {
                    if (vnProtoStableCnt[i] < nTrackCntStable || vnProtoDisapCnt[i] > nTrackCntDisap) continue;
                    //Idx2Image(mnProtoPtsIdx[i], mnColorTab[vnProtoIdx[i]], cvm_track);
                    for (size_t j = 0; j < mnProtoPtsIdx[i].size(); j++) {
                        cvm_track.data[mnProtoPtsIdx[i][j]*3] = mnColorTab[vnProtoIdx[i]][0];
                        cvm_track.data[mnProtoPtsIdx[i][j]*3+1] = mnColorTab[vnProtoIdx[i]][1];
                        cvm_track.data[mnProtoPtsIdx[i][j]*3+2] = mnColorTab[vnProtoIdx[i]][2];
                    }
                }
            }



            // Show Object Candidates
            cv::Mat cvm_mask(size_org, CV_8UC3, cv::Scalar(0, 0, 0));
            if (vbFlagTask[stTID.nProto]) {
                OpenWindow(stTID.nProto);
                int xx, yy;
                for (int i = 0; i < nProtoCnt; i++) {
                    if (vnProtoStableCnt[i] < nTrackCntStable || vnProtoDisapCnt[i] > nTrackCntDisap) continue;
                    if (vnProtoLength[i] > nProtoSizeMax || vnProtoLength[i] < nProtoSizeMin || vnProtoPtsCnt[i] < nProtoPtsMin) continue;

                    for (int j = 0; j < vnProtoPtsCnt[i]; j++) {
                        GetPixelPos(mnProtoPtsIdx[i][j], nCvWidth, xx, yy);
                        cv::Vec3b s_tmp = cvm_track.at<cv::Vec3b>(yy, xx);

                        xx = xx*nImgScale; yy = yy*nImgScale;
                        if(yy > size_org.height || xx > size_org.width - nImgScale) continue;
                        cv::Vec3b s;

                        for (int mm = 0; mm < nImgScale; mm++) {
                            for (int nn = 0; nn < nImgScale; nn++) {
                                if (xx-nn < 0) continue;
                                if (vbFlagTask[stTID.nTrack]) cvm_mask.at<cv::Vec3b>(yy-mm, xx-nn) = s_tmp;
                                else {cvm_att.at<cv::Vec3b>(yy-mm, xx+nn) = cvm_rgb_org.at<cv::Vec3b>(yy-mm, xx-nn);}
                            }
                        }
                    }
                }

                if(vbFlagTask[stTID.nTrack]) cv::add(cvm_rgb_org, cvm_mask, cvm_att);
            }
            cvm_mask.release();

            clock_gettime(CLOCK_MONOTONIC_RAW, &t_track_end); nTimeTrack = double(timespecDiff(&t_track_end, &t_track_start)/nTimeRatio);





            if (vbFlagTask[stTID.nRecognition]) {
            //if (vbFlagTask[stTID.nRecognition] && (bSwitchRecordRec || !vbFlagTask[stTID.nRecRecog])) {
                struct timespec t_att_start, t_att_end; clock_gettime(CLOCK_MONOTONIC_RAW, &t_att_start);

                //////// Sorting Objects ////////////////////////
                float tmp_diff = 100;
                std::vector<std::vector<float> > vnTmpProtoDiff(2, std::vector<float>(nProtoCnt, 0));
                std::vector<bool> vbProtoCand(nObjsNrLimit, false);
                if (!nAttTDMode) {    // Top-down Selection
                    for (int i = 0; i < nProtoCnt; i++) {
                        vnTmpProtoDiff[1][i] = (float)i;
                        if (vnProtoStableCnt[i] < nTrackCntStable || vnProtoDisapCnt[i] > nTrackCntDisap) {vnTmpProtoDiff[0][i] = tmp_diff++; continue;}
                        if (vnProtoLength[i] > nProtoSizeMax || vnProtoLength[i] < nProtoSizeMin || vnProtoPtsCnt[i] < nProtoPtsMin) {vnTmpProtoDiff[0][i] = tmp_diff++; continue;}

                        vbProtoCand[i] = true;
                        float dc = 0;
                        for (int j = 0; j < nTrackHistoBin_tmp; j++) {
                            if (mnColorHistY_lib.size() == 1) dc += fabs(mnColorHistY_lib[0][j] - mnProtoClrHist[i][j]);
                            else dc += fabs(mnColorHistY_lib[nTrackClrMode][j] - mnProtoClrHist[i][j]);
                        }
                        vnTmpProtoDiff[0][i] = dc/2;
                    }
                }
                else {     // Scuffling all objects for random order (for comparison)
                    std::vector<int> number;
                    for (int i = 0; i < nProtoCnt; i++) number.push_back(i+1);
                    std::random_shuffle(number.begin(), number.end());

                    for (int i = 0; i < nProtoCnt; i++) {
                        if (vnProtoStableCnt[i] < nTrackCntStable && vnProtoDisapCnt[i] > nTrackCntDisap) {vnTmpProtoDiff[0][i] = tmp_diff++; continue;}
                        if (nAttTDMode == 1)
                            if (vnProtoLength[i] > nProtoSizeMax || vnProtoLength[i] < nProtoSizeMin || vnProtoPtsCnt[i] < nProtoPtsMin) {vnTmpProtoDiff[0][i] = tmp_diff++; continue;}

                        vbProtoCand[i] = true;
                        vnTmpProtoDiff[1][i] = (float)i;
                        vnTmpProtoDiff[0][i] = (float)number[i];
                    }
                }
                SwapMemory (vbProtoCand, vnProtoIdx, vnProtoMemoryCnt, vnProtoStableCnt, vnProtoDisapCnt, vnProtoPtsCnt, mnProtoPtsIdx, mnProtoCenter, mnProtoRCenter, mnProtoRect, mnProtoCubic, mnProtoClrHist,
                            vnProtoLength, vnProtoFound, nProtoCnt, vnTmpProtoDiff);

                clock_gettime(CLOCK_MONOTONIC_RAW, &t_att_end); nTimeAtt = double(timespecDiff(&t_att_end, &t_att_start)/nTimeRatio);



                struct timespec t_rec_start, t_rec_end; clock_gettime(CLOCK_MONOTONIC_RAW, &t_rec_start);

                if (!bSwitchRecog) {
                    bSwitchRecog = true;
                    bSwitchRecogNewCyc = true;
                    nTimeTotal_acc = 0; nTimeTotal_avr = 0; nCntFrame_tmp = 0; nTimeFrame_acc = 0; nTimeFrame_avr = 0;
                }

                if (bSwitchRecogNewCyc) {
                    clock_gettime(CLOCK_MONOTONIC_RAW, &t_reccyc_start);
                    clock_gettime(CLOCK_MONOTONIC_RAW, &t_rec_found_start);
                    bSwitchRecogNewCyc = false;
                    bTimeRecogCycle = false;
                    nFoundCnt = 0;
                    nFoundNr = 0;
                    nCandCnt = 0;
                    nTimeRecFound = 0;
                    nTmpAttKeyCnt = 0; nTmpAttWidth = 0; nTmpAttHeight = 0;


                    for (int i = 0; i < nProtoCnt*2; i++) {
                        if (vnProtoFound[i] > 2) vnProtoFound[i] = 1;
                        else vnProtoFound[i] = 0;
                    }
                    vnRecogRating_tmp.assign(nRecogRtNr, 0);
                }



                bTimeRec = true;

                int nCandID = -1;
                SelRecognition_Pre (nProtoCnt, vbProtoCand, vnProtoFound, nCandID);

                if (vbFlagTask[stTID.nRecogOrg] || (vbFlagTask[stTID.nRecVideo] && nRecordMode) || nModeRecord == 2 || nModeRecord == 3) {
                    if (vbFlagTask[stTID.nRecogOrg]) OpenWindow(stTID.nRecogOrg);
                    cvm_rgb_org.copyTo(cvm_rec_org);
                }
                if (vbFlagTask[stTID.nRecogDs] || ((vbFlagTask[stTID.nRecogOrg] || vbFlagTask[stTID.nRecVideo]) && nRecordMode) || nModeRecord == 2 || nModeRecord == 3) {
                    if (vbFlagTask[stTID.nRecogDs]) OpenWindow(stTID.nRecogDs);
                    cvm_rgb_ds.copyTo(cvm_rec_ds);
                }


                if (nCandID < 0) {
                    clock_gettime(CLOCK_MONOTONIC_RAW, &t_reccyc_end); nTimeRecCycle = double(timespecDiff(&t_reccyc_end, &t_reccyc_start)/nTimeRatio);


                    //printf("\n");
                    bSwitchRecogNewCyc = true;
                    if (nCandCnt) {
                        if (vbFlagTask[stTID.nRecTime] && bSwitchRecordTime) {
                            for (size_t i = 0; i < mnTimeMeas1.size(); i++) {
                                if (mnTimeMeas1[i][0] == nCntRecCycle+1) mnTimeMeas1[i][3] = nCandCnt;
                            }
                        }
                        nCntRecCycle++;
                        nTimeRecCycle_acc += nTimeRecCycle; nTimeRecCycle_avr = nTimeRecCycle_acc/nCntRecCycle;
                        nTimeRecFound_acc += nTimeRecFound; nTimeRecFound_avr = nTimeRecFound_acc/nCntRecCycle;

                        clock_gettime(CLOCK_MONOTONIC_RAW, &t_reccyc_start);
                        clock_gettime(CLOCK_MONOTONIC_RAW, &t_rec_found_start);
                        bSwitchRecogNewCyc = false;
                        bTimeRecogCycle = false;
                        nFoundCnt = 0;
                        nFoundNr = 0;
                        nCandCnt = 0;
                        nTimeRecFound = 0;
                        nTmpAttKeyCnt = 0; nTmpAttWidth = 0; nTmpAttHeight = 0;

                        for (int i = 0; i < nProtoCnt*2; i++) {
                            if (vnProtoFound[i] > 2) vnProtoFound[i] = 1;
                            else vnProtoFound[i] = 0;
                        }
                        vnRecogRating_tmp.assign(nRecogRtNr, 0);

                        SelRecognition_Pre (nProtoCnt, vbProtoCand, vnProtoFound, nCandID);
                    }

                    if (((vbFlagTask[stTID.nRecogOrg] || vbFlagTask[stTID.nRecVideo]) && nRecordMode)) {
                        switch (nRecordMode) {
                        case 1: {
                            cv::Mat dst;
                            dst = cvm_record_rec (cv::Rect(nCvWidth + 2*nBorderThickness, nBorderThickness, cvm_rec_org.cols, cvm_rec_org.rows)); cvm_rec_org.copyTo(dst);
                            dst = cvm_record_rec (cv::Rect(nBorderThickness, cvm_record_rec.rows - nCvHeight -  nBorderThickness, nCvWidth, nCvHeight)); cvm_rec_ds.copyTo(dst);
                            dst = cvm_record_rec (cv::Rect(nBorderThickness, cvm_record_rec.rows - 2*nCvHeight - nTabThickness - nBorderThickness, nCvWidth, nCvHeight)); cvm_track.copyTo(dst);
                            dst = cvm_record_rec (cv::Rect(nBorderThickness+(nCvWidth-cvm_target.cols)/2, nBorderThickness+1.5*nTabThickness+(nCvHeight-cvm_target.rows)/2, cvm_target.cols, cvm_target.rows)); cvm_target.copyTo(dst);
                            dst.release();

                            cv::line (cvm_record_rec, cv::Point(nCvWidth+nBorderThickness, 0), cv::Point(nCvWidth+nBorderThickness, cvm_record_rec.rows), c_darkgray, nBorderThickness);
                            cv::rectangle (cvm_record_rec, cv::Point(0, 1), cv::Point(cvm_record_rec.cols-1, cvm_record_rec.rows-1), c_darkgray, nBorderThickness);
                            cv::rectangle (cvm_record_rec, cv::Point(0, cvm_record_rec.rows-nCvHeight-nTabThickness-nBorderThickness), cv::Point(nCvWidth, cvm_record_rec.rows-nCvHeight-nBorderThickness), c_darkgray, CV_FILLED);
                            cv::rectangle (cvm_record_rec, cv::Point(0, cvm_record_rec.rows-2*nCvHeight-2*nTabThickness-nBorderThickness), cv::Point(nCvWidth, cvm_record_rec.rows-2*nCvHeight-nTabThickness-nBorderThickness), c_darkgray, CV_FILLED);

                            cv::putText (cvm_record_rec, "Tracked Surfaces", cv::Point(nTabThickness, cvm_record_rec.rows-2*nCvHeight-1.3*nTabThickness-nBorderThickness), nFont, 0.4, c_white, 1);
                            cv::putText (cvm_record_rec, "Down-Sampled", cv::Point(nTabThickness, cvm_record_rec.rows-nCvHeight-0.3*nTabThickness-nBorderThickness), nFont, 0.4, c_white, 1);
                            cv::putText (cvm_record_rec, "Candidate", cv::Point(nTabThickness, nBorderThickness+nCvHeight+3*nTabThickness), nFont, 0.6, c_cyan, 1);
                            cv::putText (cvm_record_rec, "Target", cv::Point(nTabThickness, nBorderThickness+nTabThickness), nFont, 0.6, c_lemon, 1);

                            break;}
                        case 2: {
                            cv::Mat dst;
                            dst = cvm_record_ds (cv::Rect(0, 0, cvm_track.cols, cvm_track.rows)); cvm_track.copyTo(dst);
                            dst = cvm_record_ds (cv::Rect(nCvWidth, 0, cvm_rec_ds.cols, cvm_rec_ds.rows)); cvm_rec_ds.copyTo(dst);
                            dst.release();
                            break;}
                        }
                    }
                }

                if (nCandID >= 0) {
                    switch (nTrackClrMode) {
                    case 0: printf("Color distance (RGB)   "); break;
                    case 1: printf("Color distance (normalized rgb)   "); break;
                    case 2: printf("Color distance (Color plane directions)   "); break;
                    case 3: printf("Lab"); break;
                    case 4: printf("Color distance (RGB)  2"); break;
                    case 5: printf("Color distance (normalized rgb)   2"); break;
                    case 6: printf("Color distance (Color plane directions)     "); break;
                    case 7: printf("Lab       2"); break;
                    }

                    for (int i = 0; i < nProtoCnt; i++) {
                        if(!vbProtoCand[i]) continue;
                        printf("%8.3f", vnTmpProtoDiff[0][i]);
                    }
                    printf("\n");




                    int nCandKeyCnt, nCandRX, nCandRY, nCandRW, nCandRH;
                    SelRecognition (nCandID, nImgScale, nTimeRatio, nProtoCnt, nTrackHistoBin_max, sTimeDir, sImgExt, cvm_rgb_org, cvm_rgb_ds, cvm_rec_org, cvm_rec_ds,
                                    vnProtoPtsCnt, mnProtoPtsIdx, mnProtoRect, vnProtoFound, vnProtoLength, vnProtoDisapCnt, nTrackCntDisap, nCandCnt, vnTmpProtoDiff, nFoundCnt, nFoundNr, nFoundFrame,
                                    nCandKeyCnt, nCandRX, nCandRY, nCandRW, nCandRH,
                                    t_rec_found_start, t_rec_found_end, bSwitchRecordTime, nRecogRtNr, vnRecogRating_tmp, nModeRecord, cvm_cand);
                                    //mnProtoClrHist, mnProtoCenter);
                    nTmpAttKeyCnt = nCandKeyCnt; nTmpAttWidth = nCandRW; nTmpAttHeight = nCandRH;

                    if (((vbFlagTask[stTID.nRecogOrg] || vbFlagTask[stTID.nRecVideo]) && nRecordMode)) {
                        switch (nRecordMode) {
                        case 1: {
                            cv::Mat src, dst;
                            dst = cvm_record_rec (cv::Rect(nCvWidth + 2*nBorderThickness, nBorderThickness, cvm_rec_org.cols, cvm_rec_org.rows)); cvm_rec_org.copyTo(dst);
                            dst = cvm_record_rec (cv::Rect(nBorderThickness, cvm_record_rec.rows - nCvHeight -  nBorderThickness, nCvWidth, nCvHeight)); cvm_rec_ds.copyTo(dst);
                            dst = cvm_record_rec (cv::Rect(nBorderThickness, cvm_record_rec.rows - 2*nCvHeight - nTabThickness - nBorderThickness, nCvWidth, nCvHeight)); cvm_track.copyTo(dst);
                            dst = cvm_record_rec (cv::Rect(nBorderThickness+(nCvWidth-cvm_target.cols)/2, nBorderThickness+1.5*nTabThickness+(nCvHeight-cvm_target.rows)/2, cvm_target.cols, cvm_target.rows)); cvm_target.copyTo(dst);

                            double cx = (double)nCvWidth/nCandRW, cy = (double)nCvHeight/nCandRH; if (cx > 1) cx = 1; if (cy > 1) cy = 1;
                            double ratio = min(cx,cy)*0.7;
                            int nCX = (nCvWidth-nCandRW*ratio)/2, nCY = (nCvHeight-nCandRH*ratio)/2;
                            cv::Mat src_ds;
                            src = cvm_cand (cv::Rect(nCandRX, nCandRY, nCandRW, nCandRH));
                            cv::resize (src, src_ds, cv::Size(), ratio, ratio, cv::INTER_AREA);
                            dst = cvm_record_rec (cv::Rect(nBorderThickness + nCX, nBorderThickness+nCvHeight+2*nTabThickness + nCY, src_ds.cols, src_ds.rows)); src_ds.copyTo(dst);
                            src.release(); dst.release(); src_ds.release();

                            cv::line (cvm_record_rec, cv::Point(nCvWidth+nBorderThickness, 0), cv::Point(nCvWidth+nBorderThickness, cvm_record_rec.rows), c_darkgray, nBorderThickness);
                            cv::rectangle (cvm_record_rec, cv::Point(0, 1), cv::Point(cvm_record_rec.cols-1, cvm_record_rec.rows-1), c_darkgray, nBorderThickness);
                            cv::rectangle (cvm_record_rec, cv::Point(0, cvm_record_rec.rows-nCvHeight-nTabThickness-nBorderThickness), cv::Point(nCvWidth, cvm_record_rec.rows-nCvHeight-nBorderThickness), c_darkgray, CV_FILLED);
                            cv::rectangle (cvm_record_rec, cv::Point(0, cvm_record_rec.rows-2*nCvHeight-2*nTabThickness-nBorderThickness), cv::Point(nCvWidth, cvm_record_rec.rows-2*nCvHeight-nTabThickness-nBorderThickness), c_darkgray, CV_FILLED);

                            cv::putText (cvm_record_rec, "Tracked Surfaces", cv::Point(nTabThickness, cvm_record_rec.rows-2*nCvHeight-1.3*nTabThickness-nBorderThickness), nFont, 0.4, c_white, 1);
                            cv::putText (cvm_record_rec, "Down-Sampled", cv::Point(nTabThickness, cvm_record_rec.rows-nCvHeight-0.3*nTabThickness-nBorderThickness), nFont, 0.4, c_white, 1);
                            cv::putText (cvm_record_rec, "Candidate", cv::Point(nTabThickness, nBorderThickness+nCvHeight+3*nTabThickness), nFont, 0.6, c_cyan, 1);
                            cv::putText (cvm_record_rec, "Target", cv::Point(nTabThickness, nBorderThickness+nTabThickness), nFont, 0.6, c_lemon, 1);

                            break;}
                        case 2: {
                            cv::Mat dst;
                            dst = cvm_record_ds (cv::Rect(0, 0, cvm_track.cols, cvm_track.rows)); cvm_track.copyTo(dst);
                            dst = cvm_record_ds (cv::Rect(nCvWidth, 0, cvm_rec_ds.cols, cvm_rec_ds.rows)); cvm_rec_ds.copyTo(dst);
                            dst.release();
                            break;}
                        }
                    }
                }

                clock_gettime(CLOCK_MONOTONIC_RAW, &t_rec_end); nTimeRec = double(timespecDiff(&t_rec_end, &t_rec_start)/nTimeRatio);
            }// vbFlagTask[stTID.nRecognition]
            else {
                if (bSwitchRecog) {
                    bSwitchRecog = false;

                    bSwitchRecogNewCyc = true;

                    vbFlagTask[stTID.nRecTime] = false;
                    mnTimeMeas1.resize(0); mnTimeMeas2.resize(0);
                    ResetMemory (nObjsNrLimit, nTrackHistoBin_max, nRecogRtNr, vnProtoIdx, vnProtoMemoryCnt, vnProtoStableCnt, vnProtoDisapCnt, vnProtoPtsCnt, mnProtoPtsIdx, mnProtoCenter, mnProtoRCenter, mnProtoRect, mnProtoCubic, mnProtoClrHist,
                                 vnProtoLength, vnProtoFound, nProtoCnt, nCandCnt, nFoundCnt, nFoundNr, vnRecogRating);

                    ResetRecTime();
                }

                if (vbFlagWnd[stTID.nRecogOrg]) {cvm_rec_org = cv::Scalar(0, 0, 0); cv::add(cvm_rec_org, cvm_rgb_org, cvm_rec_org);}
                if (vbFlagWnd[stTID.nRecogDs]) {cvm_rec_ds = cv::Scalar(0, 0, 0); cv::add(cvm_rec_ds, cvm_rgb_ds, cvm_rec_ds);}
            }
        }//bFragTask[stTID.nSegmentation]
        else {
            if (bSwitchDSegm) {
                bSwitchDSegm = false;
                bSwitchRecog = false;

                bSwitchRecogNewCyc = true;

                vbFlagTask[stTID.nRecTime] = false;
                mnTimeMeas1.resize(0); mnTimeMeas2.resize(0);
                ResetMemory (nObjsNrLimit, nTrackHistoBin_max, nRecogRtNr, vnProtoIdx, vnProtoMemoryCnt, vnProtoStableCnt, vnProtoDisapCnt, vnProtoPtsCnt, mnProtoPtsIdx, mnProtoCenter, mnProtoRCenter, mnProtoRect, mnProtoCubic, mnProtoClrHist,
                             vnProtoLength, vnProtoFound, nProtoCnt, nCandCnt, nFoundCnt, nFoundNr, vnRecogRating);

                nCntSegm = 0; nTimeSegm_acc = 0; nTimeSegm_avr = 0;
                nCntTrack = 0; nTimeTrack_acc = 0; nTimeTrack_avr = 0;

                ResetRecTime();

                if (!vbFlagTask[stTID.nDepth]) {nCntDepth = 0; nCntBlur = 0; nTimeDepth_acc = 0; nTimeBlur_acc = 0; nTimeDepth_avr = 0; nTimeBlur_avr = 0;}
            }

            if (vbFlagWnd[stTID.nRecogOrg]) {cvm_rec_org = cv::Scalar(0, 0, 0); cv::add(cvm_rec_org, cvm_rgb_org, cvm_rec_org);}
            if (vbFlagWnd[stTID.nRecogDs]) {cvm_rec_ds = cv::Scalar(0, 0, 0); cv::add(cvm_rec_ds, cvm_rgb_ds, cvm_rec_ds);}
        }
        /////////////***************************//////////////////////////////////////////////////////////
        //              End of main process
        //////////////////////////////////////////////////////////////////////////////////////////////////



        //////////////////////////////////////////////////////////////////////////////////////////////////
        //              Other state of art methods for compairison
        ////////////***************************************************///////////////////////////////////

        //////// 1. Graph-based segmentation ////////////////////////////////
        if (vbFlagTask[stTID.nGSegm]) {
            struct timespec t_gbseg_start, t_gbseg_end; clock_gettime(CLOCK_MONOTONIC_RAW, &t_gbseg_start); bTimeGbSegm = true;
            if(!vbFlagWnd[stTID.nGSegm]) {
                SetFlagWnd(stTID.nGSegm);

                int gsegm_sigma = int(nGSegmSigma*10);
                cvCreateTrackbar(vsTrackbarName[17].data(), vsWndName[stTID.nGSegm].data(), &gsegm_sigma, 10, TrackbarHandler_GSegmSigma);
                cvCreateTrackbar(vsTrackbarName[18].data(), vsWndName[stTID.nGSegm].data(), &nGSegmGrThrs, 500, TrackbarHandler_none);
                cvCreateTrackbar(vsTrackbarName[19].data(), vsWndName[stTID.nGSegm].data(), &nGSegmMinSize, 1000, TrackbarHandler_none);
            }

            std::vector<std::vector<CvPoint> > mnGSegmPts;
            GbSegmentation(cvm_rgb_ds, nGSegmSigma, nGSegmGrThrs, nGSegmMinSize, mnGSegmPts);

            int idx = 0;
            std::vector<std::vector<int> > mnGSegmPtsIdx(mnGSegmPts.size(), std::vector<int>(10,0));
            for (size_t i = 0; i < mnGSegmPts.size(); i++) {
                mnGSegmPtsIdx[i].resize(mnGSegmPts[i].size());
                for(size_t j = 0; j < mnGSegmPts[i].size(); j++) {
                    idx = GetPixelIdx(mnGSegmPts[i][j].x, mnGSegmPts[i][j].y, nCvWidth);
                    mnGSegmPtsIdx[i][j] = idx;
                    cvm_gbsegm.at<cv::Vec3b>(mnGSegmPts[i][j].y, mnGSegmPts[i][j].x) = cv::Vec3b(mnColorTab[i][0], mnColorTab[i][1], mnColorTab[i][2]);
                }
            }

            clock_gettime(CLOCK_MONOTONIC_RAW, &t_gbseg_end); nTimeGbSegm = double(timespecDiff(&t_gbseg_end, &t_gbseg_start)/nTimeRatio);


            ////// Recofd Time ///////
            if (bRtGbSegm) {
                mnTimeMeasGbSegm.resize(nMeasCounter+1);
                mnTimeMeasGbSegm[nMeasCounter] = nTimeGbSegm;
                nMeasCounter++;

                if (nMeasCounter > nTimeMessFrmLimit -1) {
                    bRtGbSegm = false;
                    nMeasCounter = 0;
                    sTimeFile_detail = "time_gbsegmentation.xlsx";
                    PrintTimesFromVector(sTimeFile_detail, mnTimeMeasGbSegm);
                }
            }
        }



        //////// 2. Sift on whole image ////////////////////////////////
        if (vbFlagTask[stTID.nSIFT]) {
            struct timespec t_sift_start, t_sift_end; clock_gettime(CLOCK_MONOTONIC_RAW, &t_sift_start); bTimeSiftWhole = true;
//            if(!vbFlagWnd[stTID.nSIFT]) {
//                vbFlagWnd[stTID.nSIFT] = true;
//                cv::namedWindow(vsWndName[stTID.nSIFT]);
//                cvMoveWindow(vsWndName[stTID.nSIFT].data(), 300, 100);

//                cvCreateTrackbar(vsTrackbarName[59].data(), vsWndName[stTID.nSIFT].data(), &nFlannMatchCntWhole, 100, TrackbarHandler_none);
//            }

            Keypoint keypts, keypts_tmp;
            GetSiftKeypoints(cvm_rgb_org, nSiftScales, nSiftInitSigma, nSiftPeakThrs, 0, 0, cvm_rgb_org.cols, cvm_rgb_org.rows, keypts);
            keypts_tmp = keypts;

            clock_gettime(CLOCK_MONOTONIC_RAW, &t_sift_end); nTimeSiftWhole = double(timespecDiff(&t_sift_end, &t_sift_start)/nTimeRatio);


            struct timespec t_flann_start, t_flann_end; clock_gettime(CLOCK_MONOTONIC_RAW, &t_flann_start);

            std::vector<int> vnSiftMatched; std::vector <double> vnDeltaScale; std::vector <double> vnDeltaOri;
            double nMaxDeltaOri = -999; double nMaxDeltaScale = -999;
            double nMinDeltaOri = 999; //double nMinDeltaScale = 999;
            int nKeyptsCnt = 0; int nFlannIM=0;


            SelRecognition_FlannSift(tcount, nFlannKnn, nFlannLibCols_sift, nFlannMatchFac, mnSiftExtraFeatures, FlannIdx_Sift, FLANNParam, keypts, nKeyptsCnt, nFlannIM, vnSiftMatched, vnDeltaScale, vnDeltaOri, nMaxDeltaOri, nMinDeltaOri, nMaxDeltaScale, nMinDeltaOri);


            /////////////////////////////////////////Filtering Gray Keypoints
            double nDeltaScale=0;
            int nFlannTP=0;

            std::vector<bool> vbSiftTP(nKeyptsCnt, 0);
            if(nFlannIM > T_numb) CalcDeltaScaleOri(vnSiftMatched, vnDeltaScale, vnDeltaOri, nDeltaScale, nDeltaBinNo, nMaxDeltaOri, nMinDeltaOri, T_orient, T_scale, nFlannTP, vbSiftTP);
            else nFlannTP = 0;

            clock_gettime(CLOCK_MONOTONIC_RAW, &t_flann_end); nTimeFlannWhole = double(timespecDiff(&t_flann_end, &t_flann_start)/nTimeRatio);


            struct timespec t_siftdraw_start, t_siftdraw_end; clock_gettime(CLOCK_MONOTONIC_RAW, &t_siftdraw_start);
            cv::add(cvm_sift, cvm_rgb_org, cvm_sift);
            keypts = keypts_tmp;
            while(keypts) {
                cv::rectangle(cvm_sift, cv::Point(keypts->row - 2, keypts->col - 2), cv::Point(keypts->row + 2, keypts->col + 2), c_white, 1);
                keypts= keypts->next;
            }
            keypts = keypts_tmp;
            nKeyptsCnt = 0;
            while(keypts) {
                if(vbSiftTP[nKeyptsCnt]) {
                    cv::rectangle(cvm_sift, cv::Point(keypts->row - 3, keypts->col - 3), cv::Point(keypts->row + 3, keypts->col + 3), c_blue, 1);
                    cv::rectangle(cvm_sift, cv::Point(keypts->row - 1, keypts->col - 1), cv::Point(keypts->row + 1, keypts->col + 1), c_blue, 1);
                }
                nKeyptsCnt++;
                keypts= keypts->next;
            }
            clock_gettime(CLOCK_MONOTONIC_RAW, &t_siftdraw_end); nTimeSiftDrawWhole = double(timespecDiff(&t_siftdraw_end, &t_siftdraw_start)/nTimeRatio);

            FreeKeypoints(keypts); //FreeKeypoints(keypts_tmp);


            /////// Record Time ///////////
            if (bRtSiftW) {
                mnTimeMeasSiftWhole.resize(nMeasCounter+1);
                mnTimeMeasSiftWhole[nMeasCounter].assign(3,0);
                mnTimeMeasSiftWhole[nMeasCounter][0] = nTimeSiftWhole;
                mnTimeMeasSiftWhole[nMeasCounter][1] = nTimeFlannWhole;
                mnTimeMeasSiftWhole[nMeasCounter][2] = (float)nKeyptsCnt;
                nMeasCounter++;

                if (nMeasCounter > nTimeMessFrmLimit -1) {
                    bRtSiftW = false;
                    nMeasCounter = 0;
                    sTimeFile_detail = "time_siftwhole.xlsx";
                    PrintTimesSiftWhole(sTimeFile_detail, mnTimeMeasSiftWhole, cvm_rgb_org.cols, cvm_rgb_org.rows);
                }
            }
        }
        /////////////************************************************************/////////////////////////
        //                End of ther state of art methods for compairison
        //////////////////////////////////////////////////////////////////////////////////////////////////




        ////////// Recording snapshots for Segmentation /////////////////////////////////////
        if (vbFlagTask[stTID.nRecSegm]){
            if (nModeRecord == 1) {
                if (!bSwitchRecordSegm) {
                    bSwitchRecordSegm = true;
                    mkdir(sDataDir.data(), 0777); MakeNiDirectory (sRecordSegDirPref, sRecordSegDir);
                }
                else {
                    std::string sDat;

                    char s_frame[128];
                    switch (nModeRecord) {
                    case 1:
                        sprintf(s_frame, "snap_%06i_01_rgb", nCntFrame); sDat = sRecordSegDir + "/" + s_frame + sImgExt; cv::imwrite(sDat, cvm_rgb_ds);
                        //sprintf(s_frame, "snap_%06i_02_dpt", nCntFrame); sDat = sRecordSegDir + "/" + s_frame + sImgExt; cv::imwrite(sDat, cvm_depth);
                        //sprintf(s_frame, "snap_%06i_03_dgy", nCntFrame); sDat = sRecordSegDir + "/" + s_frame + sImgExt; cv::imwrite(sDat, cvm_dgrady);
                        //sprintf(s_frame, "snap_%06i_04_dgb", nCntFrame); sDat = sRecordSegDir + "/" + s_frame + sImgExt; cv::imwrite(sDat, cvm_dgrady_blur);
                        //sprintf(s_frame, "snap_%06i_05_seg_nopostproc", nCntFrame); sDat = sRecordSegDir + "/" + s_frame + sImgExt; cv::imwrite(sDat, cvm_segment_tmp);
                        //sprintf(s_frame, "snap_%06i_06_seg", nCntFrame); sDat = sRecordSegDir + "/" + s_frame + sImgExt; cv::imwrite(sDat, cvm_segment);
                        break;
                    }
                    sprintf(s_frame, "snap_%06i_07_trk", nCntFrame); sDat = sRecordSegDir + "/" + s_frame + sImgExt; cv::imwrite(sDat, cvm_track);
                }
            }
        }
        else bSwitchRecordSegm = false;


        ////////// Recording snapshots for Recognition /////////////////////////////////////
        if (vbFlagTask[stTID.nRecRecog]){
            if (nModeRecord == 2 || nModeRecord == 3) {
                if (!bSwitchRecordRec) {
                    bSwitchRecordRec = true;
                    mkdir(sDataDir.data(), 0777); MakeNiDirectory (sRecordRecDirPref, sRecordRecDir);
                }
                else {
                    std::string sDat;
                    char s_frame[128];
                    switch (nModeRecord) {
                    case 2:
                        if (vbFlagWnd[stTID.nRecogOrg]) {
                            sprintf(s_frame, "snap_%06i_08_rec", nCntFrame); sDat = sRecordRecDir + "/" + s_frame + sImgExt; cv::imwrite(sDat, cvm_rec_org);
                        }
                        if (vbFlagWnd[stTID.nRecogDs] || !vbFlagWnd[stTID.nRecogOrg]) {
                            sprintf(s_frame, "snap_%06i_08_rec_ds", nCntFrame); sDat = sRecordRecDir + "/" + s_frame + sImgExt; cv::imwrite(sDat, cvm_rec_ds);
                            //sprintf(s_frame, "snap_%06i_09_recc", nCntFrame); sDat = sRecordRecDir + "/" + s_frame + sImgExt; cv::imwrite(sDat, cvm_hist);
                        }
                        break;
                    case 3:
                        sprintf(s_frame, "snap_%06i_01_rgb", nCntFrame); sDat = sRecordRecDir + "/" + s_frame + sImgExt; cv::imwrite(sDat, cvm_rgb_ds);
                        sprintf(s_frame, "snap_%06i_02_dpt", nCntFrame); sDat = sRecordRecDir + "/" + s_frame + sImgExt; cv::imwrite(sDat, cvm_depth);
                        sprintf(s_frame, "snap_%06i_03_dgy", nCntFrame); sDat = sRecordRecDir + "/" + s_frame + sImgExt; cv::imwrite(sDat, cvm_dgrady);
                        sprintf(s_frame, "snap_%06i_04_dgb", nCntFrame); sDat = sRecordRecDir + "/" + s_frame + sImgExt; cv::imwrite(sDat, cvm_dgrady_blur);
                        sprintf(s_frame, "snap_%06i_05_seg_nopostproc", nCntFrame); sDat = sRecordRecDir + "/" + s_frame + sImgExt; cv::imwrite(sDat, cvm_segment_tmp);
                        sprintf(s_frame, "snap_%06i_06_seg", nCntFrame); sDat = sRecordRecDir + "/" + s_frame + sImgExt; cv::imwrite(sDat, cvm_segment);
                        sprintf(s_frame, "snap_%06i_08_rec", nCntFrame); sDat = sRecordRecDir + "/" + s_frame + sImgExt; cv::imwrite(sDat, cvm_rec_org);
                        sprintf(s_frame, "snap_%06i_08_rec_ds", nCntFrame); sDat = sRecordRecDir + "/" + s_frame + sImgExt; cv::imwrite(sDat, cvm_rec_ds);
                        break;
                    }
                    sprintf(s_frame, "snap_%06i_07_trk", nCntFrame); sDat = sRecordRecDir + "/" + s_frame + sImgExt; cv::imwrite(sDat, cvm_track);
                }
            }
        }
        else bSwitchRecordRec = false;


        ////** Snapping images once **/////////////////
        if (vbFlagTask[stTID.nSnap]){
            mkdir(sDataDir.data(), 0777); MakeNiDirectory (sSnapDirPref, sSnapDir);
            std::string sDat;
            sDat = sSnapDir + "/" + "snap_00_org" + sImgExt; cv::imwrite(sDat, cvm_rgb_org);
            sDat = sSnapDir + "/" + "snap_01_rgb" + sImgExt; cv::imwrite(sDat, cvm_rgb_ds);
            sDat = sSnapDir + "/" + "snap_02_dpt" + sImgExt; cv::imwrite(sDat, cvm_depth);
            sDat = sSnapDir + "/" + "snap_03_dgy" + sImgExt; cv::imwrite(sDat, cvm_dgrady);
            sDat = sSnapDir + "/" + "snap_03_dgx" + sImgExt; cv::imwrite(sDat, cvm_dgradx);
            sDat = sSnapDir + "/" + "snap_04_dgy_blur" + sImgExt; cv::imwrite(sDat, cvm_dgrady_blur);
            sDat = sSnapDir + "/" + "snap_04_dgy_smooth" + sImgExt; cv::imwrite(sDat, cvm_dgrady_smth);
            sDat = sSnapDir + "/" + "snap_05_seg_noPostProc" + sImgExt; cv::imwrite(sDat, cvm_segment_tmp);
            sDat = sSnapDir + "/" + "snap_06_seg" + sImgExt; cv::imwrite(sDat, cvm_segment);
            sDat = sSnapDir + "/" + "snap_07_trk" + sImgExt; cv::imwrite(sDat, cvm_track);
            sDat = sSnapDir + "/" + "snap_08_obj" + sImgExt; cv::imwrite(sDat, cvm_att);
            sDat = sSnapDir + "/" + "snap_08_rec" + sImgExt; cv::imwrite(sDat, cvm_rec_org);
            sDat = sSnapDir + "/" + "snap_08_rec_ds" + sImgExt; cv::imwrite(sDat, cvm_rec_ds);

            cv::Mat cvm_tmp_tmp(cvm_rgb_ds.size(), cvm_rgb_ds.type());

            cvm_tmp_tmp = cv::Scalar(0, 0, 0);
            for (size_t i = 0; i < vnDGradY.size(); i++) {
                if (vnDGradX[i] >= nDGradNan*0.8 || vnDGradY[i] >= nDGradNan*0.8) {cvm_tmp_tmp.data[3*i] = 0; cvm_tmp_tmp.data[3*i+1] = 0; cvm_tmp_tmp.data[3*i+2] = 0;}
                else {cvm_tmp_tmp.data[3*i] = 192; cvm_tmp_tmp.data[3*i+1] = 192; cvm_tmp_tmp.data[3*i+2] = 192;}
            }
            //cv::Mat cvm_snap_tmp(cv::Size(nCropW, nCropH), cvm_rgb_ds.type());
            //accCutImage(cvm_tmp_tmp, cvm_snap_tmp, nCropX, nCropY, nCropW, nCropH);
            sDat = sSnapDir + "/" + "snap_03_dgl" + sImgExt; cv::imwrite(sDat, cvm_tmp_tmp);

            cvm_tmp_tmp = cv::Scalar(0, 0, 0);
            cv::add(cvm_tmp_tmp, cvm_dgrady_smth, cvm_tmp_tmp);
            for (size_t i = 0; i < vnDGradY.size(); i++) if (vnDGradX[i] >= nDGradNan*0.8 || vnDGradY[i] >= nDGradNan*0.8) {cvm_tmp_tmp.data[3*i] = 0; cvm_tmp_tmp.data[3*i+1] = 0; cvm_tmp_tmp.data[3*i+2] = 0;}
            sDat = sSnapDir + "/" + "snap_04_dgy_pre" + sImgExt; cv::imwrite(sDat, cvm_tmp_tmp);


            cvm_tmp_tmp.release();

            if (vbFlagWnd[stTID.nGSegm]) {sDat = sSnapDir + "/" + "snap_06_seg_berkeley" + sImgExt; cv::imwrite(sDat, cvm_gbsegm);}
            if (vbFlagWnd[stTID.nSIFT]) {sDat = sSnapDir + "/" + "snap_09_rec_whole" + sImgExt; cv::imwrite(sDat, cvm_sift);}
        }


        ////** Record videos **/////////////////
        if (vbFlagTask[stTID.nRecVideo]){
            if (!bSwitchRecordVideo) {
                bSwitchRecordVideo = true;
                mkdir(sDataDir.data(), 0777);  MakeNiDirectory (sVideoDirPref, sVideoDir);
                std::string sDat;
                int fourcc = CV_FOURCC('D', 'I', 'V', 'X'); // MPEG-4 codec
                //int fourcc = CV_FOURCC('X', 'V', 'I', 'D'); // MPEG-4 codec
                //int fourcc = CV_FOURCC('P','I','M','1'); // MPEG-1 codec
                //int fourcc = CV_FOURCC('M', 'J', 'P', 'G'); // Motion Jpeg
                int frame_rate = int(nFrameRate_avr); if (frame_rate > 30) frame_rate = 30; if (frame_rate < 15) frame_rate = 15;

                double fr = 18.0;
                switch (nRecordMode) {
                case 0:
                    if (vbFlagWnd[stTID.nTrack]) {sDat = sVideoDir + "/" + "video_07_trk" + ".avi"; writer_track.open(sDat, fourcc, (double)frame_rate, size_ds, true);}
                    if (vbFlagWnd[stTID.nRecogOrg]) {sDat = sVideoDir + "/" + "video_08_rec" + ".avi"; writer_rec_org.open(sDat, fourcc, (double)frame_rate, size_org, true);}
                    if (vbFlagWnd[stTID.nRecogDs]) {sDat = sVideoDir + "/" + "video_08_rec_ds" + ".avi"; writer_rec_ds.open(sDat, fourcc, (double)frame_rate, size_ds, true);}
                    break;
                case 1:
                    sDat = sVideoDir + "/" + "video_08_rec" + ".avi"; writer_total.open(sDat, fourcc, fr, size_total, true);
                    break;
                case 2:
                    sDat = sVideoDir + "/" + "video_08_rec" + ".avi"; writer_total.open(sDat, fourcc, fr, cv::Size(nCvWidth*2, nCvHeight), true);
                    break;
                }
            }
            else {
                switch (nRecordMode) {
                case 0:
                    if (vbFlagWnd[stTID.nTrack]) writer_track.write(cvm_track);
                    if (vbFlagWnd[stTID.nRecogOrg]) writer_rec_org.write(cvm_rec_org);
                    if (vbFlagWnd[stTID.nRecogDs]) writer_rec_ds.write(cvm_rec_ds);
                    break;
                case 1:
                    writer_total.write(cvm_record_rec);
                    break;
                case 2:
                    writer_total.write(cvm_record_ds);

                    //char s_frame[128];
                    //std::string sDat;
                    //sprintf(s_frame, "snap_%06i_10_total", nCntFrame);
                    //sDat = sVideoDir + "/" + s_frame + sImgExt; cv::imwrite(sDat, cvm_record_rec);
                    break;
                }
            }
        }
        else bSwitchRecordVideo = false;



        ////** Record times **/////////////////
        if (vbFlagTask[stTID.nRecTime]) {
            if (vbFlagTask[stTID.nRecognition] || vbFlagTask[stTID.nSegmentation]) {
                if (!bSwitchRecordTime) {
                    bSwitchRecordTime = true;
                    for (int i = 0; i < nTaskNrMax; i++) {CloseWindow(i); if(vnBtnProp[i] < 20 && i != stTID.nRecTime) vbFlagTask[i] = false;}

                    bSwitchRecogNewCyc = true;
                    mnTimeMeas1.resize(0); mnTimeMeas2.resize(0);
                    ResetMemory (nObjsNrLimit, nTrackHistoBin_max, nRecogRtNr, vnProtoIdx, vnProtoMemoryCnt, vnProtoStableCnt, vnProtoDisapCnt, vnProtoPtsCnt, mnProtoPtsIdx, mnProtoCenter, mnProtoRCenter, mnProtoRect, mnProtoCubic, mnProtoClrHist,
                                 vnProtoLength, vnProtoFound, nProtoCnt, nCandCnt, nFoundCnt, nFoundNr, vnRecogRating);
                    ResetTime();
                }
                else {
                    if (nCntFrame_tmp > nTimeMessFrmLimit - 1) {
                        vbFlagTask[stTID.nRecTime] = false;
                        clock_gettime(CLOCK_MONOTONIC_RAW, &t_total_end); nTimeTotal = double(timespecDiff(&t_total_end, &t_total_start)/nTimeRatio);
                    }
                }

                mnTimeMeas3.resize(nCntFrame_tmp+1);
                mnTimeMeas3[nCntFrame_tmp].assign(13, 0);

                mnTimeMeas3[nCntFrame_tmp][0] = nTimePre;
                mnTimeMeas3[nCntFrame_tmp][1] = nTimeDepth;
                mnTimeMeas3[nCntFrame_tmp][2] = nTimeBlur;
                mnTimeMeas3[nCntFrame_tmp][3] = nTimeSegm;
                mnTimeMeas3[nCntFrame_tmp][4] = nTimeTrack;
                mnTimeMeas3[nCntFrame_tmp][5] = nTimeAtt;
                mnTimeMeas3[nCntFrame_tmp][6] = nTimeSift;
                mnTimeMeas3[nCntFrame_tmp][7] = nTimeFlann;
                mnTimeMeas3[nCntFrame_tmp][8] = nTimeRec;
                if (nCntFrame_tmp) mnTimeMeas3[nCntFrame_tmp-1][9] = nTimeTotal_prev;
                if (!vbFlagTask[stTID.nRecTime]) mnTimeMeas3[nCntFrame_tmp][9] = nTimeTotal;
                mnTimeMeas3[nCntFrame_tmp][10] = (float)nTmpAttKeyCnt;
                mnTimeMeas3[nCntFrame_tmp][11] = (float)nTmpAttWidth;
                mnTimeMeas3[nCntFrame_tmp][12] = (float)nTmpAttHeight;
            }
        }
        else {
            if (bSwitchRecordTime) {
                bSwitchRecordTime = false;

                mkdir(sDataDir.data(), 0777); MakeNiDirectory (sTimeDirPref, sTimeDir);
                sTimeFile = sTimeDir + "/" + "time.txt";
                sTimeFile_detail = sTimeDir + "/" + "time_detail.xlsx";

                PrintTimes1 (sTimeFile, nCntRecCycle, mnTimeMeas1, mnTimeMeas2);
                PrintTimes2 (sTimeFile, nCntRecCycle, nTimeRecFound_avr, nTimeRecFound_min, nTimeRecFound_max, vnRecogRating, mnTimeMeas2,
                             nCntFrame_tmp, nTimeTotal_avr, nFrameRate_avr, nTimeDepth_avr, nTimeBlur_avr, nTimeSegm_avr, nTimeTrack_avr, nTimeAtt_avr, nTimeRec_avr, nTimeSift_avr, nTimeFlann_avr);
                PrintTimes3 (sTimeFile_detail, mnTimeMeas3);


                bSwitchRecogNewCyc = true;
                mnTimeMeas1.resize(0); mnTimeMeas2.resize(0);
                ResetMemory (nObjsNrLimit, nTrackHistoBin_max, nRecogRtNr, vnProtoIdx, vnProtoMemoryCnt, vnProtoStableCnt, vnProtoDisapCnt, vnProtoPtsCnt, mnProtoPtsIdx, mnProtoCenter, mnProtoRCenter, mnProtoRect, mnProtoCubic, mnProtoClrHist,
                             vnProtoLength, vnProtoFound, nProtoCnt, nCandCnt, nFoundCnt, nFoundNr, vnRecogRating);

                ResetTime();
            }
        }


        ////** Reset times **/////////////////
        if (vbFlagTask[stTID.nRstTime]) ResetTime();


        ////** Parameter setting **/////////////////
        if (vbFlagTask[stTID.nPrmSegm]) {
            if(!vbFlagWnd[stTID.nPrmSegm]) {
                vbFlagWnd[stTID.nPrmSegm] = true;
                cv::namedWindow(vsWndName[stTID.nPrmSegm]);
                cvMoveWindow(vsWndName[stTID.nPrmSegm].data(), 800, 100);

                int dp = int(nTrackDPos*100), ds = int(nTrackDSize*100), dc = int(nTrackDClr*100), dt = int(nTrackDist*100);
                int fp = int(nTrackPFac*100), fs = int(nTrackSFac*100), fc = int(nTrackCFac*100);

                cvCreateTrackbar(vsTrackbarName[20].data(), vsWndName[stTID.nPrmSegm].data(), &nTrackMode, 1, TrackbarHandler_none);
                cvCreateTrackbar(vsTrackbarName[21].data(), vsWndName[stTID.nPrmSegm].data(), &nTrackClrMode, 7, TrackbarHandler_ColorMode);

                cvCreateTrackbar(vsTrackbarName[22].data(), vsWndName[stTID.nPrmSegm].data(), &dp, 100, TrackbarHandler_DistPos);
                cvCreateTrackbar(vsTrackbarName[23].data(), vsWndName[stTID.nPrmSegm].data(), &ds, 100, TrackbarHandler_DistSize);
                cvCreateTrackbar(vsTrackbarName[24].data(), vsWndName[stTID.nPrmSegm].data(), &dc, 100, TrackbarHandler_DistClr);
                cvCreateTrackbar(vsTrackbarName[25].data(), vsWndName[stTID.nPrmSegm].data(), &fp, 100, TrackbarHandler_FacPos);
                cvCreateTrackbar(vsTrackbarName[26].data(), vsWndName[stTID.nPrmSegm].data(), &fs, 100, TrackbarHandler_FacSize);
                cvCreateTrackbar(vsTrackbarName[27].data(), vsWndName[stTID.nPrmSegm].data(), &fc, 100, TrackbarHandler_FacClr);
                cvCreateTrackbar(vsTrackbarName[28].data(), vsWndName[stTID.nPrmSegm].data(), &dt, 500, TrackbarHandler_DistTotal);

                //cvCreateTrackbar(vsTrackbarName[38].data(), vsWndName[stTID.nPrmSegm].data(), &nTrackCntStable, 5, TrackbarHandler_none);
                //cvCreateTrackbar(vsTrackbarName[39].data(), vsWndName[stTID.nPrmSegm].data(), &nTrackCntDisap, 2, TrackbarHandler_none);
                cvCreateTrackbar(vsTrackbarName[31].data(), vsWndName[stTID.nPrmSegm].data(), &nProtoSizeMax, 1000, TrackbarHandler_ProMax);
                cvCreateTrackbar(vsTrackbarName[32].data(), vsWndName[stTID.nPrmSegm].data(), &nProtoSizeMin, 1000, TrackbarHandler_ProMin);
                cvCreateTrackbar(vsTrackbarName[33].data(), vsWndName[stTID.nPrmSegm].data(), &nProtoPtsMin, 1000, TrackbarHandler_none);
                //cvCreateTrackbar(vsTrackbarName[34].data(), vsWndName[stTID.nPrmSegm].data(), &nProtoAspect1, 100, TrackbarHandler_none);
                //cvCreateTrackbar(vsTrackbarName[35].data(), vsWndName[stTID.nPrmSegm].data(), &nProtoAspect2, 100, TrackbarHandler_none);
            }
        }

        if (vbFlagTask[stTID.nPrmRecog]) {
            if(!vbFlagWnd[stTID.nPrmRecog]) {
                vbFlagWnd[stTID.nPrmRecog] = true;
                cv::namedWindow(vsWndName[stTID.nPrmRecog]);
                cvMoveWindow(vsWndName[stTID.nPrmRecog].data(), 850, 100);

                //int dlimit = int(nDLimit*10);
                int recdc = int(nRecogDClr*100);
                int sift_sigma = int(nSiftInitSigma*10);
                int sift_peak = int(nSiftPeakThrs*1000);
                int flannmf = int(nFlannMatchFac*100);

                cvCreateTrackbar(vsTrackbarName[1].data(), vsWndName[stTID.nPrmRecog].data(), &nSnapFormat, 2, TrackbarHandler_none);
                //cvCreateTrackbar(vsTrackbarName[2].data(), vsWndName[stTID.nPrmRecog].data(), &dlimit, 100, TrackbarHandler_ZLimit);
                //cvCreateTrackbar(vsTrackbarName[3].data(), vsWndName[stTID.nPrmRecog].data(), &nDGradFilterSize, 27, TrackbarHandler_ZGradFilterSize);

                //cvCreateTrackbar(vsTrackbarName[11].data(), vsWndName[stTID.nPrmRecog].data(), &nDSegmSizeThres, 1000, TrackbarHandler_none);
                //cvCreateTrackbar(vsTrackbarName[12].data(), vsWndName[stTID.nPrmRecog].data(), &mp, 10000, TrackbarHandler_MergePro);

                cvCreateTrackbar(vsTrackbarName[99].data(), vsWndName[stTID.nPrmRecog].data(), &nAttTDMode, 2, TrackbarHandler_none);
                cvCreateTrackbar(vsTrackbarName[36].data(), vsWndName[stTID.nPrmRecog].data(), &recdc, 120, TrackbarHandler_ColorThres);
                cvCreateTrackbar(vsTrackbarName[41].data(), vsWndName[stTID.nPrmRecog].data(), &nSiftScales, 5, TrackbarHandler_SiftScales);
                cvCreateTrackbar(vsTrackbarName[42].data(), vsWndName[stTID.nPrmRecog].data(), &sift_sigma, 20, TrackbarHandler_SiftSigma);
                cvCreateTrackbar(vsTrackbarName[43].data(), vsWndName[stTID.nPrmRecog].data(), &sift_peak, 100, TrackbarHandler_SiftPeak);
                cvCreateTrackbar(vsTrackbarName[51].data(), vsWndName[stTID.nPrmRecog].data(), &nFlannKnn, 5, TrackbarHandler_FlannKnn);
                cvCreateTrackbar(vsTrackbarName[52].data(), vsWndName[stTID.nPrmRecog].data(), &flannmf, 100, TrackbarHandler_FlannMFac);
                cvCreateTrackbar(vsTrackbarName[53].data(), vsWndName[stTID.nPrmRecog].data(), &nFlannMatchCnt, 100, TrackbarHandler_none);
            }
        }

        ////** Reset Parameters **////////////////
        if (vbFlagTask[stTID.nRstPrm]) ResetParameter();




        ///////////////////////////////////////////////////
        if(vbFlagWnd[stTID.nRgbOrg]) cv::imshow(vsWndName[stTID.nRgbOrg], cvm_rgb_org);
        if(vbFlagWnd[stTID.nRgbDs]) cv::imshow(vsWndName[stTID.nRgbDs], cvm_rgb_ds);
        if(vbFlagWnd[stTID.nDepth]) {AttachImgs(cvm_depth, cvm_dgrady, cvm_dgrady_blur, cvm_dgrady_smth, nCvWidth, nCvHeight, cvm_depth_process); cv::imshow(vsWndName[stTID.nDepth], cvm_depth_process);}
        if(vbFlagWnd[stTID.nDSegm]) cv::imshow(vsWndName[stTID.nDSegm], cvm_segment);
        if(vbFlagWnd[stTID.nTrack]) cv::imshow(vsWndName[stTID.nTrack], cvm_track);
        if(vbFlagWnd[stTID.nProto]) cv::imshow(vsWndName[stTID.nProto], cvm_att);
        if(vbFlagWnd[stTID.nRecogOrg]) {if (!nRecordMode) cv::imshow(vsWndName[stTID.nRecogOrg], cvm_rec_org); else cv::imshow(vsWndName[stTID.nRecogOrg], cvm_record_rec);}
        if(vbFlagWnd[stTID.nRecogDs]) cv::imshow(vsWndName[stTID.nRecogDs], cvm_rec_ds);
        if(vbFlagWnd[stTID.nPrmSegm]) cv::imshow(vsWndName[stTID.nPrmSegm], cvm_set_seg);
        if(vbFlagWnd[stTID.nPrmRecog]) cv::imshow(vsWndName[stTID.nPrmRecog], cvm_set_rec);
        if(vbFlagWnd[stTID.nGSegm]) cv::imshow(vsWndName[stTID.nGSegm], cvm_gbsegm);
        if(vbFlagWnd[stTID.nSIFT]) cv::imshow(vsWndName[stTID.nSIFT], cvm_sift);


        DrawPad(cvm_main, vnBtnProp, nPadOver, nPadCol1, nPadCol2);
        if(vbFlagTask[stTID.nSnap]) vbFlagTask[stTID.nSnap] = false;
        if(vbFlagTask[stTID.nRstTime]) vbFlagTask[stTID.nRstTime] = false;
        if(vbFlagTask[stTID.nRstPrm]) vbFlagTask[stTID.nRstPrm] = false;

        clock_gettime(CLOCK_MONOTONIC_RAW, &t_total_end); nTimeTotal = double(timespecDiff(&t_total_end, &t_total_start)/nTimeRatio); nTimeTotal_prev = nTimeTotal;
        nTimeCurr = double(timespecDiff(&t_total_end, &t_total_origin)/nTimeRatio);
        nTimeFrame = nTimeCurr - nTimePrev; nTimePrev = nTimeCurr;

        nTimePre_acc += nTimePre; nTimePre_avr = nTimePre_acc/nCntFrame;
        if (bTimeDepth) {nCntDepth++; nTimeDepth_acc += nTimeDepth; nTimeDepth_avr = nTimeDepth_acc/nCntDepth;}
        if (bTimeBlur) {nCntBlur++; nTimeBlur_acc += nTimeBlur; nTimeBlur_avr = nTimeBlur_acc/nCntBlur;}
        if (bTimeSegm) {nCntSegm++; nTimeSegm_acc += nTimeSegm; nTimeSegm_avr = nTimeSegm_acc/nCntSegm;}
        if (bTimeTrack) {nCntTrack++; nTimeTrack_acc += nTimeTrack; nTimeTrack_avr = nTimeTrack_acc/nCntTrack;}
        if (bTimeRec) {nCntRec++; nTimeRec_acc += nTimeRec; nTimeRec_avr = nTimeRec_acc/nCntRec; nTimeAtt_acc += nTimeAtt; nTimeAtt_avr = nTimeAtt_acc/nCntRec;}
        if (bTimeSift) {nCntSift++; nTimeSift_acc += nTimeSift; nTimeSift_avr = nTimeSift_acc/nCntSift;}
        if (bTimeFlann) {nCntFlann++; nTimeFlann_acc += nTimeFlann; nTimeFlann_avr = nTimeFlann_acc/nCntFlann;}
        if (bTimeGbSegm) {nCntGbSegm++; nTimeGbSegm_acc += nTimeGbSegm; nTimeGbSegm_avr = nTimeGbSegm_acc/nCntGbSegm;}
        if (bTimeSiftWhole) {nCntSiftWhole++; nTimeSiftWhole_acc += nTimeSiftWhole; nTimeSiftWhole_avr = nTimeSiftWhole_acc/nCntSiftWhole;
            nTimeFlannWhole_acc += nTimeFlannWhole; nTimeFlannWhole_avr = nTimeFlannWhole_acc/nCntSiftWhole;}
        if (vbFlagTask[stTID.nRecognition]) {if (bTimeSift) {nCntFrame_tmp++;
                nTimeTotal_acc += nTimeTotal; nTimeTotal_avr = nTimeTotal_acc/nCntFrame_tmp; nFrameRate_avr = 1000/nTimeTotal_avr;
                nTimeFrame_acc += nTimeFrame; nTimeFrame_avr = nTimeFrame_acc/nCntFrame_tmp;}}
        else {nCntFrame_tmp++; nTimeTotal_acc += nTimeTotal; nTimeTotal_avr = nTimeTotal_acc/nCntFrame_tmp; nFrameRate_avr = 1000/nTimeTotal_avr;
            nTimeFrame_acc += nTimeFrame; nTimeFrame_avr = nTimeFrame_acc/nCntFrame_tmp;}


        char sTotal[128];
        sprintf(sTotal, "TOTAL:"); cv::putText(cvm_main, sTotal, cv::Point(nPadCol1+10, nPadOver-40), nFont, nFontSize, c_magenta, 1);
        sprintf(sTotal, "%8.2f", nTimeTotal_avr); cv::putText(cvm_main, sTotal, cv::Point(nPadCol1+150, nPadOver-40), nFont, nFontSize, c_magenta, 1);
        sprintf(sTotal, "%8.2f Hz", nFrameRate_avr); cv::putText(cvm_main, sTotal, cv::Point(nPadCol1+150, nPadOver-30), nFont, nFontSize, c_magenta, 1);
        sprintf(sTotal, "Frame period:"); cv::putText(cvm_main, sTotal, cv::Point(nPadCol1+10, nPadOver-15), nFont, nFontSize, c_magenta, 1);
        sprintf(sTotal, "%8.2f", nTimeFrame_avr); cv::putText(cvm_main, sTotal, cv::Point(nPadCol1+150, nPadOver-15), nFont, nFontSize, c_magenta, 1);
        if (nTimeFrame_avr) nFrameRate_avr = 1000/nTimeFrame_avr; else nFrameRate_avr = 0;
        sprintf(sTotal, "Frame rate:"); cv::putText(cvm_main, sTotal, cv::Point(nPadCol1+10, nPadOver-5), nFont, nFontSize, c_magenta, 1);
        sprintf(sTotal, "%8.2f Hz", nFrameRate_avr); cv::putText(cvm_main, sTotal, cv::Point(nPadCol1+150, nPadOver-5), nFont, nFontSize, c_magenta, 1);

        if(vbFlagTask[stTID.nInfo]) {
            DrawInfo (cvm_main, nSegNr, nPadOver, nPadCol1, nCntFrame_tmp, nTimeTotal,
                      nTimePre, nTimeDepth, nTimeBlur, nTimePre_avr, nTimeDepth_avr, nTimeBlur_avr,
                      nTimeSegm, nTimeTrack, nTimeAtt, nTimeRec, nTimeSift, nTimeFlann,
                      nTimeSegm_avr, nTimeTrack_avr, nTimeAtt_avr, nTimeRec_avr, nTimeSift_avr, nTimeFlann_avr,
                      nTimeRecCycle, nFoundNr, nTimeRecFound, nTimeRecCycle_avr, nTimeRecFound_avr,
                      nTimeGbSegm, nTimeSiftWhole, nTimeSiftDrawWhole, nTimeFlannWhole, nTimeGbSegm_avr, nTimeSiftWhole_avr, nTimeFlannWhole_avr);
        }
        if(vbFlagTask[stTID.nPrmInfo]) {
            DrawSettings(cvm_main, nPadCol2, nSnapFormat, nDLimit, nDGradFilterSize,
                         nTrackMode, nTrackClrMode, nTrackDPos, nTrackDSize, nTrackDClr, nTrackPFac, nTrackSFac, nTrackCFac, nTrackDist,
                         nProtoSizeMax, nProtoSizeMin, nProtoPtsMin, nProtoAspect1, nProtoAspect2, nAttTDMode,
                         nRecogDClr, nSiftScales, nSiftInitSigma, nSiftPeakThrs, nFlannKnn, nFlannMatchFac, nFlannMatchCnt,
                         nGSegmSigma, nGSegmGrThrs, nGSegmMinSize);
        }
        cv::imshow(sTitle, cvm_main);

        //nCvCurrentkey = getchar(); nCvCurrentkey = cv::waitKey(1);
        nCvCurrentkey = cv::waitKey(1);
        nCntFrame++;
    }

    cvm_main.release();
    cvm_set_seg.release();
    cvm_set_rec.release();
    cvm_rgb_org.release();
    cvm_rgb_ds.release();
    cvm_depth.release();
    cvm_depth_smooth.release();
    cvm_dgradx.release();
    cvm_dgrady.release();
    cvm_segment.release();
    cvm_segment_tmp.release();
    cvm_track.release();
    cvm_att.release();
    cvm_gbsegm.release();

    cvm_rec_org.release();
    cvm_rec_ds.release();
    cvm_sift.release();

    cvm_cand.release();
    cvm_record_rec.release();
    cvm_record_ds.release();

    cvm_dgrady_blur.release();
    cvm_dgrady_smth.release();
    cvm_depth_process.release();

    //ros::shutdown();
    //ros::ok();
    //ros::spinOnce();
}







class NiVisionNode
{
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
    image_transport::Publisher image_pub_;

    ros::Subscriber cloud_sub;

private:
    bool bFlagWait;
    int width, height, width_tmp, height_tmp;

public:
    NiVisionNode(ros::NodeHandle &n) :
        nh_(n), it_(nh_)
    {
        bFlagWait = true;
        width = 100; height = 100, width_tmp = 100, height_tmp = 100;

        cloud_sub = nh_.subscribe ("input_pc", 30, &NiVisionNode::cloudCb, this);
        image_sub_ = it_.subscribe("input_img", 1, &NiVisionNode::imageCb, this);

        //pub = nh.advertise<sensor_msgs::PointCloud2> ("output", 1);

        //image_pub_ = it_.advertise("out", 1);
    }

    ~NiVisionNode() {
    }

    void cloudCb (const sensor_msgs::PointCloud2ConstPtr& cloud){
        m.lock ();
        //sensor_msgs::PointCloud2 output;
        //pub.publish (output);

        cloud_ = cloud;
        m.unlock ();
    }

    void imageCb(const sensor_msgs::ImageConstPtr& msg) {
        cv_bridge::CvImagePtr cv_ptr;
        try {
            if (bFlagWait && cloud_){
                bFlagWait = false;
                printf("O.K.\n\n\n");
                //cvm_image_camera.create(cv::Size(width, height), CV_8UC3);
                nCvWidth = cloud_->width;
                nCvHeight = cloud_->height;
                nCvSize = nCvWidth * nCvHeight;
                boost::thread visualization_thread (&updateImage);
            }
            cv_ptr = cv_bridge::toCvCopy(msg, enc::BGR8);
            cv_ptr->image.copyTo(cvm_image_camera);

            m.unlock ();
        }
        catch (cv_bridge::Exception& e) {ROS_ERROR("cv_bridge exception: %s", e.what()); return;}

        //image_pub_.publish(cv_ptr->toImageMsg());
    }
};

int main(int argc, char** argv)
{
    //////////// Set parameters as default ///////////////////////////////
    InitParameter(argc, argv);
    //sObjNameTmp.append(sSIFTLibFileName, 0, strlen(sSIFTLibFileName.data())-5);
    //sObjNameTmp.append(sSIFTLibFileName, 10, 7);

//    unsigned found = sSIFTLibFileName.find_first_of("_");
//    int fcnt = 0;
//    while (found!=std::string::npos) {
//        printf("%d %d %d\n", fcnt++, found);
//      sObjNameTmp[found]='*';
//      found = sSIFTLibFileName.find_first_of("aeiou",found+1);
//    }
    //std::cout << sSIFTLibFileName << '\n';
    //std::cout << sObjNameTmp << '\n';

    //if (int(strlen(sSIFTLibFileName.data())) <= 10) sObjNameTmp.append(sSIFTLibFileName, 9, strlen(sSIFTLibFileName.data())-4-9-1);
    //else sObjNameTmp.append(sSIFTLibFileName, 15, strlen(sSIFTLibFileName.data())-4-15-1);
    //sTarget.append(sSIFTLibFileName, 0, strlen(sSIFTLibFileName.data())-5);

    sSIFTLibFileName = sLibFilePath + sSIFTLibFileName;
    sColorLibFileName = sLibFilePath + sColorLibFileName;

    switch (nRecogFeature) {
    case 20:
        BuildFlannIndex(1, sColorLibFileName); printf("Color FLANN Index Computed.\n\n");
        BuildFlannIndex(2, sSIFTLibFileName); printf("SIFT FLANN Index Computed.\n\n");
        break;
    }

    ros::init(argc, argv, "ni_vision");
    ros::NodeHandle nh("~");
    NiVisionNode ic(nh);
    ros::spin();
    return 0;
}
