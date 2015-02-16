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
#include <pcl/common/io.h>      //for getFieldIndex()


// Fast SIFT and FLANN Library
#include "siftfast/siftfast.h"
#include "flann/flann.h"


// Terminal tools
#include "terminal_tools/parse.h"



// -----------------------------------------
// -----Sub functions-----------------------
// -----------------------------------------
#include "func_header.hpp"
#include "func_operations.hpp"

#include "func_init.hpp"
#include "func_preproc.hpp"
#include "func_segmentation.hpp"
#include "func_segmentation_gb.hpp"
#include "func_recognition.hpp"
#include "func_etc.hpp"




//////// Dimension of downsampled image /////////////
int nDsWidth;
int nDsHeight;
int nDsSize;


// Global data
sensor_msgs::PointCloud2ConstPtr cloud_, cloud_old_;
boost::mutex m;

pcl::PointCloud<pcl::PointXYZ> cloud_xyz;
pcl::PointCloud<pcl::PointXYZRGB> cloud_xyz_rgb;

cv::Mat cvm_image_camera;

int nCvCurrentkey = -1;

///// temp ////
int nCropX = 1, nCropY = 17, nCropW = 292, nCropH = 220;

bool bRtGbSegm = false, bRtSiftW = false;
int nMeasCounter = 0;







/* Updates the gui and every window that is opend in the gui.
 */
void updateImage() {

    ros::Duration d (0.001);

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////                                                                                        /////////////////////////////////
    ////////////////                    Declartion and initialization of variables                          /////////////////////////////////
    ////////////////                                                                                        /////////////////////////////////
    ////////////////----------------------------------------------------------------------------------------/////////////////////////////////

    bool bPointRgb;                                         // Flag whether point-cloud is in colored or monochrome

    InitVariables();
    cv::setMouseCallback(sTitle, MouseHandler, NULL);

    std::vector<int> vnBtnProp(nTaskNrMax, 0);              // Setting button properties
    vnBtnProp[stTID.nSegmentation] = 20; vnBtnProp[stTID.nRecognition] = 20;

    int nPadOver, nPadCol1, nPadCol2, nPadHeight;           // Setting Gui-control pad properties
    SetPad(nBtnSize, mnBtnPos, nPadOver, nPadCol1, nPadCol2, nPadHeight);

    int nBorderThickness = 2, nTabThickness = 20;           // Setting window and image sizes and properties
    cv::Size size_org = cvm_image_camera.size();
    cv::Size size_ds = cv::Size(nDsWidth, nDsHeight);
    cv::Size size_total = cv::Size(size_org.width + size_ds.width + 3*nBorderThickness, size_org.height + 2*nBorderThickness);
    int nImgScale = size_org.width / size_ds.width;

    int nTabWidth = nPadCol1 + 3*nBtnSize -13 + 360;
    cv::Mat cvm_main(nPadHeight, nTabWidth, CV_8UC3);       // Image buffer for gui-pad

    cv::Mat cvm_rgb_org(size_org, CV_8UC3);                 // Image buffer for RGB image, original
    cv::Mat cvm_rgb_ds(size_ds, CV_8UC3);                   // Image buffer for RGB image, down-sampled
    cv::Mat cvm_depth(size_ds, CV_8UC3);                    // Image buffer for depth map
    cv::Mat cvm_segment(size_ds, CV_8UC3);                  // Image buffer for Segmentation, segment map
    cv::Mat cvm_segment_raw(size_ds, CV_8UC3);              // Image buffer for Segmentation, segment map before post-processing
    cv::Mat cvm_track(size_ds, CV_8UC3);                    // Image buffer for Tracking
    cv::Mat cvm_att(size_org, CV_8UC3);                     // Image buffer for Attention
    cv::Mat cvm_rec_org(size_org, CV_8UC3);                 // Image buffer for Recognition
    cv::Mat cvm_rec_ds(size_ds, CV_8UC3);                   // Image buffer for Recognition, down-sampled

    cv::Mat cvm_dgradx(size_ds, CV_8UC3);                   // Image buffer for depth-gradient map, horisontal
    cv::Mat cvm_dgrady(size_ds, CV_8UC3);                   // Image buffer for depth-gradient map, vertical
    cv::Mat cvm_dgrady_blur(size_ds, CV_8UC3);              // Image buffer for depth-gradient map, vertical, blurring
    cv::Mat cvm_dgrady_smth(size_ds, CV_8UC3);              // Image buffer for depth-gradient map, vertical, smoothing, finally processed
    cv::Mat cvm_depth_process(nDsHeight*2, nDsWidth*2, CV_8UC3);    // Image buffer for presentation of the depth processing

    cv::Mat cvm_cand(size_org, CV_8UC3);                    // Image buffer for actual candidate object
    cv::Mat cvm_record_rec(size_total, CV_8UC3);            // Image buffer for recording
    cv::Mat cvm_record_ds(nDsHeight, nDsWidth*2 , CV_8UC3); // Image buffer for recording, compact

    cv::Mat cvm_gbsegm(size_ds, CV_8UC3);                   // Image buffer for graph-based segmentation
    cv::Mat cvm_sift(size_org, CV_8UC3);                    // Image buffer for SIFT on the whole image

    cv::Mat cvm_set_seg(50, 400, CV_8UC3);                  // Image buffer for settings, for Segmentation
    cv::Mat cvm_set_rec(50, 400, CV_8UC3);                  // Image buffer for settings, for Recognition


    cv::VideoWriter writer_rgb_ds, writer_track, writer_rec_org, writer_rec_ds, writer_total;   // Video recording buffers

    ///////// Color tabs /////////////
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


    float nDGradNan = 1000;


    ////////////** for Tracking **////////////////////////////////////////////////////////////
    int nObjsNrLimit = 1000;
    int nTrackHistoBin_max = stTrack.HistoBin * stTrack.HistoBin * stTrack.HistoBin;

    SurfProp stMems;
    stMems.vnIdx.resize(nObjsNrLimit, 0);
    stMems.vnPtsCnt.resize(nObjsNrLimit, 0);
    stMems.mnPtsIdx.assign(nObjsNrLimit, std::vector<int>(0, 0));
    stMems.mnRect.assign(nObjsNrLimit, std::vector<int>(4, 0));
    stMems.mnRCenter.assign(nObjsNrLimit, std::vector<int>(2, 0));
    stMems.mnCubic.assign(nObjsNrLimit, std::vector<float>(6, 0));
    stMems.mnCCenter.assign(nObjsNrLimit, std::vector<float>(3, 0));
    stMems.mnColorHist.assign(nObjsNrLimit, std::vector<float>(nTrackHistoBin_max, 0));
    stMems.vnLength.resize(nObjsNrLimit, 0);
    stMems.vnMemCtr.resize(nObjsNrLimit, 0); stMems.vnStableCtr.resize(nObjsNrLimit, 0); stMems.vnLostCtr.resize(nObjsNrLimit, 0);
    stMems.vnFound.resize(nObjsNrLimit, 0);

    std::vector<int> vnMemsValidIdx;
    std::vector<std::vector<float> > mnMemsRelPose;

    //////////////////////////////////////////////////////////////////////////////////////////


    ////////////** for Recognition **/////////////////////////////////////////////////////////
    int nMemsCnt = 0;
    int nFoundCnt = 0, nFoundNr = 0, nFoundFrame = 0;
    int nRecogRtNr = 10; std::vector<int> vnRecogRating(nRecogRtNr, 0), vnRecogRating_tmp(nRecogRtNr, 0);


    std::string sSnapDir; std::string sSnapDirPref = sDataDir + "/" + "NI_Snapshots_";
    std::string sVideoDir; std::string sVideoDirPref = sDataDir + "/" + "NI_Videos_";
    std::string sTimeDir; std::string sTimeDirPref = sDataDir + "/" + "NI_Time Measuring_";
    std::string sTimeFile, sTimeFile_detail;
    std::string sImgExt;

    bool bSwitchSegm = false;
    bool bSwitchRecog = false;
    bool bSwitchRecogNewCyc = true;
    bool bSwitchRecordTime = false;
    bool bSwitchRecordVideo = false;

    std::string sTitleFile = sTargetImgPath + sTarget + "." + "jpg";
    cv::Mat cvm_target_org = cv::imread(sTitleFile.data(), CV_LOAD_IMAGE_COLOR);
    cv::Mat cvm_target;
    if(cvm_target_org.data) {
        double sx = (double)nDsWidth/cvm_target_org.cols;
        double sy = (double)nDsHeight/cvm_target_org.rows;
        cv::resize (cvm_target_org, cvm_target, cv::Size(), min(sx,sy), min(sx,sy), cv::INTER_AREA);
    }


    int nTimeRatio = 1000000;
    int nCandCnt = 0;
    nTimeRecCycle = 0;

    struct timespec t_reccyc_start, t_reccyc_end;
    struct timespec t_rec_found_start, t_rec_found_end;


    double nTimeTotal_prev = 0;
    double nTimeCurr = 0, nTimePrev = 0;
    struct timespec t_total_origin;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t_total_origin);
    int nTmpAttKeyCnt = 0, nTmpAttWidth = 0, nTmpAttHeight = 0;

    //////////////--------------------------------------------------------------------------------------/////////////////////////////////////
    //////////////                                                                                      /////////////////////////////////////
    //////////////                End of Declaration and Initialization of variables                     /////////////////////////////////////
    //////////////                                                                                      /////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



    bool bSwitchRecordRec = false;

    while (true) {

        d.sleep ();
        if(bFlagEnd) std::terminate();   // finish the program

        if (!cloud_) continue;
        //if (cloud_old_ == cloud_) continue;

        m.lock ();
        if (pcl::getFieldIndex (*cloud_, "rgb") != -1) {bPointRgb = true; pcl::fromROSMsg (*cloud_, cloud_xyz_rgb);}
        else {bPointRgb = false; pcl::fromROSMsg (*cloud_, cloud_xyz);}
        cloud_old_ = cloud_;
        m.unlock ();


        struct timespec t_total_start, t_total_end; clock_gettime(CLOCK_MONOTONIC_RAW, &t_total_start);


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////                                                                                      /////////////////////////
//////////////////////////                            Pre-Setting for the loop                                  /////////////////////////
//////////////////////////                                                                                      /////////////////////////
//////////////////////////--------------------------------------------------------------------------------------/////////////////////////
        struct timespec t_pre_start, t_pre_end; clock_gettime(CLOCK_MONOTONIC_RAW, &t_pre_start);


        nTimePre = 0; nTimeDepth = 0; nTimeBlur = 0; nTimeTotal = 0;
        nTimeSegm = 0; nTimeTrack = 0; nTimeAtt = 0; nTimeRec = 0; nTimeSift = 0; nTimeFlann = 0;
        nTimeGbSegm = 0; nTimeSiftWhole = 0; nTimeSiftDrawWhole = 0; nTimeFlannWhole = 0;
        bTimeDepth = false; bTimeBlur = false;
        bTimeSegm = false; bTimeTrack = false; bTimeRec = false; bTimeSift = false; bTimeFlann = false;
        bTimeGbSegm = false; bTimeSiftWhole = false;

        cvm_main = cv::Scalar(0, 0, 0);
        cvm_set_seg = cv::Scalar(0, 0, 0); cvm_set_rec = cv::Scalar(0, 0, 0);
        cvm_rgb_org = cv::Scalar(0, 0, 0); cvm_rgb_ds = cv::Scalar(0, 0, 0);
        cvm_depth = cv::Scalar(0, 0, 0); cvm_dgradx = cv::Scalar(0, 0, 0); cvm_dgrady = cv::Scalar(0, 0, 0);
        cvm_segment = cv::Scalar(0, 0, 0); cvm_segment_raw = cv::Scalar(0, 0, 0);
        cvm_track = cv::Scalar(0, 0, 0);
        cvm_att = cv::Scalar(0, 0, 0); cvm_cand = cv::Scalar(0, 0, 0);
        cvm_rec_org = cv::Scalar(0, 0, 0); cvm_rec_ds = cv::Scalar(0, 0, 0);

        cvm_record_rec = cv::Scalar(0, 0, 0); cvm_record_ds = cv::Scalar(0, 0, 0);
        cvm_dgrady_blur = cv::Scalar(0, 0, 0); cvm_dgrady_smth = cv::Scalar(0, 0, 0); cvm_depth_process = cv::Scalar(0, 0, 0);

        cvm_gbsegm = cv::Scalar(0, 0, 0); cvm_sift = cv::Scalar(0, 0, 0);


        for (int i = 0; i < nTaskNrMax; i++) if (!vbFlagTask[i]) CloseWindow(i);


        // The value of nTrackHistoBin_tmp depends on the number of channels of the selected color model
        int nTrackHistoBin_tmp;
        switch(stTrack.ClrMode) {
        case 0: case 1: case 2: nTrackHistoBin_tmp = nTrackHistoBin_max; break;
        case 3: nTrackHistoBin_tmp = stTrack.HistoBin * stTrack.HistoBin; break;
        case 4: case 5: case 6: nTrackHistoBin_tmp = nTrackHistoBin_max; break;
        case 7: nTrackHistoBin_tmp = stTrack.HistoBin * stTrack.HistoBin; break;
        }


        ///// Setting snapshot format ////////////////
        switch (nSnapFormat) {case 0: sImgExt = ".bmp"; break; case 1: sImgExt = ".tif"; break; case 2: sImgExt = ".jpg"; break;}




        //////*  copy point cloud  *///////////
        pcl::PointCloud<pcl::PointXYZ> cloud_Input;
        pcl::PointCloud<pcl::PointXYZRGB> cloud_Input_rgb;

        if (bPointRgb) pcl::copyPointCloud (cloud_xyz_rgb, cloud_Input_rgb);
        else pcl::copyPointCloud (cloud_xyz, cloud_Input);


        //////*  Copying original RGB image and generating downsampled image  *//////
        cvm_image_camera.copyTo(cvm_rgb_org);
        for(size_t i = 0; i < nDsSize; i++) {
            uint8_t r_, g_, b_;
            unpack_rgb(cloud_Input_rgb.points[i].rgb, r_, g_, b_);
            cvm_rgb_ds.data[i*3] = b_; cvm_rgb_ds.data[i*3 + 1] = g_; cvm_rgb_ds.data[i*3 + 2] = r_;
        }


        //////*  Set an indices for cloud  *///////////////
        std::vector<int> vnCloudIdx(nDsSize);
        for (int i = 0; i < nDsSize; ++i) vnCloudIdx[i] = i;

        //////*  Show RBG image  *//////////////////////
        if (vbFlagTask[stTID.nRgbOrg]) OpenWindow(stTID.nRgbOrg);
        if (vbFlagTask[stTID.nRgbDs]) OpenWindow(stTID.nRgbDs);


        clock_gettime(CLOCK_MONOTONIC_RAW, &t_pre_end); nTimePre = double(timespecDiff(&t_pre_end, &t_pre_start)/nTimeRatio);
        //////////////////------------------------------------------------------------------------------/////////////////////////
        //////////////////                                                                              /////////////////////////
        //////////////////                               End of Pre-Setting                             /////////////////////////
        //////////////////                                                                              /////////////////////////
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //////////////////                  Pre-Processing for the Segmentation                     //////////////////////
        //////////////////                  Generation of depth-gradient map                        //////////////////////
        //////////////////--------------------------------------------------------------------------//////////////////////

        std::vector<float> vnX(nDsSize, 0), vnY(nDsSize, 0), vnZ(nDsSize, 0);   // vnZ: Depth Map
        std::vector<int> vnCloudIdx_d(nDsSize, 0);                              // valid Point Cloud indices

        std::vector<float> vnDGradX(nDsSize, nDGradNan);                        // Depth-Gradient Map, horizontal
        std::vector<float> vnDGradY(nDsSize, nDGradNan);                        // Depth-Gradient Map, vertical
        std::vector<float> vnDGradY_blur(nDsSize, nDGradNan);                   // Depth-Gradient Map, vertical, blured
        std::vector<float> vnDGradY_smth(nDsSize, nDGradNan);                   // Depth-Gradient Map, vertical, smoothen
        std::vector<float> vnDGradY_pre(nDsSize, nDGradNan);                    // Depth-Gradient Map, vertical, finally processed

        float nDRange = 6;
        int nDIdxCntTmp = 0;
        float nDMin = 10, nDMax = 0;
        float nDGradXMin = 99, nDGradXMax = -99;
        float nDGradYMin = 99, nDGradYMax = -99;

        if (vbFlagTask[stTID.nSegmentation] || vbFlagTask[stTID.nDepth]) {
            struct timespec t_depth_start, t_depth_end; clock_gettime(CLOCK_MONOTONIC_RAW, &t_depth_start); bTimeDepth = true;

            //////*  Making depth map: storing coordinates from point cloud  *////////////
            if (bPointRgb) MakeDepthMap (cloud_Input_rgb, nDsSize, nDsWidth, nDMax, nDMin, nDIdxCntTmp, vnCloudIdx_d, vnX, vnY, vnZ);
            else MakeDepthMap (cloud_Input, nDsSize, nDMax, nDMin, nDIdxCntTmp, vnCloudIdx_d, vnX, vnY, vnZ);

            //////*  Making depth-gradient map  *////////////////////////////////////////
            MakeDGradMap (vnZ, vnCloudIdx_d, nDIdxCntTmp, nDGradConst, nSegmDThres, nDGradNan, nDsWidth, nDGradXMin, nDGradXMax, nDGradYMin, nDGradYMax, vnDGradX, vnDGradY);
            clock_gettime(CLOCK_MONOTONIC_RAW, &t_depth_end); nTimeDepth = double(timespecDiff(&t_depth_end, &t_depth_start)/nTimeRatio);


            //////*  Processing depth gradient map  *////////////////////////////////////
            struct timespec t_blur_start, t_blur_end; clock_gettime(CLOCK_MONOTONIC_RAW, &t_blur_start); bTimeBlur = true;

            vnDGradY_blur = vnDGradY; vnDGradY_smth = vnDGradY;
            Segm_SmoothDepthGrad (vnDGradY, vnCloudIdx_d, size_ds, nSegmDThres, -nSegmDThres, nDGradNan, nDGradFilterMode, nDGradFilterSize, nDGradSmthMode, nDGradSmthCenter, nDGradSmthBnd1, nDGradSmthBnd2, nDGradSmthFac, vnDGradY_blur, vnDGradY_smth);
            vnDGradY_pre = vnDGradY_smth;
            for (int i = 0; i < nDIdxCntTmp; i++) {if (vnDGradX[vnCloudIdx_d[i]] > nDGradNan*0.8 || vnDGradY[vnCloudIdx_d[i]] > nDGradNan*0.8) vnDGradY_pre[vnCloudIdx_d[i]] = nDGradNan;}

            //////*  Visualizing depth and its variations  */////////////////////////////
            if (vbFlagTask[stTID.nDepth]) {
                if (vbFlagTask[stTID.nDepth]) {
                    if (!vbFlagWnd[stTID.nDepth]) {
                        vbFlagWnd[stTID.nDepth] = true;
                        cv::namedWindow(vsWndName[stTID.nDepth]);
                        cvMoveWindow(vsWndName[stTID.nDepth].data(), mWndPos[stTID.nDepth][0], mWndPos[stTID.nDepth][1]);
                        int dgc = int(nDGradConst*100);
                        cvCreateTrackbar(vsTrackbarName[81].data(), vsWndName[stTID.nDepth].data(), &nDGradSmthCenter, 255, TrackbarHandler_none);
                        cvCreateTrackbar(vsTrackbarName[82].data(), vsWndName[stTID.nDepth].data(), &nDGradSmthBnd1, 30, TrackbarHandler_none);
                        cvCreateTrackbar(vsTrackbarName[83].data(), vsWndName[stTID.nDepth].data(), &nDGradSmthBnd2, 30, TrackbarHandler_none);
                        //cvCreateTrackbar(vsTrackbarName[84].data(), vsWndName[stTID.nDepth].data(), &dgc, 100, TrackbarHandler_DGradC);
                    }
                }
                if (nDMax > nDRange) DrawDepth(vnZ, vnCloudIdx_d, bDepthDispMode, nDRange, nDMin, cvm_depth);
                else DrawDepth(vnZ, vnCloudIdx_d, bDepthDispMode, nDMax, nDMin, cvm_depth);

                DrawDepthGrad (vnDGradY, vnCloudIdx_d, bDepthDispMode, nSegmDThres, -nSegmDThres, nDGradNan, cvm_dgrady);
                DrawDepthGrad (vnDGradX, vnCloudIdx_d, bDepthDispMode, nSegmDThres, -nSegmDThres, nDGradNan, cvm_dgradx);
                DrawDepthGrad (vnDGradY_blur, vnCloudIdx_d, bDepthDispMode, nSegmDThres, -nSegmDThres, nDGradNan, cvm_dgrady_blur);
                DrawDepthGrad (vnDGradY_smth, vnCloudIdx_d, bDepthDispMode, nSegmDThres, -nSegmDThres, nDGradNan, cvm_dgrady_smth);
            }

            clock_gettime(CLOCK_MONOTONIC_RAW, &t_blur_end); nTimeBlur = double(timespecDiff(&t_blur_end, &t_blur_start)/nTimeRatio);
        }
        //////////////////--------------------------------------------------------------------------//////////////////////
        //////////////////             End of the Pre-Process for the Segmentation                  //////////////////////
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////



        int nSurfCnt = 0;
        if (vbFlagTask[stTID.nSegmentation]){

            /////*  Trigger  *///////////
            if (!bSwitchSegm) {bSwitchSegm = true; nTimeTotal_acc = 0; nTimeTotal_avr = 0; nCtrFrame_tmp = 0; nTimeFrame_acc = 0; nTimeFrame_avr = 0;}

            /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            //////////////////                                                                                  /////////////////////////
            //////////////////                             SEGMENTATION                                         /////////////////////////
            //////////////////                                                                                  /////////////////////////
            //////////////////----------------------------------------------------------------------------------/////////////////////////
            struct timespec t_segm_start, t_segm_end; clock_gettime(CLOCK_MONOTONIC_RAW, &t_segm_start); bTimeSegm = true;

            std::vector<int> vnSegmMap(nDsSize, 0);
            std::vector<int> vnSegmMap_tmp = vnSegmMap;
            Segmentation(vnDGradY_pre, vnCloudIdx_d, nSegmSizeThres, nSegmGradDist, nDGradNan, nDsWidth, nDsSize, 0, nDsWidth, 0, nDsHeight, vnSegmMap_tmp, vnSegmMap, nSurfCnt);
            nSurfCnt++;

            //////*  Making Segments Point Indices  *//////////////////////
            std::vector<int> vnSurfPtsCnt(nSurfCnt, 0);
            std::vector<std::vector<int> > mnSurfPtsIdx(nSurfCnt, std::vector<int>(nDsSize, 0));
            Map2Objects(vnSegmMap, nSurfCnt, nDsSize, mnSurfPtsIdx, vnSurfPtsCnt);

            //////*  Visualizing Segmentation  *///////////////////////////
            if (vbFlagTask[stTID.nSegm]){
                if (vbFlagTask[stTID.nSegm]) OpenWindow(stTID.nSegm);
                for (int i = 0; i < nDsSize; i++) {
                    if (!vnSegmMap[i]) {
                        //AssignColor(i, mnColorTab[0], cvm_dgrady_blur);
                        //cvm_dgrady_blur.data[i*3] = 0; cvm_dgrady_blur.data[i*3+1] = 0; cvm_dgrady_blur.data[i*3+2] = 0;
                        continue;
                    }
                    //AssignColor(i, mnColorTab[vnSegmMap[i]], cvm_segment);
                    cvm_segment.data[i*3] = mnColorTab[vnSegmMap[i]][0]; cvm_segment.data[i*3+1] = mnColorTab[vnSegmMap[i]][1]; cvm_segment.data[i*3+2] = mnColorTab[vnSegmMap[i]][2];
                    //AssignColor(i, mnColorTab[vnSegmMap_tmp[i]], cvm_segment_raw);
                    cvm_segment_raw.data[i*3] = mnColorTab[vnSegmMap_tmp[i]][0]; cvm_segment_raw.data[i*3+1] = mnColorTab[vnSegmMap_tmp[i]][1]; cvm_segment_raw.data[i*3+2] = mnColorTab[vnSegmMap_tmp[i]][2];
                }
            }

            clock_gettime(CLOCK_MONOTONIC_RAW, &t_segm_end); nTimeSegm = double(timespecDiff(&t_segm_end, &t_segm_start)/nTimeRatio);
            //////////////////----------------------------------------------------------------------------------/////////////////////////
            //////////////////                                                                                  /////////////////////////
            //////////////////                          End of the SEGMENTATION                                 /////////////////////////
            //////////////////                                                                                  /////////////////////////
            /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



            /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            //////////////////                                                                                  /////////////////////////
            //////////////////                                 TRACKING                                         /////////////////////////
            //////////////////                                                                                  /////////////////////////
            //////////////////----------------------------------------------------------------------------------/////////////////////////
            struct timespec t_track_start, t_track_end; clock_gettime(CLOCK_MONOTONIC_RAW, &t_track_start); bTimeTrack = true;

            SurfProp stSurf;
            stSurf.vnIdx.resize(nSurfCnt, 0); stSurf.vnPtsCnt = vnSurfPtsCnt; stSurf.mnPtsIdx = mnSurfPtsIdx;
            stSurf.mnRect.assign(nSurfCnt, std::vector<int>(4,0)); stSurf.mnRCenter.assign(nSurfCnt, std::vector<int>(2,0));
            stSurf.mnCubic.assign(nSurfCnt, std::vector<float>(6,0)); stSurf.mnCCenter.assign(nSurfCnt, std::vector<float>(3,0));
            stSurf.vnLength.resize(nSurfCnt, 0);
            stSurf.mnColorHist.resize(nSurfCnt, std::vector<float> (nTrackHistoBin_max, 0));
            stSurf.vnMemCtr.resize(nSurfCnt, 0); stSurf.vnStableCtr.resize(nSurfCnt, 0); stSurf.vnLostCtr.resize(nSurfCnt, 0);

            Tracking_Pre (nSegmCutSize, nDsWidth, nDsHeight, vnX, vnY, vnZ, cvm_rgb_ds, stTrack, stSurf, nSurfCnt);
            Tracking (nSurfCnt, nObjsNrLimit, stTrack, nTrackHistoBin_tmp, stSurf, stMems, nMemsCnt, vnMemsValidIdx, mnMemsRelPose, vbFlagTask[stTID.nRes1], nCtrFrame_tmp);

            //////*  Making tracking map to a neighborhood matrix for surface saliencies  */////////////////////////////
            std::vector<int> vnTrkMap(nDsSize, -1);         // Tracking Map
            //std::vector<int> vnTrkMapComp(nDsSize, -1);     // for the making of the Neighbor Matrix
            for (int i = 0; i < nMemsCnt; i++) {
                if (stMems.vnStableCtr[i] < stTrack.CntStable || stMems.vnLostCtr[i] > stTrack.CntLost) continue;
                for (size_t j = 0; j < stMems.mnPtsIdx[i].size(); j++) {
                    vnTrkMap[stMems.mnPtsIdx[i][j]] = stMems.vnIdx[i];
                    //vnTrkMapComp[stMems.mnPtsIdx[i][j]] = i;
                }
            }


            // Making Neighboring matrix between attention candidates
            //std::vector<std::vector<bool> > mbCandNeighbor(nMemsCnt, std::vector<bool>(nMemsCnt, false));
            //Segm_NeighborMatrix1(vnTrkMapComp, vnCloudIdx_d, 2, nDsWidth, mbCandNeighbor);

            // Show Tracking
            if (vbFlagTask[stTID.nTrack] || ((vbFlagTask[stTID.nRecogOrg] || vbFlagTask[stTID.nRecVideo]) && nRecordMode)) {
                if(vbFlagTask[stTID.nTrack] && !vbFlagWnd[stTID.nTrack]) SetFlagWnd(stTID.nTrack);

                for (int i = 0; i < nDsSize; i++) {
                    if (vnTrkMap[i] < 0) continue;
                    cvm_track.data[i*3] = mnColorTab[vnTrkMap[i]][0]; cvm_track.data[i*3+1] = mnColorTab[vnTrkMap[i]][1]; cvm_track.data[i*3+2] = mnColorTab[vnTrkMap[i]][2];
                }
            }

            ///////////////////////////////////////////////////////////////////
            if (vbFlagTask[stTID.nRes1]) {
                if (!bSwitchRecordRec) {
                    bSwitchRecordRec = true;
                    mkdir(sDataDir.data(), 0777); MakeNiDirectory (sSnapDirPref, sSnapDir);
                }
                else {
                    std::string sDat;
                    char s_frame[128];
                    sprintf(s_frame, "snap_%06i_trk", nCtrFrame_tmp); sDat = sSnapDir + "/" + s_frame + sImgExt; cv::imwrite(sDat, cvm_track);
                    //sprintf(s_frame, "snap_%06i_seg", nCtrFrame_tmp); sDat = sSnapDir + "/" + s_frame + sImgExt; cv::imwrite(sDat, cvm_segment);
                    //sprintf(s_frame, "snap_%06i_rgb", nCtrFrame_tmp); sDat = sSnapDir + "/" + s_frame + sImgExt; cv::imwrite(sDat, cvm_rgb_ds);
                }
            }
            else bSwitchRecordRec = false;

            ///////////////////////////////////////////////////////////////////////////////////


            // Show Object Candidates
            cv::Mat cvm_mask(size_org, CV_8UC3, cv::Scalar(0, 0, 0));
            if (vbFlagTask[stTID.nAtt]) {
                OpenWindow(stTID.nAtt);
                int xx, yy;
                for (int i = 0; i < nMemsCnt; i++) {
                    if (stMems.vnStableCtr[i] < stTrack.CntStable || stMems.vnLostCtr[i] > stTrack.CntLost) continue;
                    if (stMems.vnLength[i]*1000 > nAttSizeMax || stMems.vnLength[i]*1000 < nAttSizeMin || stMems.vnPtsCnt[i] < nAttPtsMin) continue;

                    for (int j = 0; j < stMems.vnPtsCnt[i]; j++) {
                        GetPixelPos(stMems.mnPtsIdx[i][j], nDsWidth, xx, yy);
                        cv::Vec3b s_tmp = cvm_track.at<cv::Vec3b>(yy, xx);

                        xx = xx*nImgScale; yy = yy*nImgScale;
                        if(yy > size_org.height || xx > size_org.width - nImgScale) continue;
                        cv::Vec3b s;

                        for (int mm = 0; mm < nImgScale; mm++) {
                            for (int nn = 0; nn < nImgScale; nn++) {
                                if (xx-nn < 0) continue;
                                if (vbFlagTask[stTID.nTrack]) cvm_mask.at<cv::Vec3b>(yy-mm, xx-nn) = s_tmp;
                                else {cvm_att.at<cv::Vec3b>(yy-mm, xx+nn) = cvm_rgb_org.at<cv::Vec3b>(yy-mm, xx+nn);}
                            }
                        }
                    }
                }

                if(vbFlagTask[stTID.nTrack]) cv::add(cvm_rgb_org, cvm_mask, cvm_att);
            }
            cvm_mask.release();

            clock_gettime(CLOCK_MONOTONIC_RAW, &t_track_end); nTimeTrack = double(timespecDiff(&t_track_end, &t_track_start)/nTimeRatio);
            //////////////////----------------------------------------------------------------------------------/////////////////////////
            //////////////////                                                                                  /////////////////////////
            //////////////////                              End of the TRACKING                                 /////////////////////////
            //////////////////                                                                                  /////////////////////////
            /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



            /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            //////////////////                                                                                  /////////////////////////
            //////////////////                                   ATTENTION                                      /////////////////////////
            //////////////////                                                                                  /////////////////////////
            //////////////////----------------------------------------------------------------------------------/////////////////////////


            //////////** Top-down guidance **/////////////////

            if (vbFlagTask[stTID.nRecognition]) {

                struct timespec t_att_start, t_att_end; clock_gettime(CLOCK_MONOTONIC_RAW, &t_att_start);

                //////// Sorting Objects ////////////////////////
                float tmp_diff = 100;
                std::vector<std::pair<float, int> > veCandClrDist(nMemsCnt);
                std::vector<bool> vbProtoCand(nObjsNrLimit, false);
                // Top-down Selection
                if (!nAttTDMode) {
                    for (int i = 0; i < nMemsCnt; i++) {
                        veCandClrDist[i].second = i;
                        if (stMems.vnStableCtr[i] < stTrack.CntStable || stMems.vnLostCtr[i] > stTrack.CntLost) {veCandClrDist[i].first = tmp_diff++; continue;}
                        if (stMems.vnLength[i]*1000 > nAttSizeMax || stMems.vnLength[i]*1000 < nAttSizeMin || stMems.vnPtsCnt[i] < nAttPtsMin) {veCandClrDist[i].first = tmp_diff++; continue;}

                        vbProtoCand[i] = true;
                        float dc = 0;
                        for (int j = 0; j < nTrackHistoBin_tmp; j++) {
                            if (mnColorHistY_lib.size() == 1) dc += fabs(mnColorHistY_lib[0][j] - stMems.mnColorHist[i][j]);
                            else dc += fabs(mnColorHistY_lib[stTrack.ClrMode][j] - stMems.mnColorHist[i][j]);
                        }
                        veCandClrDist[i].first = dc/2;
                    }
                }
                // Scuffling object surfaces for random order (for comparison)
                else {
                    std::vector<int> number;
                    for (int i = 0; i < nMemsCnt; i++) number.push_back(i+1);
                    std::random_shuffle(number.begin(), number.end());

                    for (int i = 0; i < nMemsCnt; i++) {
                        veCandClrDist[i].second = i;
                        if (stMems.vnStableCtr[i] < stTrack.CntStable && stMems.vnLostCtr[i] > stTrack.CntLost) {veCandClrDist[i].first = tmp_diff++; continue;}
                        if (nAttTDMode == 1)
                            if (stMems.vnLength[i]*1000 > nAttSizeMax || stMems.vnLength[i]*1000 < nAttSizeMin || stMems.vnPtsCnt[i] < nAttPtsMin) {veCandClrDist[i].first = tmp_diff++; continue;}

                        vbProtoCand[i] = true;
                        veCandClrDist[i].first = (float)number[i];
                    }
                }
                Attention_TopDown (vbProtoCand, stMems, nMemsCnt, veCandClrDist);

                clock_gettime(CLOCK_MONOTONIC_RAW, &t_att_end); nTimeAtt = double(timespecDiff(&t_att_end, &t_att_start)/nTimeRatio);


                //////////////////----------------------------------------------------------------------------------/////////////////////////
                //////////////////                                                                                  /////////////////////////
                //////////////////                                End of ATTENTION                                  /////////////////////////
                //////////////////                                                                                  /////////////////////////
                /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


                /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                //////////////////                                                                                  /////////////////////////
                //////////////////                                   RECOGNITION                                    /////////////////////////
                //////////////////                                                                                  /////////////////////////
                //////////////////----------------------------------------------------------------------------------/////////////////////////
                struct timespec t_rec_start, t_rec_end; clock_gettime(CLOCK_MONOTONIC_RAW, &t_rec_start);

                if (!bSwitchRecog) {
                    bSwitchRecog = true;
                    bSwitchRecogNewCyc = true;
                    nTimeTotal_acc = 0; nTimeTotal_avr = 0; nCtrFrame_tmp = 0; nTimeFrame_acc = 0; nTimeFrame_avr = 0;
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


                    for (int i = 0; i < nMemsCnt*2; i++) {
                        if (stMems.vnFound[i] > 2) stMems.vnFound[i] = 1;
                        else stMems.vnFound[i] = 0;
                    }
                    vnRecogRating_tmp.assign(nRecogRtNr, 0);
                }



                bTimeRec = true;

                int nCandID = -1;
                Attention_Selection (nMemsCnt, vbProtoCand, stMems.vnFound, nCandID);       // Selection to attention

                if (vbFlagTask[stTID.nRecogOrg] || (vbFlagTask[stTID.nRecVideo] && nRecordMode)) {
                    if (vbFlagTask[stTID.nRecogOrg]) OpenWindow(stTID.nRecogOrg);
                    cvm_rgb_org.copyTo(cvm_rec_org);
                }
                if (vbFlagTask[stTID.nRecogDs] || ((vbFlagTask[stTID.nRecogOrg] || vbFlagTask[stTID.nRecVideo]) && nRecordMode)) {
                    if (vbFlagTask[stTID.nRecogDs]) OpenWindow(stTID.nRecogDs);
                    cvm_rgb_ds.copyTo(cvm_rec_ds);
                }


                /////// If all candidates are inspected (nCandID = -1), sart a new recognition cycle ////////////////////////
                if (nCandID < 0) {
                    clock_gettime(CLOCK_MONOTONIC_RAW, &t_reccyc_end); nTimeRecCycle = double(timespecDiff(&t_reccyc_end, &t_reccyc_start)/nTimeRatio);


                    bSwitchRecogNewCyc = true;
                    if (nCandCnt) {
                        if (vbFlagTask[stTID.nRecTime] && bSwitchRecordTime) {
                            for (size_t i = 0; i < mnTimeMeas1.size(); i++) {
                                if (mnTimeMeas1[i][0] == nCtrRecCycle+1) mnTimeMeas1[i][3] = nCandCnt;
                            }
                        }
                        nCtrRecCycle++;
                        nTimeRecCycle_acc += nTimeRecCycle; nTimeRecCycle_avr = nTimeRecCycle_acc/nCtrRecCycle;
                        nTimeRecFound_acc += nTimeRecFound; nTimeRecFound_avr = nTimeRecFound_acc/nCtrRecCycle;

                        clock_gettime(CLOCK_MONOTONIC_RAW, &t_reccyc_start);
                        clock_gettime(CLOCK_MONOTONIC_RAW, &t_rec_found_start);
                        bSwitchRecogNewCyc = false;
                        bTimeRecogCycle = false;
                        nFoundCnt = 0;
                        nFoundNr = 0;
                        nCandCnt = 0;
                        nTimeRecFound = 0;
                        nTmpAttKeyCnt = 0; nTmpAttWidth = 0; nTmpAttHeight = 0;

                        for (int i = 0; i < nMemsCnt*2; i++) {
                            if (stMems.vnFound[i] > 2) stMems.vnFound[i] = 1;
                            else stMems.vnFound[i] = 0;
                        }
                        vnRecogRating_tmp.assign(nRecogRtNr, 0);

                        Attention_Selection (nMemsCnt, vbProtoCand, stMems.vnFound, nCandID);       // Selection to attention
                    }

                    if (((vbFlagTask[stTID.nRecogOrg] || vbFlagTask[stTID.nRecVideo]) && nRecordMode)) {
                        switch (nRecordMode) {
                        case 1: {
                            cv::Mat dst;
                            dst = cvm_record_rec (cv::Rect(nDsWidth + 2*nBorderThickness, nBorderThickness, cvm_rec_org.cols, cvm_rec_org.rows)); cvm_rec_org.copyTo(dst);
                            dst = cvm_record_rec (cv::Rect(nBorderThickness, cvm_record_rec.rows - nDsHeight -  nBorderThickness, nDsWidth, nDsHeight)); cvm_rec_ds.copyTo(dst);
                            dst = cvm_record_rec (cv::Rect(nBorderThickness, cvm_record_rec.rows - 2*nDsHeight - nTabThickness - nBorderThickness, nDsWidth, nDsHeight)); cvm_track.copyTo(dst);
                            dst = cvm_record_rec (cv::Rect(nBorderThickness+(nDsWidth-cvm_target.cols)/2, nBorderThickness+1.5*nTabThickness+(nDsHeight-cvm_target.rows)/2, cvm_target.cols, cvm_target.rows)); cvm_target.copyTo(dst);
                            dst.release();

                            cv::line (cvm_record_rec, cv::Point(nDsWidth+nBorderThickness, 0), cv::Point(nDsWidth+nBorderThickness, cvm_record_rec.rows), c_darkgray, nBorderThickness);
                            cv::rectangle (cvm_record_rec, cv::Point(0, 1), cv::Point(cvm_record_rec.cols-1, cvm_record_rec.rows-1), c_darkgray, nBorderThickness);
                            cv::rectangle (cvm_record_rec, cv::Point(0, cvm_record_rec.rows-nDsHeight-nTabThickness-nBorderThickness), cv::Point(nDsWidth, cvm_record_rec.rows-nDsHeight-nBorderThickness), c_darkgray, CV_FILLED);
                            cv::rectangle (cvm_record_rec, cv::Point(0, cvm_record_rec.rows-2*nDsHeight-2*nTabThickness-nBorderThickness), cv::Point(nDsWidth, cvm_record_rec.rows-2*nDsHeight-nTabThickness-nBorderThickness), c_darkgray, CV_FILLED);

                            cv::putText (cvm_record_rec, "Tracked Surfaces", cv::Point(nTabThickness, cvm_record_rec.rows-2*nDsHeight-1.3*nTabThickness-nBorderThickness), nFont, 0.4, c_white, 1);
                            cv::putText (cvm_record_rec, "Down-Sampled", cv::Point(nTabThickness, cvm_record_rec.rows-nDsHeight-0.3*nTabThickness-nBorderThickness), nFont, 0.4, c_white, 1);
                            cv::putText (cvm_record_rec, "Candidate", cv::Point(nTabThickness, nBorderThickness+nDsHeight+3*nTabThickness), nFont, 0.6, c_cyan, 1);
                            cv::putText (cvm_record_rec, "Target", cv::Point(nTabThickness, nBorderThickness+nTabThickness), nFont, 0.6, c_lemon, 1);

                            break;}
                        case 2: {
                            cv::Mat dst;
                            dst = cvm_record_ds (cv::Rect(0, 0, cvm_track.cols, cvm_track.rows)); cvm_track.copyTo(dst);
                            dst = cvm_record_ds (cv::Rect(nDsWidth, 0, cvm_rec_ds.cols, cvm_rec_ds.rows)); cvm_rec_ds.copyTo(dst);
                            dst.release();
                            break;}
                        }
                    }
                }

                if (nCandID >= 0) {

                    // Print information about the color distance
                    if(flag_pcd) {
                        printf("Color distance (normalized rgb)   ");
                        for (int i = 0; i < nMemsCnt; i++) {
                            if(!vbProtoCand[i]) continue;
                            printf("%8.3f", veCandClrDist[i].first);
                        }
                        printf("\n");
                    }


                    int nCandKeyCnt, nCandRX, nCandRY, nCandRW, nCandRH;
                    Recognition (nCandID, nImgScale, nDsWidth, nTimeRatio, nMemsCnt, nTrackHistoBin_max, cvm_rgb_org, cvm_rec_org, cvm_rec_ds,
                                    stMems, stTrack.CntLost, nCandCnt, veCandClrDist[nCandID].first, nFoundCnt, nFoundNr, nFoundFrame,
                                    nCandKeyCnt, nCandRX, nCandRY, nCandRW, nCandRH,
                                    t_rec_found_start, t_rec_found_end, bSwitchRecordTime, nRecogRtNr, vnRecogRating_tmp, cvm_cand);

                    nTmpAttKeyCnt = nCandKeyCnt; nTmpAttWidth = nCandRW; nTmpAttHeight = nCandRH;

                    if (((vbFlagTask[stTID.nRecogOrg] || vbFlagTask[stTID.nRecVideo]) && nRecordMode)) {
                        switch (nRecordMode) {
                        case 1: {
                            cv::Mat src, dst;
                            dst = cvm_record_rec (cv::Rect(nDsWidth + 2*nBorderThickness, nBorderThickness, cvm_rec_org.cols, cvm_rec_org.rows)); cvm_rec_org.copyTo(dst);
                            dst = cvm_record_rec (cv::Rect(nBorderThickness, cvm_record_rec.rows - nDsHeight -  nBorderThickness, nDsWidth, nDsHeight)); cvm_rec_ds.copyTo(dst);
                            dst = cvm_record_rec (cv::Rect(nBorderThickness, cvm_record_rec.rows - 2*nDsHeight - nTabThickness - nBorderThickness, nDsWidth, nDsHeight)); cvm_track.copyTo(dst);
                            dst = cvm_record_rec (cv::Rect(nBorderThickness+(nDsWidth-cvm_target.cols)/2, nBorderThickness+1.5*nTabThickness+(nDsHeight-cvm_target.rows)/2, cvm_target.cols, cvm_target.rows)); cvm_target.copyTo(dst);

                            double cx = (double)nDsWidth/nCandRW, cy = (double)nDsHeight/nCandRH; if (cx > 1) cx = 1; if (cy > 1) cy = 1;
                            double ratio = min(cx,cy)*0.7;
                            int nCX = (nDsWidth-nCandRW*ratio)/2, nCY = (nDsHeight-nCandRH*ratio)/2;
                            cv::Mat src_ds;
                            src = cvm_cand (cv::Rect(nCandRX, nCandRY, nCandRW, nCandRH));
                            cv::resize (src, src_ds, cv::Size(), ratio, ratio, cv::INTER_AREA);
                            dst = cvm_record_rec (cv::Rect(nBorderThickness + nCX, nBorderThickness+nDsHeight+2*nTabThickness + nCY, src_ds.cols, src_ds.rows)); src_ds.copyTo(dst);
                            src.release(); dst.release(); src_ds.release();

                            cv::line (cvm_record_rec, cv::Point(nDsWidth+nBorderThickness, 0), cv::Point(nDsWidth+nBorderThickness, cvm_record_rec.rows), c_darkgray, nBorderThickness);
                            cv::rectangle (cvm_record_rec, cv::Point(0, 1), cv::Point(cvm_record_rec.cols-1, cvm_record_rec.rows-1), c_darkgray, nBorderThickness);
                            cv::rectangle (cvm_record_rec, cv::Point(0, cvm_record_rec.rows-nDsHeight-nTabThickness-nBorderThickness), cv::Point(nDsWidth, cvm_record_rec.rows-nDsHeight-nBorderThickness), c_darkgray, CV_FILLED);
                            cv::rectangle (cvm_record_rec, cv::Point(0, cvm_record_rec.rows-2*nDsHeight-2*nTabThickness-nBorderThickness), cv::Point(nDsWidth, cvm_record_rec.rows-2*nDsHeight-nTabThickness-nBorderThickness), c_darkgray, CV_FILLED);

                            cv::putText (cvm_record_rec, "Tracked Surfaces", cv::Point(nTabThickness, cvm_record_rec.rows-2*nDsHeight-1.3*nTabThickness-nBorderThickness), nFont, 0.4, c_white, 1);
                            cv::putText (cvm_record_rec, "Down-Sampled", cv::Point(nTabThickness, cvm_record_rec.rows-nDsHeight-0.3*nTabThickness-nBorderThickness), nFont, 0.4, c_white, 1);
                            cv::putText (cvm_record_rec, "Candidate", cv::Point(nTabThickness, nBorderThickness+nDsHeight+3*nTabThickness), nFont, 0.6, c_cyan, 1);
                            cv::putText (cvm_record_rec, "Target", cv::Point(nTabThickness, nBorderThickness+nTabThickness), nFont, 0.6, c_lemon, 1);

                            break;}
                        case 2: {
                            cv::Mat dst;
                            dst = cvm_record_ds (cv::Rect(0, 0, cvm_track.cols, cvm_track.rows)); cvm_track.copyTo(dst);
                            dst = cvm_record_ds (cv::Rect(nDsWidth, 0, cvm_rec_ds.cols, cvm_rec_ds.rows)); cvm_rec_ds.copyTo(dst);
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
                    ResetMemory (nObjsNrLimit, nTrackHistoBin_max, nRecogRtNr, stMems, nMemsCnt, nCandCnt, nFoundCnt, nFoundNr, vnRecogRating);

                    ResetRecTime();
                }

                if (vbFlagWnd[stTID.nRecogOrg]) {cvm_rec_org = cv::Scalar(0, 0, 0); cv::add(cvm_rec_org, cvm_rgb_org, cvm_rec_org);}
                if (vbFlagWnd[stTID.nRecogDs]) {cvm_rec_ds = cv::Scalar(0, 0, 0); cv::add(cvm_rec_ds, cvm_rgb_ds, cvm_rec_ds);}
            }
            //////////////////----------------------------------------------------------------------------------/////////////////////////
            //////////////////                                                                                  /////////////////////////
            //////////////////                          End of the RECOGNITIOON                                 /////////////////////////
            //////////////////                                                                                  /////////////////////////
            /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


        }//bFragTask[stTID.nSegmentation]
        else {
            if (bSwitchSegm) {
                bSwitchSegm = false;
                bSwitchRecog = false;

                bSwitchRecogNewCyc = true;

                vbFlagTask[stTID.nRecTime] = false;
                mnTimeMeas1.resize(0); mnTimeMeas2.resize(0);
                ResetMemory (nObjsNrLimit, nTrackHistoBin_max, nRecogRtNr, stMems, nMemsCnt, nCandCnt, nFoundCnt, nFoundNr, vnRecogRating);

                nCtrSegm = 0; nTimeSegm_acc = 0; nTimeSegm_avr = 0;
                nCtrTrack = 0; nTimeTrack_acc = 0; nTimeTrack_avr = 0;

                ResetRecTime();

                if (!vbFlagTask[stTID.nDepth]) {nCtrDepth = 0; nCtrBlur = 0; nTimeDepth_acc = 0; nTimeBlur_acc = 0; nTimeDepth_avr = 0; nTimeBlur_avr = 0;}
            }

            if (vbFlagWnd[stTID.nRecogOrg]) {cvm_rec_org = cv::Scalar(0, 0, 0); cv::add(cvm_rec_org, cvm_rgb_org, cvm_rec_org);}
            if (vbFlagWnd[stTID.nRecogDs]) {cvm_rec_ds = cv::Scalar(0, 0, 0); cv::add(cvm_rec_ds, cvm_rgb_ds, cvm_rec_ds);}
        }
        //////////////////----------------------------------------------------------------------------------/////////////////////////////
        //////////////////                                                                                  /////////////////////////////
        //////////////////                          End of the Main Process                                 /////////////////////////////
        //////////////////                                                                                  /////////////////////////////
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////








        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //////////////////                                                                                  /////////////////////////////
        //////////////////                  Other state of art methods for compairison                      /////////////////////////////
        //////////////////                                                                                  /////////////////////////////
        //////////////////----------------------------------------------------------------------------------/////////////////////////////

        //////////**  1. Graph-based segmentation  **////////////////////////////////
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
                    idx = GetPixelIdx(mnGSegmPts[i][j].x, mnGSegmPts[i][j].y, nDsWidth);
                    mnGSegmPtsIdx[i][j] = idx;
                    cvm_gbsegm.at<cv::Vec3b>(mnGSegmPts[i][j].y, mnGSegmPts[i][j].x) = cv::Vec3b(mnColorTab[i][0], mnColorTab[i][1], mnColorTab[i][2]);
                }
            }

            clock_gettime(CLOCK_MONOTONIC_RAW, &t_gbseg_end); nTimeGbSegm = double(timespecDiff(&t_gbseg_end, &t_gbseg_start)/nTimeRatio);


            //////*  Record Time  *///////
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



        //////////**  2. Sift on a whole image  **////////////////////////////////
        if (vbFlagTask[stTID.nSIFT]) {
            if(!vbFlagWnd[stTID.nSIFT]) {
                vbFlagWnd[stTID.nSIFT] = true;
                cv::namedWindow(vsWndName[stTID.nSIFT]);
                cvMoveWindow(vsWndName[stTID.nSIFT].data(), 300, 100);
            }

            //////*  SIFT  *////////////
            struct timespec t_sift_start, t_sift_end; clock_gettime(CLOCK_MONOTONIC_RAW, &t_sift_start); bTimeSiftWhole = true;

            Keypoint keypts, keypts_tmp;
            GetSiftKeypoints(cvm_rgb_org, nSiftScales, nSiftInitSigma, nSiftPeakThrs, 0, 0, cvm_rgb_org.cols, cvm_rgb_org.rows, keypts);
            keypts_tmp = keypts;

            clock_gettime (CLOCK_MONOTONIC_RAW, &t_sift_end); nTimeSiftWhole = double(timespecDiff(&t_sift_end, &t_sift_start)/nTimeRatio);


            //////*  FLANN  *////////////
            struct timespec t_flann_start, t_flann_end; clock_gettime(CLOCK_MONOTONIC_RAW, &t_flann_start);

            std::vector<int> vnSiftMatched; std::vector <double> vnDeltaScale; std::vector <double> vnDeltaOri;
            double nMaxDeltaOri = -999; double nMaxDeltaScale = -999; double nMinDeltaOri = 999; //double nMinDeltaScale = 999;
            int nKeyptsCnt = 0; int nFlannIM=0;

            Recognition_Flann (tcount, nFlannKnn, nFlannLibCols_sift, nFlannMatchFac, mnSiftExtraFeatures, FlannIdx_Sift, FLANNParam, keypts, nKeyptsCnt, nFlannIM, vnSiftMatched, vnDeltaScale, vnDeltaOri, nMaxDeltaOri, nMinDeltaOri, nMaxDeltaScale, nMinDeltaOri);

            //////  Filtering Gray Keypoints  ////////////
            double nDeltaScale=0; int nFlannTP=0;
            std::vector<bool> vbSiftTP(nKeyptsCnt, 0);
            if(nFlannIM > T_numb) CalcDeltaScaleOri(vnSiftMatched, vnDeltaScale, vnDeltaOri, nDeltaScale, nDeltaBinNo, nMaxDeltaOri, nMinDeltaOri, T_orient, T_scale, nFlannTP, vbSiftTP);
            else nFlannTP = 0;

            clock_gettime (CLOCK_MONOTONIC_RAW, &t_flann_end); nTimeFlannWhole = double(timespecDiff(&t_flann_end, &t_flann_start)/nTimeRatio);


            //////*  Visualizing the SIFT keypoints and the FLANN matching  *////////////
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
                    cv::rectangle(cvm_sift, cv::Point(keypts->row - 2, keypts->col - 2), cv::Point(keypts->row + 2, keypts->col + 2), c_white, 1);
                    cv::rectangle(cvm_sift, cv::Point(keypts->row - 3, keypts->col - 3), cv::Point(keypts->row + 3, keypts->col + 3), c_blue, 1);
                    cv::rectangle(cvm_sift, cv::Point(keypts->row - 1, keypts->col - 1), cv::Point(keypts->row + 1, keypts->col + 1), c_blue, 1);
                }
                nKeyptsCnt++;
                keypts= keypts->next;
            }
            FreeKeypoints(keypts); FreeKeypoints(keypts_tmp);
            clock_gettime(CLOCK_MONOTONIC_RAW, &t_siftdraw_end); nTimeSiftDrawWhole = double(timespecDiff(&t_siftdraw_end, &t_siftdraw_start)/nTimeRatio);


            //////*  Record Time  *///////////
            if (bRtSiftW) {
                if(vbFlagWnd[stTID.nSIFT]) CloseWindow(stTID.nSIFT);
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
        //////////////////----------------------------------------------------------------------------------/////////////////////////////
        //////////////////                                                                                  /////////////////////////////
        //////////////////               End of the state of art methods for compairison                    /////////////////////////////
        //////////////////                                                                                  /////////////////////////////
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////





        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //////////////////                                                                                  /////////////////////////////
        //////////////////                          Other NiVision functions                                /////////////////////////////
        //////////////////                                                                                  /////////////////////////////
        //////////////////----------------------------------------------------------------------------------/////////////////////////////


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
            sDat = sSnapDir + "/" + "snap_05_seg_noPostProc" + sImgExt; cv::imwrite(sDat, cvm_segment_raw);
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
                    if (vbFlagWnd[stTID.nRgbDs]) {sDat = sVideoDir + "/" + "video_01_rgb" + ".avi"; writer_rgb_ds.open(sDat, fourcc, (double)frame_rate, size_ds, true);}
                    if (vbFlagWnd[stTID.nTrack]) {sDat = sVideoDir + "/" + "video_07_trk" + ".avi"; writer_track.open(sDat, fourcc, (double)frame_rate, size_ds, true);}
                    if (vbFlagWnd[stTID.nRecogOrg]) {sDat = sVideoDir + "/" + "video_08_rec" + ".avi"; writer_rec_org.open(sDat, fourcc, (double)frame_rate, size_org, true);}
                    if (vbFlagWnd[stTID.nRecogDs]) {sDat = sVideoDir + "/" + "video_08_rec_ds" + ".avi"; writer_rec_ds.open(sDat, fourcc, (double)frame_rate, size_ds, true);}
                    break;
                case 1:
                    sDat = sVideoDir + "/" + "video_08_rec" + ".avi"; writer_total.open(sDat, fourcc, fr, size_total, true);
                    break;
                case 2:
                    sDat = sVideoDir + "/" + "video_08_rec" + ".avi"; writer_total.open(sDat, fourcc, fr, cv::Size(nDsWidth*2, nDsHeight), true);
                    break;
                }
            }
            else {
                switch (nRecordMode) {
                case 0:
                    if (vbFlagWnd[stTID.nRgbDs]) writer_rgb_ds.write(cvm_rgb_ds);
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
                    //sprintf(s_frame, "snap_%06i_10_total", nCtrFrame);
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
                    ResetMemory (nObjsNrLimit, nTrackHistoBin_max, nRecogRtNr, stMems, nMemsCnt, nCandCnt, nFoundCnt, nFoundNr, vnRecogRating);
                    ResetTime();
                }
                else {
                    if (nCtrFrame_tmp > nTimeMessFrmLimit - 1) {
                        vbFlagTask[stTID.nRecTime] = false;
                        clock_gettime(CLOCK_MONOTONIC_RAW, &t_total_end); nTimeTotal = double(timespecDiff(&t_total_end, &t_total_start)/nTimeRatio);
                    }
                }

                mnTimeMeas3.resize(nCtrFrame_tmp+1);
                mnTimeMeas3[nCtrFrame_tmp].assign(13, 0);

                mnTimeMeas3[nCtrFrame_tmp][0] = nTimePre;
                mnTimeMeas3[nCtrFrame_tmp][1] = nTimeDepth;
                mnTimeMeas3[nCtrFrame_tmp][2] = nTimeBlur;
                mnTimeMeas3[nCtrFrame_tmp][3] = nTimeSegm;
                mnTimeMeas3[nCtrFrame_tmp][4] = nTimeTrack;
                mnTimeMeas3[nCtrFrame_tmp][5] = nTimeAtt;
                mnTimeMeas3[nCtrFrame_tmp][6] = nTimeSift;
                mnTimeMeas3[nCtrFrame_tmp][7] = nTimeFlann;
                mnTimeMeas3[nCtrFrame_tmp][8] = nTimeRec;
                if (nCtrFrame_tmp) mnTimeMeas3[nCtrFrame_tmp-1][9] = nTimeTotal_prev;
                if (!vbFlagTask[stTID.nRecTime]) mnTimeMeas3[nCtrFrame_tmp][9] = nTimeTotal;
                mnTimeMeas3[nCtrFrame_tmp][10] = (float)nTmpAttKeyCnt;
                mnTimeMeas3[nCtrFrame_tmp][11] = (float)nTmpAttWidth;
                mnTimeMeas3[nCtrFrame_tmp][12] = (float)nTmpAttHeight;
            }
        }
        else {
            if (bSwitchRecordTime) {
                bSwitchRecordTime = false;

                mkdir(sDataDir.data(), 0777); MakeNiDirectory (sTimeDirPref, sTimeDir);
                sTimeFile = sTimeDir + "/" + "time.txt";
                sTimeFile_detail = sTimeDir + "/" + "time_detail.xlsx";

                PrintTimes1 (sTimeFile, nCtrRecCycle, mnTimeMeas1, mnTimeMeas2);
                PrintTimes2 (sTimeFile, nCtrRecCycle, nTimeRecFound_avr, nTimeRecFound_min, nTimeRecFound_max, vnRecogRating, mnTimeMeas2,
                             nCtrFrame_tmp, nTimeTotal_avr, nFrameRate_avr, nTimeDepth_avr, nTimeBlur_avr, nTimeSegm_avr, nTimeTrack_avr, nTimeAtt_avr, nTimeRec_avr, nTimeSift_avr, nTimeFlann_avr);
                PrintTimes3 (sTimeFile_detail, mnTimeMeas3);


                bSwitchRecogNewCyc = true;
                mnTimeMeas1.resize(0); mnTimeMeas2.resize(0);
                ResetMemory (nObjsNrLimit, nTrackHistoBin_max, nRecogRtNr, stMems, nMemsCnt, nCandCnt, nFoundCnt, nFoundNr, vnRecogRating);

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

                int dp = int(stTrack.DPos*100), ds = int(stTrack.DSize*100), dc = int(stTrack.DClr*100), dt = int(stTrack.Dist*100);
                int fp = int(stTrack.FPos*100), fs = int(stTrack.FSize*100), fc = int(stTrack.FClr*100);

                cvCreateTrackbar(vsTrackbarName[22].data(), vsWndName[stTID.nPrmSegm].data(), &dp, 100, TrackbarHandler_DistPos);
                cvCreateTrackbar(vsTrackbarName[23].data(), vsWndName[stTID.nPrmSegm].data(), &ds, 100, TrackbarHandler_DistSize);
                cvCreateTrackbar(vsTrackbarName[24].data(), vsWndName[stTID.nPrmSegm].data(), &dc, 100, TrackbarHandler_DistClr);
                cvCreateTrackbar(vsTrackbarName[25].data(), vsWndName[stTID.nPrmSegm].data(), &fp, 100, TrackbarHandler_FacPos);
                cvCreateTrackbar(vsTrackbarName[26].data(), vsWndName[stTID.nPrmSegm].data(), &fs, 100, TrackbarHandler_FacSize);
                cvCreateTrackbar(vsTrackbarName[27].data(), vsWndName[stTID.nPrmSegm].data(), &fc, 100, TrackbarHandler_FacClr);
                cvCreateTrackbar(vsTrackbarName[28].data(), vsWndName[stTID.nPrmSegm].data(), &dt, 500, TrackbarHandler_DistTotal);

                //cvCreateTrackbar(vsTrackbarName[38].data(), vsWndName[stTID.nPrmSegm].data(), &stTrack.CntStable, 5, TrackbarHandler_none);
                //cvCreateTrackbar(vsTrackbarName[39].data(), vsWndName[stTID.nPrmSegm].data(), &stTrack.CntLost, 2, TrackbarHandler_none);
                cvCreateTrackbar(vsTrackbarName[31].data(), vsWndName[stTID.nPrmSegm].data(), &nAttSizeMax, 1000, TrackbarHandler_ProMax);
                cvCreateTrackbar(vsTrackbarName[32].data(), vsWndName[stTID.nPrmSegm].data(), &nAttSizeMin, 1000, TrackbarHandler_ProMin);
                cvCreateTrackbar(vsTrackbarName[33].data(), vsWndName[stTID.nPrmSegm].data(), &nAttPtsMin, 1000, TrackbarHandler_none);
            }
        }

        if (vbFlagTask[stTID.nPrmRecog]) {
            if(!vbFlagWnd[stTID.nPrmRecog]) {
                vbFlagWnd[stTID.nPrmRecog] = true;
                cv::namedWindow(vsWndName[stTID.nPrmRecog]);
                cvMoveWindow(vsWndName[stTID.nPrmRecog].data(), 850, 100);

                int recdc = int(nRecogDClr*100);
                int sift_sigma = int(nSiftInitSigma*10);
                int sift_peak = int(nSiftPeakThrs*1000);
                int flannmf = int(nFlannMatchFac*100);

                cvCreateTrackbar(vsTrackbarName[1].data(), vsWndName[stTID.nPrmRecog].data(), &nSnapFormat, 2, TrackbarHandler_none);
                //cvCreateTrackbar(vsTrackbarName[2].data(), vsWndName[stTID.nPrmRecog].data(), &dlimit, 100, TrackbarHandler_ZLimit);
                //cvCreateTrackbar(vsTrackbarName[3].data(), vsWndName[stTID.nPrmRecog].data(), &nDGradFilterSize, 27, TrackbarHandler_ZGradFilterSize);

                //cvCreateTrackbar(vsTrackbarName[11].data(), vsWndName[stTID.nPrmRecog].data(), &nSegmSizeThres, 1000, TrackbarHandler_none);
                //cvCreateTrackbar(vsTrackbarName[12].data(), vsWndName[stTID.nPrmRecog].data(), &mp, 10000, TrackbarHandler_MergePro);

                cvCreateTrackbar(vsTrackbarName[99].data(), vsWndName[stTID.nPrmRecog].data(), &nAttTDMode, 2, TrackbarHandler_none);
                cvCreateTrackbar(vsTrackbarName[36].data(), vsWndName[stTID.nPrmRecog].data(), &recdc, 120, TrackbarHandler_ColorThres);
                cvCreateTrackbar(vsTrackbarName[41].data(), vsWndName[stTID.nPrmRecog].data(), &nSiftScales, 5, TrackbarHandler_SiftScales);
                cvCreateTrackbar(vsTrackbarName[42].data(), vsWndName[stTID.nPrmRecog].data(), &sift_sigma, 20, TrackbarHandler_SiftSigma);
                cvCreateTrackbar(vsTrackbarName[43].data(), vsWndName[stTID.nPrmRecog].data(), &sift_peak, 100, TrackbarHandler_SiftPeak);
                cvCreateTrackbar(vsTrackbarName[51].data(), vsWndName[stTID.nPrmRecog].data(), &nFlannKnn, 5, TrackbarHandler_FlannKnn);
                cvCreateTrackbar(vsTrackbarName[52].data(), vsWndName[stTID.nPrmRecog].data(), &flannmf, 100, TrackbarHandler_FlannMFac);
                cvCreateTrackbar(vsTrackbarName[53].data(), vsWndName[stTID.nPrmRecog].data(), &nFlannMatchCnt, 100, TrackbarHandler_none);
                cvCreateTrackbar(vsTrackbarName[54].data(), vsWndName[stTID.nPrmRecog].data(), &flag_pcd, 1, TrackbarHandler_none);
            }
        }

        ////** Reset Parameters **////////////////
        if (vbFlagTask[stTID.nRstPrm]) ResetParameter();

        //////////////////----------------------------------------------------------------------------------/////////////////////////////
        //////////////////                                                                                  /////////////////////////////
        //////////////////                      End of the other functions                                  /////////////////////////////
        //////////////////                                                                                  /////////////////////////////
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////






        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //////////////////                                                                                  /////////////////////////////
        //////////////////                            Drawing Operations                                    /////////////////////////////
        //////////////////                                                                                  /////////////////////////////
        //////////////////----------------------------------------------------------------------------------/////////////////////////////

        if(vbFlagWnd[stTID.nRgbOrg]) cv::imshow(vsWndName[stTID.nRgbOrg], cvm_rgb_org);
        if(vbFlagWnd[stTID.nRgbDs]) cv::imshow(vsWndName[stTID.nRgbDs], cvm_rgb_ds);
        if(vbFlagWnd[stTID.nDepth]) {AttachImgs(cvm_depth, cvm_dgrady, cvm_dgrady_blur, cvm_dgrady_smth, nDsWidth, nDsHeight, cvm_depth_process); cv::imshow(vsWndName[stTID.nDepth], cvm_depth_process);}
        if(vbFlagWnd[stTID.nSegm]) cv::imshow(vsWndName[stTID.nSegm], cvm_segment);
        if(vbFlagWnd[stTID.nTrack]) cv::imshow(vsWndName[stTID.nTrack], cvm_track);
        if(vbFlagWnd[stTID.nAtt]) cv::imshow(vsWndName[stTID.nAtt], cvm_att);
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

        nTimePre_acc += nTimePre; nTimePre_avr = nTimePre_acc/nCtrFrame;
        if (bTimeDepth) {nCtrDepth++; nTimeDepth_acc += nTimeDepth; nTimeDepth_avr = nTimeDepth_acc/nCtrDepth;}
        if (bTimeBlur) {nCtrBlur++; nTimeBlur_acc += nTimeBlur; nTimeBlur_avr = nTimeBlur_acc/nCtrBlur;}
        if (bTimeSegm) {nCtrSegm++; nTimeSegm_acc += nTimeSegm; nTimeSegm_avr = nTimeSegm_acc/nCtrSegm;}
        if (bTimeTrack) {nCtrTrack++; nTimeTrack_acc += nTimeTrack; nTimeTrack_avr = nTimeTrack_acc/nCtrTrack;}
        if (bTimeRec) {nCtrRec++; nTimeRec_acc += nTimeRec; nTimeRec_avr = nTimeRec_acc/nCtrRec; nTimeAtt_acc += nTimeAtt; nTimeAtt_avr = nTimeAtt_acc/nCtrRec;}
        if (bTimeSift) {nCtrSift++; nTimeSift_acc += nTimeSift; nTimeSift_avr = nTimeSift_acc/nCtrSift;}
        if (bTimeFlann) {nCtrFlann++; nTimeFlann_acc += nTimeFlann; nTimeFlann_avr = nTimeFlann_acc/nCtrFlann;}
        if (bTimeGbSegm) {nCtrGbSegm++; nTimeGbSegm_acc += nTimeGbSegm; nTimeGbSegm_avr = nTimeGbSegm_acc/nCtrGbSegm;}
        if (bTimeSiftWhole) {nCtrSiftWhole++; nTimeSiftWhole_acc += nTimeSiftWhole; nTimeSiftWhole_avr = nTimeSiftWhole_acc/nCtrSiftWhole;
            nTimeFlannWhole_acc += nTimeFlannWhole; nTimeFlannWhole_avr = nTimeFlannWhole_acc/nCtrSiftWhole;}
        if (vbFlagTask[stTID.nRecognition]) {if (bTimeSift) {nCtrFrame_tmp++;
                nTimeTotal_acc += nTimeTotal; nTimeTotal_avr = nTimeTotal_acc/nCtrFrame_tmp; nFrameRate_avr = 1000/nTimeTotal_avr;
                nTimeFrame_acc += nTimeFrame; nTimeFrame_avr = nTimeFrame_acc/nCtrFrame_tmp;}}
        else {nCtrFrame_tmp++; nTimeTotal_acc += nTimeTotal; nTimeTotal_avr = nTimeTotal_acc/nCtrFrame_tmp; nFrameRate_avr = 1000/nTimeTotal_avr;
            nTimeFrame_acc += nTimeFrame; nTimeFrame_avr = nTimeFrame_acc/nCtrFrame_tmp;}


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
            DrawInfo (cvm_main, nSurfCnt, nPadCol1, nTimeTotal,
                      nTimePre_avr, nTimeDepth_avr, nTimeBlur_avr,
                      nTimeSegm, nTimeTrack, nTimeAtt, nTimeRec, nTimeSift, nTimeFlann,
                      nTimeSegm_avr, nTimeTrack_avr, nTimeAtt_avr, nTimeRec_avr, nTimeSift_avr, nTimeFlann_avr,
                      nTimeRecCycle, nFoundNr, nTimeRecFound,
                      nTimeGbSegm, nTimeSiftWhole, nTimeSiftDrawWhole, nTimeFlannWhole, nTimeGbSegm_avr, nTimeSiftWhole_avr, nTimeFlannWhole_avr);
        }
        if(vbFlagTask[stTID.nPrmInfo]) {
            DrawSettings(cvm_main, nPadCol2, nSnapFormat, nDLimit, nDGradFilterSize,
                         stTrack,
                         nAttSizeMax, nAttSizeMin, nAttPtsMin, nAttTDMode,
                         nRecogDClr, nSiftScales, nSiftInitSigma, nSiftPeakThrs, nFlannKnn, nFlannMatchFac, nFlannMatchCnt,
                         nGSegmSigma, nGSegmGrThrs, nGSegmMinSize);
        }
        cv::imshow(sTitle, cvm_main);

        //////////////////----------------------------------------------------------------------------------/////////////////////////////
        //////////////////                                                                                  /////////////////////////////
        //////////////////                      End of the Drawing Operations                               /////////////////////////////
        //////////////////                                                                                  /////////////////////////////
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


        nCvCurrentkey = cv::waitKey(1);
        nCtrFrame++;
    }

    cvm_main.release();
    cvm_set_seg.release(); cvm_set_rec.release();
    cvm_rgb_org.release(); cvm_rgb_ds.release();
    cvm_depth.release(); cvm_dgradx.release(); cvm_dgrady.release();
    cvm_segment.release(); cvm_segment_raw.release();
    cvm_track.release();
    cvm_att.release(); cvm_cand.release();
    cvm_rec_org.release(); cvm_rec_ds.release();

    cvm_gbsegm.release(); cvm_sift.release();

    cvm_record_rec.release(); cvm_record_ds.release();

    cvm_dgrady_blur.release(); cvm_dgrady_smth.release(); cvm_depth_process.release();

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

    }

    ~NiVisionNode() {
    }

    void cloudCb (const sensor_msgs::PointCloud2ConstPtr& cloud){
        m.lock ();

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
                nDsWidth = cloud_->width;
                nDsHeight = cloud_->height;
                nDsSize = nDsWidth * nDsHeight;

                boost::thread visualization_thread (&updateImage);
            }
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
            cv_ptr->image.copyTo(cvm_image_camera);

            m.unlock ();
        }
        catch (cv_bridge::Exception& e) {ROS_ERROR("cv_bridge exception: %s", e.what()); return;}

    }
};



int main(int argc, char** argv)
{
    //////////// Set parameters as default ///////////////////////////////
    InitParameter(argc, argv);

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
