/*
 * Initialization of parameters with the input from the console
 * Defining names for button, windows and trackbars
 * Drawing the gui
 * Handling mouse events
 * Open and closing of windows
 * Different trackbarhandlers
 *
 * Since the early version of opencv did not provide good methods to create a gui, we had to develop
 * our own approach. The gui is basically a window on which different rectangles are drawn which
 * represent the button.
 */

////////// System Paramters ///////////////////////////////////////
bool bDepthDispMode = false, bDepthDispMode_default = false;
double nDLimit = 0, nDLimit_default = 3;


double nDSegmDThres = 0, nDSegmDThres_default = 0.04;
double nDGradConst = 0, nDGradConst_default = 0.2;
int nDGradFilterMode = 0, nDGradFilterMode_default = 2;
int nDGradFilterSize = 0, nDGradFilterSize_default = 5;
int nDGradSmthMode = 0, nDGradSmthMode_default = 1;
int nDGradSmthCenter = 128, nDGradSmthCenter_default = 128;
double nDGradSmthBThres1 = 0, nDGradSmthBThres1_default = 0.0015;
double nDGradSmthBThres2 = 0, nDGradSmthBThres2_default = 0.0045;
int nDGradSmthBnd1 = 0;
int nDGradSmthBnd2 = 0;
double nDGradSmthFac = 3.0, nDGradSmthFac_default = 1.0;

int nDSegmSizeThres = 0, nDSegmSizeThres_default = 200;
double nDSegmGradDist = 0, nDSegmGradDist_default = 0.005;
int nDSegmCutSize = 0, nDSegmCutSize_default = 10;

double nGSegmSigma = 0, nGSegmSigma_default = 0.8;
int nGSegmGrThrs = 0, nGSegmGrThrs_default = 500;
int nGSegmMinSize = 0, nGSegmMinSize_default = 450;

struct TrackProp {
    int Mode, ClrMode, HistoBin;
    double DPos, DSize, DClr, Dist, FPos, FSize, FClr, MFac;
    int CntMem, CntStable, CntDisap;
}; TrackProp stTrack, stTrack_default;

int nAttTDMode = 0, nAttTDMode_default = 0;
int nAttSizeMax = 0, nAttSizeMax_default = 350;
int nAttSizeMin = 0, nAttSizeMin_default = 100;
int nAttPtsMin = 0, nAttPtsMin_default = 200;
int nAttAspect1 = 0, nAttAspect1_default = 20;
int nAttAspect2 = 0, nAttAspect2_default = 30;


int nSiftScales = 0, nSiftScales_default = 3;
double nSiftInitSigma = 0, nSiftInitSigma_default = 1.6;
double nSiftPeakThrs = 0, nSiftPeakThrs_default = 0.015;

int nFlannKnn = 0, nFlannKnn_default = 2;
double nFlannTargetPrecision = 0, nFlannTargetPrecision_default = 0.9;
double nFlannMatchFac, nFlannMatchFac_default = 0.7;
int nFlannMatchCnt, nFlannMatchCnt_default = 6;


int nRecogFeature = 0, nRecogFeature_default = 0;
double nRecogDClr = 0, nRecogDClr_default = 0.35;
bool bRecogClrMask = false, bRecogClrMask_default = false;
int nRecogMaskDispMode = 0, nRecogMaskDispMode_default = 0;


int nRecordMode = 0, nRecordMode_default = 0;
int nTimeMessFrmLimit = 0, nTimeMessFrmLimit_default = 50;
int nSnapFormat = 0, nSnapFormat_default = 0;
int flag_pcd = 0; // Flag for printing color distance

std::string sDataDir;


struct ProtoProp {
    std::vector<std::vector<int> > mnRect, mnRCenter; std::vector<std::vector<float> > mnCubic, mnCCenter;
    std::vector<int> vnLength; std::vector<std::vector<float> > mnColorHist;
    std::vector<int> vnMemoryCnt, vnStableCnt, vnDisapCnt;
};


/////// for User Interface /////////////////////////
int nFont = CV_FONT_HERSHEY_SIMPLEX;
float nFontSize = 0.4;
int nBtnSize = 80;
float nBtnFontSize = 0.5*nBtnSize/100;
int nTaskNrMax = 40;
bool bFlagEnd = false;
std::vector<bool> vbFlagTask(nTaskNrMax, 0);
std::vector<bool> vbFlagWnd(nTaskNrMax, 0);
std::vector<std::vector<int> > mnBtnPos(nTaskNrMax, std::vector<int>(4,0));
std::vector<std::vector<int> > mWndPos(nTaskNrMax, std::vector<int>(2,0));

std::string sTitle = "NI Vision Launcher";
std::vector<std::string> vsWndName(nTaskNrMax);
std::vector<std::string> vsBtnName(nTaskNrMax);
std::vector<std::string> vsTrackbarName(100);

cv::Scalar c_white(255, 255, 255);
cv::Scalar c_gray(127, 127, 127);
cv::Scalar c_darkgray(63, 63, 63);
cv::Scalar c_black(0, 0, 0);
cv::Scalar c_red(0, 0, 255);
cv::Scalar c_green(0, 255, 0);
cv::Scalar c_blue(255, 0, 0);
cv::Scalar c_cyan(255, 255, 0);
cv::Scalar c_magenta(255, 0, 255);
cv::Scalar c_yellow(0, 255, 255);
cv::Scalar c_lemon(32, 255, 2550);
cv::Scalar c_darkblue(127, 0, 0);
cv::Scalar c_darkgreen(0, 127, 0);
cv::Scalar c_darkred(0, 0, 127);
cv::Scalar c_lightblue(255, 127, 0);
cv::Scalar c_turkey(127, 127, 0);
cv::Scalar c_lightgreen(0, 255, 127);
cv::Scalar c_orange(0, 127, 255);
cv::Scalar c_kakki(0, 127, 127);
cv::Scalar c_pink(127, 0, 255);
cv::Scalar c_violet(255, 0, 127);

////////// ID of Tasks //////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct TaskID {
    int nRgbOrg, nRgbDs, nDepth, nInfo, nRecVideo, nSnap;
    int nSegmentation, nDSegm, nGSegm, nTrack, nProto;
    int nRecognition, nRecogOrg, nRecogDs, nSIFT;
    int nRecTime, nRstTime, nPrmInfo, nPrmSett, nPrmSegm, nPrmRecog, nRstPrm;
}; struct TaskID stTID;


//// for time estimation ////
int nCntFrame = 1, nCntFrame_tmp = 0;
int nCntDepth = 0, nCntBlur = 0;
bool bTimeDepth, bTimeBlur;
double nTimePre, nTimeDepth, nTimeBlur, nTimeTotal, nTimeFrame;
double nTimePre_acc = 0, nTimeDepth_acc = 0, nTimeBlur_acc = 0, nTimeTotal_acc = 0, nTimeFrame_acc;
double nTimePre_avr = 0, nTimeDepth_avr = 0, nTimeBlur_avr = 0, nTimeTotal_avr = 0, nTimeFrame_avr = 0, nFrameRate_avr = 0;
bool bTimeSegm, bTimeTrack, bTimeRec, bTimeSift, bTimeFlann;
int nCntSegm = 0, nCntTrack = 0, nCntRec = 0, nCntSift = 0, nCntFlann = 0;
double nTimeSegm, nTimeTrack, nTimeAtt, nTimeRec, nTimeSift, nTimeFlann;
double nTimeSegm_acc = 0, nTimeTrack_acc = 0, nTimeAtt_acc = 0, nTimeRec_acc = 0, nTimeSift_acc = 0, nTimeFlann_acc = 0;
double nTimeSegm_avr = 0, nTimeTrack_avr = 0, nTimeAtt_avr = 0, nTimeRec_avr = 0, nTimeSift_avr = 0, nTimeFlann_avr = 0;
int nCntRecCycle = 0, nCntGbSegm = 0, nCntSiftWhole = 0;
bool bTimeRecogCycle, bTimeGbSegm, bTimeSiftWhole;
double nTimeRecCycle, nTimeRecFound, nTimeGbSegm, nTimeSiftWhole, nTimeSiftDrawWhole, nTimeFlannWhole;
double nTimeRecCycle_acc = 0, nTimeRecFound_acc = 0, nTimeGbSegm_acc = 0, nTimeSiftWhole_acc = 0, nTimeFlannWhole_acc = 0;
double nTimeRecCycle_avr = 0, nTimeRecFound_avr = 0, nTimeGbSegm_avr = 0, nTimeSiftWhole_avr = 0, nTimeFlannWhole_avr = 0;
double nTimeRecFound_max = 0, nTimeRecFound_min = 100000;


//////// for Flann Libraries ////////////////////
std::string sTarget;
std::string sSIFTLibFileName;
std::string sColorLibFileName;
std::string sTargetImgPath;
std::string sLibFilePath;
std::vector<std::vector<float> > mnColorHistY_lib;

flann_index_t FlannIdx_Sift;
int nFlannLibCols_sift;
float* nFlannDataset;
flann_index_t FlannIdx_Color;
struct FLANNParameters FLANNParam;


int tcount = 1;
std::vector <std::vector <float> > mnSiftExtraFeatures;
int nDeltaBinNo;
int T_numb;
double T_orient;
double T_scale;


/* BUILDING COLOR INDEX
 * This function stores the feature vectors from the lib file in row major form and returns a pointer to the first address.
 *
 * Input:
 * mFeatureSet - matrix of features which was extracted from the lib file
 *
 * Output:
 * Pointer to the first address of the feature vector
 */
float* ReadFlannDataset_Color (cv::Mat mFeatureSet) {
    float *data;
    float *p;
    int i,j;
    data = (float*) malloc (mFeatureSet.rows * mFeatureSet.cols * sizeof(float));
    if (!data) {printf("Cannot allocate memory.\n"); exit(1);}

    printf("Memory allocated for Color FLANN: %d * float\n", mFeatureSet.rows * mFeatureSet.cols);
    p = data;

    mnColorHistY_lib.resize(mFeatureSet.rows);
    for (i = 0; i < mFeatureSet.rows; ++i) {
        mnColorHistY_lib[i].resize(mFeatureSet.cols);
        for (j = 0; j < mFeatureSet.cols; ++j) {
            float tmp = mFeatureSet.at<float>(i, j);
            *p = tmp;
            p++;
            mnColorHistY_lib[i][j] = mFeatureSet.at<float>(i, j);
        }
    }

    stTrack.HistoBin = round(pow(mFeatureSet.cols,(1/3.0)));
    return data;
}



/* BUILDING SIFT INDEX
 * This function stores the feature vectors from the lib file in row major form and returns a pointer to the first address.
 *
 * Input:
 * mFeatureSet - matrix of features which was extracted from the lib file
 *
 * Output:
 * Pointer to the first address of the feature vector
 */
float* ReadFlannDataset_SiftOnePos (cv::Mat mFeatureSet) {

    nFlannLibCols_sift = 128;

    float *data;
    float *p;
    int i,j;

    switch (nRecogFeature) {
    case 20: {
        data = (float*) malloc(mFeatureSet.rows * nFlannLibCols_sift * sizeof(float));
        mnSiftExtraFeatures.resize(mFeatureSet.rows, std::vector<float> (5,0));
        if (!data) {printf("Cannot allocate memory.\n"); exit(1);}

        printf("Memory allocated for FLANN: %d * float\n",mFeatureSet.rows * nFlannLibCols_sift);
        p = data;

        for (i = 0; i < mFeatureSet.rows; ++i) {
            for (j = 0; j < nFlannLibCols_sift; ++j) {
                float tmp = mFeatureSet.at<float>(i,j);
                *p=tmp;
                p++;
            }

            mnSiftExtraFeatures[i][0] = mFeatureSet.at<float>(i,128); //row
            mnSiftExtraFeatures[i][1] = mFeatureSet.at<float>(i,129); //col
            mnSiftExtraFeatures[i][2] = mFeatureSet.at<float>(i,130); //scale
            mnSiftExtraFeatures[i][3] = mFeatureSet.at<float>(i,131); //ori
            mnSiftExtraFeatures[i][4] = mFeatureSet.at<float>(i,132); //fpyramidscale
        }
    } break;
    case 30:
        break;
    }

    return data;
}



/* Build Flann Index
 */
void BuildFlannIndex (int libnr, std::string sLibFileName) {   //Read the library file
    cv::Mat mFeatureSet;
    cv::FileStorage fs(sLibFileName, cv::FileStorage::READ);
    fs["TestObjectFeatureVectors"] >> mFeatureSet;
    fs.release();
    float speedup;

    FLANNParameters DEFAULT_FLANN_PARAMETERS_USR = {
        FLANN_INDEX_KDTREE,
        32, 0.2f, 0.0f,
        4, 4,
        32, 11, FLANN_CENTERS_RANDOM,
        0.9f, 0.01f, 0, 0.1f,
        FLANN_LOG_NONE, 0
    };

    //FLANNParam = DEFAULT_FLANN_PARAMETERS;
    FLANNParam = DEFAULT_FLANN_PARAMETERS_USR;
    FLANNParam.algorithm = FLANN_INDEX_KDTREE;
    FLANNParam.trees = 8;
    FLANNParam.log_level = FLANN_LOG_INFO;
    FLANNParam.checks = 64;
    FLANNParam.target_precision = 0.9;

    switch (libnr) {        // 1: Color histogram, 2: SIFT for one view-point
    case 1:
        nFlannDataset = ReadFlannDataset_Color(mFeatureSet);  //Store the Input file into memory!
        //FlannIdx_Color = flann_build_index(nFlannDataset, mFeatureSet.rows, mFeatureSet.rows, &speedup, &FLANNParam);
        break;
    case 2:
        nFlannDataset = ReadFlannDataset_SiftOnePos(mFeatureSet);  //Store the Input file into memory!
        FlannIdx_Sift = flann_build_index(nFlannDataset, mFeatureSet.rows, nFlannLibCols_sift, &speedup, &FLANNParam);
        break;
    }
}




/* Initialize parameter with the console input
 *
 * Input:
 * argc - number of input arguments on the console
 * argv - console input
 */
void InitParameter (int argc, char** argv) {

    stTrack.Mode = 0, stTrack_default.Mode = 0;
    stTrack.ClrMode = 0, stTrack_default.ClrMode = 1;
    stTrack.HistoBin = 0, stTrack_default.HistoBin = 8;
    stTrack.DPos = 0, stTrack_default.DPos = 0.2;
    stTrack.DSize = 0, stTrack_default.DSize = 0.15;
    stTrack.DClr = 0, stTrack_default.DClr = 0.2;
    stTrack.Dist = 0, stTrack_default.Dist = 1.6;
    stTrack.FPos = 0, stTrack_default.FPos = 0.1;
    stTrack.FSize = 0, stTrack_default.FSize = 0.5;
    stTrack.FClr = 0, stTrack_default.FClr = 0.4;
    stTrack.MFac = 0, stTrack_default.MFac = 100;
    stTrack.CntMem = 0, stTrack_default.CntMem = 5;
    stTrack.CntStable = 0, stTrack_default.CntStable = 2;
    stTrack.CntDisap = 0, stTrack_default.CntDisap = 1;

    terminal_tools::parse_argument (argc, argv, "-ddmod", bDepthDispMode);
    if(bDepthDispMode == 0) bDepthDispMode = 0; bDepthDispMode_default = bDepthDispMode;
    terminal_tools::parse_argument (argc, argv, "-dlim", nDLimit);
    if(nDLimit == 0) nDLimit = nDLimit_default; nDLimit_default = nDLimit;

    terminal_tools::parse_argument (argc, argv, "-dgc", nDGradConst);
    terminal_tools::parse_argument (argc, argv, "-dgfmod", nDGradFilterMode);
    if(nDGradFilterMode == 0) nDGradFilterMode = nDGradFilterMode_default; nDGradFilterMode_default = nDGradFilterMode;
    terminal_tools::parse_argument (argc, argv, "-dgfsize", nDGradFilterSize);
    if(nDGradFilterSize == 0) nDGradFilterSize = nDGradFilterSize_default; nDGradFilterSize_default = nDGradFilterSize;
    if(nDGradConst == 0) nDGradConst = nDGradConst_default; nDGradConst_default = nDGradConst;
    terminal_tools::parse_argument (argc, argv, "-dgsmtmod", nDGradSmthMode);
    if(nDGradSmthMode == 0) nDGradSmthMode = nDGradSmthMode_default; nDGradSmthMode_default = nDGradSmthMode;
    terminal_tools::parse_argument (argc, argv, "-dgtauf1", nDGradSmthBThres1);
    if(nDGradSmthBThres1 == 0) nDGradSmthBThres1 = nDGradSmthBThres1_default; nDGradSmthBThres1_default = nDGradSmthBThres1;
    terminal_tools::parse_argument (argc, argv, "-dgtauf2", nDGradSmthBThres2);
    if(nDGradSmthBThres2 == 0) nDGradSmthBThres2 = nDGradSmthBThres2_default; nDGradSmthBThres2_default = nDGradSmthBThres2;

    terminal_tools::parse_argument (argc, argv, "-dstaud", nDSegmDThres);
    if(nDSegmDThres == 0) nDSegmDThres = nDSegmDThres_default; nDSegmDThres_default = nDSegmDThres;
    terminal_tools::parse_argument (argc, argv, "-dstaug", nDSegmGradDist);
    if(nDSegmGradDist == 0) nDSegmGradDist = nDSegmGradDist_default; nDSegmGradDist_default = nDSegmGradDist;
    terminal_tools::parse_argument (argc, argv, "-dsgcs", nDSegmCutSize);
    if(nDSegmCutSize == 0) nDSegmCutSize = nDSegmCutSize_default; nDSegmCutSize_default = nDSegmCutSize;
    terminal_tools::parse_argument (argc, argv, "-dstaus", nDSegmSizeThres);
    if(nDSegmSizeThres == 0) nDSegmSizeThres = nDSegmSizeThres_default; nDSegmSizeThres_default = nDSegmSizeThres;

    nDGradSmthBnd1 = nDGradSmthBThres1*254/nDSegmDThres/2; nDGradSmthBnd2 = nDGradSmthBThres2*254/nDSegmDThres/2;

    terminal_tools::parse_argument (argc, argv, "-gssigma", nGSegmSigma);
    if(nGSegmSigma == 0) nGSegmSigma = nGSegmSigma_default; nGSegmSigma_default = nGSegmSigma;
    terminal_tools::parse_argument (argc, argv, "-gsgth", nGSegmGrThrs);
    if(nGSegmGrThrs == 0) nGSegmGrThrs = nGSegmGrThrs_default; nGSegmGrThrs_default = nGSegmGrThrs;
    terminal_tools::parse_argument (argc, argv, "-gsmins", nGSegmMinSize);
    if(nGSegmMinSize == 0) nGSegmMinSize = nGSegmMinSize_default; nGSegmMinSize_default = nGSegmMinSize;


    terminal_tools::parse_argument (argc, argv, "-trkmod", stTrack.Mode);
    if(stTrack.Mode == 0) stTrack.Mode = stTrack_default.Mode; stTrack_default.Mode = stTrack.Mode;
    terminal_tools::parse_argument (argc, argv, "-trkcmod", stTrack.ClrMode);
    if(stTrack.ClrMode == 0) stTrack.ClrMode = stTrack_default.ClrMode; stTrack_default.ClrMode = stTrack.ClrMode;

    terminal_tools::parse_argument (argc, argv, "-trkdp", stTrack.DPos);
    if(stTrack.DPos == 0) stTrack.DPos = stTrack_default.DPos; stTrack_default.DPos = stTrack.DPos;
    terminal_tools::parse_argument (argc, argv, "-trkds", stTrack.DSize);
    if(stTrack.DSize == 0) stTrack.DSize = stTrack_default.DSize; stTrack_default.DSize = stTrack.DSize;
    terminal_tools::parse_argument (argc, argv, "-trkdc", stTrack.DClr);
    if(stTrack.DClr == 0) stTrack.DClr = stTrack_default.DClr; stTrack_default.DClr = stTrack.DClr;
    terminal_tools::parse_argument (argc, argv, "-trkd", stTrack.Dist);
    if(stTrack.Dist == 0) stTrack.Dist = stTrack_default.Dist; stTrack_default.Dist = stTrack.Dist;
    terminal_tools::parse_argument (argc, argv, "-trkpf", stTrack.FPos);
    if(stTrack.FPos == 0) stTrack.FPos = stTrack_default.FPos; stTrack_default.FPos = stTrack.FPos;
    terminal_tools::parse_argument (argc, argv, "-trksf", stTrack.FSize);
    if(stTrack.FSize == 0) stTrack.FSize = stTrack_default.FSize; stTrack_default.FSize = stTrack.FSize;
    terminal_tools::parse_argument (argc, argv, "-trkcf", stTrack.FClr);
    if(stTrack.FClr == 0) stTrack.FClr = stTrack_default.FClr; stTrack_default.FClr = stTrack.FClr;
    terminal_tools::parse_argument (argc, argv, "-trkmf", stTrack.MFac);
    if(stTrack.MFac == 0) stTrack.MFac = stTrack_default.MFac; stTrack_default.MFac = stTrack.MFac;

    terminal_tools::parse_argument (argc, argv, "-trkcm", stTrack.CntMem);
    if(stTrack.CntMem == 0) stTrack.CntMem = stTrack_default.CntMem; stTrack_default.CntMem = stTrack.CntMem;
    terminal_tools::parse_argument (argc, argv, "-trkcs", stTrack.CntStable);
    if(stTrack.CntStable == 0) stTrack.CntStable = stTrack_default.CntStable; stTrack_default.CntStable = stTrack.CntStable;
    terminal_tools::parse_argument (argc, argv, "-trkcd", stTrack.CntDisap);
    if(stTrack.CntDisap == 0) stTrack.CntDisap = stTrack_default.CntDisap; stTrack_default.CntDisap = stTrack.CntDisap;

    terminal_tools::parse_argument (argc, argv, "-natttd", nAttTDMode);
    if(nAttTDMode == 0) nAttTDMode = nAttTDMode_default; nAttTDMode_default = nAttTDMode;
    terminal_tools::parse_argument (argc, argv, "-candmax", nAttSizeMax);
    if(nAttSizeMax == 0) nAttSizeMax = nAttSizeMax_default; nAttSizeMax_default = nAttSizeMax;
    terminal_tools::parse_argument (argc, argv, "-candmin", nAttSizeMin);
    if(nAttSizeMin == 0) nAttSizeMin = nAttSizeMin_default; nAttSizeMin_default = nAttSizeMin;
    terminal_tools::parse_argument (argc, argv, "-candpm", nAttPtsMin);
    if(nAttPtsMin == 0) nAttPtsMin = nAttPtsMin_default; nAttPtsMin_default = nAttPtsMin;
    terminal_tools::parse_argument (argc, argv, "-candar1", nAttAspect1);
    if(nAttAspect1 == 0) nAttAspect1 = nAttAspect1_default; nAttAspect1_default = nAttAspect1;
    terminal_tools::parse_argument (argc, argv, "-candar2", nAttAspect2);
    if(nAttAspect2 == 0) nAttAspect2 = nAttAspect2_default; nAttAspect2_default = nAttAspect2;


    terminal_tools::parse_argument (argc, argv, "-siftsc", nSiftScales);
    if(nSiftScales == 0) nSiftScales = nSiftScales_default; nSiftScales_default = nSiftScales;
    terminal_tools::parse_argument (argc, argv, "-siftis", nSiftInitSigma);
    if(nSiftInitSigma == 0) nSiftInitSigma = nSiftInitSigma_default; nSiftInitSigma_default = nSiftInitSigma;
    terminal_tools::parse_argument (argc, argv, "-siftpt", nSiftPeakThrs);
    if(nSiftPeakThrs == 0) nSiftPeakThrs = nSiftPeakThrs_default; nSiftPeakThrs_default = nSiftPeakThrs;

    terminal_tools::parse_argument (argc, argv, "-flknn", nFlannKnn);
    if(nFlannKnn == 0) nFlannKnn = nFlannKnn_default; nFlannKnn_default = nFlannKnn;
    terminal_tools::parse_argument (argc, argv, "-fltp",nFlannTargetPrecision);
    if(nFlannTargetPrecision == 0) nFlannTargetPrecision = nFlannTargetPrecision_default; nFlannTargetPrecision_default = nFlannTargetPrecision;
    terminal_tools::parse_argument (argc, argv, "-flmf", nFlannMatchFac);
    if(nFlannMatchFac == 0) nFlannMatchFac = nFlannMatchFac_default; nFlannMatchFac_default = nFlannMatchFac;
    terminal_tools::parse_argument (argc, argv, "-flmc", nFlannMatchCnt);
    if(nFlannMatchCnt == 0) nFlannMatchCnt = nFlannMatchCnt_default; nFlannMatchCnt_default = nFlannMatchCnt;

    terminal_tools::parse_argument (argc, argv, "-recfeat", nRecogFeature);
    if(nRecogFeature == 0) nRecogFeature = nRecogFeature_default; nRecogFeature_default = nRecogFeature;
    terminal_tools::parse_argument (argc, argv, "-recdc", nRecogDClr);
    if(nRecogDClr == 0) nRecogDClr = nRecogDClr_default; nRecogDClr_default = nRecogDClr;
    terminal_tools::parse_argument (argc, argv, "-reccmm", bRecogClrMask);
    if(bRecogClrMask == 0) bRecogClrMask = bRecogClrMask_default; bRecogClrMask_default = bRecogClrMask;
    terminal_tools::parse_argument (argc, argv, "-recvmd", nRecogMaskDispMode);
    if(nRecogMaskDispMode == 0) nRecogMaskDispMode = nRecogMaskDispMode_default; nRecogMaskDispMode_default = nRecogMaskDispMode;


    terminal_tools::parse_argument (argc, argv, "-recordmod", nRecordMode);
    if(nRecordMode == 0) nRecordMode = nRecordMode_default; nRecordMode_default = nRecordMode;
    terminal_tools::parse_argument (argc, argv, "-tmessfl", nTimeMessFrmLimit);
    if(nTimeMessFrmLimit == 0) nTimeMessFrmLimit = nTimeMessFrmLimit_default; nTimeMessFrmLimit_default = nTimeMessFrmLimit;
    terminal_tools::parse_argument (argc, argv, "-snapfm", nSnapFormat);
    if(nSnapFormat == 0) nSnapFormat = nSnapFormat_default; nSnapFormat_default = nSnapFormat;


    terminal_tools::parse_argument (argc, argv, "-siftlibfile", sSIFTLibFileName);
    terminal_tools::parse_argument (argc, argv, "-colorlibfile", sColorLibFileName);
    terminal_tools::parse_argument (argc, argv, "-libpath", sLibFilePath);

    terminal_tools::parse_argument (argc, argv, "-target", sTarget);
    terminal_tools::parse_argument (argc, argv, "-targetpath", sTargetImgPath);
    terminal_tools::parse_argument (argc, argv, "-datadir", sDataDir);



    // Sahils parameters
    terminal_tools::parse_argument (argc, argv, "-deltabin", nDeltaBinNo);
    if(nDeltaBinNo == 0) nDeltaBinNo = 12;
    terminal_tools::parse_argument (argc, argv, "-tnumb", T_numb);
    if(T_numb == 0) T_numb = 6;
    terminal_tools::parse_argument (argc, argv, "-torient", T_orient);
    if(T_orient == 0) T_orient = 0.1;
    terminal_tools::parse_argument (argc, argv, "-tscale", T_scale);
    if(T_scale == 0) T_scale = 0.001;




    printf ("\n");
    printf ("==================================================================\n");
    printf ("** Parameter settings **\n");
    printf ("==================================================================\n");

    printf ("== Pre Process Parameter ==\n");
    if (bDepthDispMode)     printf ("Depth display mode: .............................     blue-close\n");
    else                printf ("Depth display mode: .............................     red-close\n");
    printf ("Depth limit: .................................... %8.2f mm\n", nDLimit);
    printf ("\n");

    printf ("== Segmentation Parameter ==\n");
    printf ("Segmentation - Distance Threshold: .............. %5.0f mm\n", nDSegmDThres*1000);
    printf ("Segmentation - Distance gradient threshold: ..... %7.1f mm/pixel\n", nDSegmGradDist*1000);
    printf ("Segmentation - Cut out small segments: .......... %5d pixels\n", nDSegmCutSize);
    printf ("Tracking - Position displacement threshold: ..... %8.2f\n", stTrack.DPos);
    printf ("Tracking - Size difference threshold: ........... %8.2f %%\n", stTrack.DSize*100);
    printf ("Tracking - Color difference threshold: .......... %8.2f\n", stTrack.DClr);
    printf ("Tracking - Position displacement factor: ........ %8.2f\n", stTrack.FPos);
    printf ("Tracking - Size difference factor: .............. %8.2f\n", stTrack.FSize);
    printf ("Tracking - Color difference factor: ............. %8.2f\n", stTrack.FClr);
    printf ("Tracking - Total difference threshold: .......... %8.2f\n", stTrack.Dist);
    printf ("Tracking - Stablity threshold: .................. %5d\n", stTrack.CntStable);
    printf ("\n");

    printf ("== Recognition Parameter ==\n");
    switch (nRecogFeature) {
    case 1: printf ("Feature mode: ...................................     RGB Color Histogram\n"); break;
    case 2: printf ("Feature mode: ...................................     No feature Calculation-Opponent Color Space\n"); break;
    case 3: printf ("Feature mode: ...................................     Transformed Color Space Histogram\n"); break;
    case 10: case 20:
    printf ("Feature mode: ................................... SIFT\n");
    printf ("2D SIFT - Scales: ............................... %5d\n", nSiftScales);
    printf ("2D SIFT - Initial sigma: ........................ %10.4f\n", nSiftInitSigma);
    printf ("2D SIFT - Peak threshold: ....................... %10.4f\n", nSiftPeakThrs); break;
    case 11:
    printf ("Feature mode: ................................... SIFT (Hue+Gray)\n");
    printf ("2D SIFT - Scales: ............................... %5d\n", nSiftScales);
    printf ("2D SIFT - Initial sigma: ........................ %10.4f\n", nSiftInitSigma);
    printf ("2D SIFT - Peak threshold: ....................... %10.4f\n", nSiftPeakThrs); break;
    }

    printf ("==================================================================\n\n\n");
}




/* Resets all the segmentation and recognition parameter to their default value
 */
void ResetParameter () {
    bDepthDispMode = bDepthDispMode_default;
    nDLimit = nDLimit_default;

    nDSegmDThres = nDSegmDThres_default;
    nDGradConst = nDGradConst_default;
    nDGradFilterMode = nDGradFilterMode_default;
    nDGradFilterSize = nDGradFilterSize_default;
    nDGradSmthMode = nDGradSmthMode_default;
    nDGradSmthCenter = nDGradSmthCenter_default;
    nDGradSmthBThres1 = nDGradSmthBThres1_default;
    nDGradSmthBThres2 = nDGradSmthBThres2_default;
    nDGradSmthBnd1 = nDGradSmthBThres1*254/nDSegmDThres/2; nDGradSmthBnd2 = nDGradSmthBThres2*254/nDSegmDThres/2;
    nDGradSmthFac = nDGradSmthFac_default;

    nDSegmSizeThres = nDSegmSizeThres_default;
    nDSegmGradDist = nDSegmGradDist_default;
    nDSegmCutSize = nDSegmCutSize_default;

    nGSegmSigma = nGSegmSigma_default;
    nGSegmGrThrs = nGSegmGrThrs_default;
    nGSegmMinSize = nGSegmMinSize_default;


    stTrack.Mode = stTrack_default.Mode;
    stTrack.ClrMode = stTrack_default.ClrMode;
    //stTrack.HistoBin = stTrack_default.HistoBin;
    stTrack.DPos = stTrack_default.DPos;
    stTrack.DSize = stTrack_default.DSize;
    stTrack.DClr = stTrack_default.DClr;
    stTrack.Dist = stTrack_default.Dist;
    stTrack.FPos = stTrack_default.FPos;
    stTrack.FSize = stTrack_default.FSize;
    stTrack.FClr = stTrack_default.FClr;
    stTrack.MFac = stTrack_default.MFac;
    stTrack.CntMem = stTrack_default.CntMem;
    stTrack.CntStable = stTrack_default.CntStable;
    stTrack.CntDisap = stTrack_default.CntDisap;


    nAttSizeMax = nAttSizeMax_default;
    nAttSizeMin = nAttSizeMin_default;
    nAttPtsMin = nAttPtsMin_default;
    nAttAspect1 = nAttAspect1_default;
    nAttAspect2 = nAttAspect2_default;

    nSiftScales = nSiftScales_default;
    nSiftInitSigma = nSiftInitSigma_default;
    nSiftPeakThrs = nSiftPeakThrs_default;

    nFlannKnn = nFlannKnn_default;
    nFlannMatchFac = nFlannMatchFac_default;
    nFlannMatchCnt = nFlannMatchCnt_default;

    nAttTDMode = nAttTDMode_default;
    nRecogFeature = nRecogFeature_default;
    nRecogDClr = nRecogDClr_default;
    nRecogMaskDispMode = nRecogMaskDispMode_default;

    nRecordMode = nRecordMode_default;
    nTimeMessFrmLimit = nTimeMessFrmLimit_default;
    nSnapFormat = nSnapFormat_default;
    flag_pcd = 0;

    if (vbFlagWnd[stTID.nPrmSegm]) {
        cvSetTrackbarPos(vsTrackbarName[20].data(), vsWndName[stTID.nPrmSegm].data(), stTrack.Mode);
        //cvSetTrackbarPos(vsTrackbarName[21].data(), vsWndName[stTID.nPrmSegm].data(), stTrack.ClrMode);
        cvSetTrackbarPos(vsTrackbarName[22].data(), vsWndName[stTID.nPrmSegm].data(), stTrack.DPos*100);
        cvSetTrackbarPos(vsTrackbarName[23].data(), vsWndName[stTID.nPrmSegm].data(), stTrack.DSize*100);
        cvSetTrackbarPos(vsTrackbarName[24].data(), vsWndName[stTID.nPrmSegm].data(), stTrack.DClr*100);
        cvSetTrackbarPos(vsTrackbarName[25].data(), vsWndName[stTID.nPrmSegm].data(), stTrack.FPos*100);
        cvSetTrackbarPos(vsTrackbarName[26].data(), vsWndName[stTID.nPrmSegm].data(), stTrack.FSize*100);
        cvSetTrackbarPos(vsTrackbarName[27].data(), vsWndName[stTID.nPrmSegm].data(), stTrack.FClr*100);
        cvSetTrackbarPos(vsTrackbarName[28].data(), vsWndName[stTID.nPrmSegm].data(), stTrack.Dist*100);

        //cvSetTrackbarPos(vsTrackbarName[38].data(), vsWndName[stTID.nPrmSegm].data(), stTrack.CntStable);
        //cvSetTrackbarPos(vsTrackbarName[39].data(), vsWndName[stTID.nPrmSegm].data(), stTrack.CntDisap);
        cvSetTrackbarPos(vsTrackbarName[31].data(), vsWndName[stTID.nPrmSegm].data(), nAttSizeMax);
        cvSetTrackbarPos(vsTrackbarName[32].data(), vsWndName[stTID.nPrmSegm].data(), nAttSizeMin);
        cvSetTrackbarPos(vsTrackbarName[33].data(), vsWndName[stTID.nPrmSegm].data(), nAttPtsMin);
        //cvSetTrackbarPos(vsTrackbarName[34].data(), vsWndName[stTID.nPrmSegm].data(), nAttAspect1);
        //cvSetTrackbarPos(vsTrackbarName[35].data(), vsWndName[stTID.nPrmSegm].data(), nAttAspect2);
    }
    if (vbFlagWnd[stTID.nPrmRecog]) {
        cvSetTrackbarPos(vsTrackbarName[1].data(), vsWndName[stTID.nPrmRecog].data(), nSnapFormat);
        cvSetTrackbarPos(vsTrackbarName[99].data(), vsWndName[stTID.nPrmRecog].data(), nAttTDMode);
        cvSetTrackbarPos(vsTrackbarName[36].data(), vsWndName[stTID.nPrmRecog].data(), nRecogDClr*100);
        cvSetTrackbarPos(vsTrackbarName[41].data(), vsWndName[stTID.nPrmRecog].data(), nSiftScales);
        cvSetTrackbarPos(vsTrackbarName[42].data(), vsWndName[stTID.nPrmRecog].data(), nSiftInitSigma*10);
        cvSetTrackbarPos(vsTrackbarName[43].data(), vsWndName[stTID.nPrmRecog].data(), nSiftPeakThrs*1000);
        cvSetTrackbarPos(vsTrackbarName[51].data(), vsWndName[stTID.nPrmRecog].data(), nFlannKnn);
        cvSetTrackbarPos(vsTrackbarName[52].data(), vsWndName[stTID.nPrmRecog].data(), nFlannMatchFac*100);
        cvSetTrackbarPos(vsTrackbarName[53].data(), vsWndName[stTID.nPrmRecog].data(), nFlannMatchCnt);
        cvSetTrackbarPos(vsTrackbarName[54].data(), vsWndName[stTID.nPrmRecog].data(), 0);
    }
    if (vbFlagWnd[stTID.nDepth]) {
        cvSetTrackbarPos(vsTrackbarName[81].data(), vsWndName[stTID.nDepth].data(), nDGradSmthCenter);
        cvSetTrackbarPos(vsTrackbarName[82].data(), vsWndName[stTID.nDepth].data(), nDGradSmthBnd1);
        cvSetTrackbarPos(vsTrackbarName[83].data(), vsWndName[stTID.nDepth].data(), nDGradSmthBnd2);
        //cvSetTrackbarPos(vsTrackbarName[84].data(), vsWndName[stTID.nDepth].data(), nDGradConst*100);
    }
    if (vbFlagWnd[stTID.nGSegm]) {
        cvSetTrackbarPos(vsTrackbarName[17].data(), vsWndName[stTID.nGSegm].data(), nGSegmSigma*10);
        cvSetTrackbarPos(vsTrackbarName[18].data(), vsWndName[stTID.nGSegm].data(), nGSegmGrThrs);
        cvSetTrackbarPos(vsTrackbarName[19].data(), vsWndName[stTID.nGSegm].data(), nGSegmMinSize);
    }
}



/* Initializing names for windows, buttons, trackbars. Assigning task IDs
 */
void InitVariables () {
    cv::namedWindow(sTitle); cvMoveWindow(sTitle.data(), 80, 20);

    stTID.nRgbOrg = 0, stTID.nRgbDs = 1, stTID.nDepth = 2, stTID.nInfo = 3, stTID.nRecVideo = 7, stTID.nSnap = 8;
    stTID.nSegmentation = 10, stTID.nDSegm = 12, stTID.nGSegm = 13, stTID.nTrack = 15, stTID.nProto = 16;
    stTID.nRecognition = 20, stTID.nRecogOrg = 21, stTID.nRecogDs = 22, stTID.nSIFT = 23;
    stTID.nRecTime = 31, stTID.nRstTime = 33, stTID.nPrmInfo = 34, stTID.nPrmSett = 35, stTID.nPrmSegm = 36, stTID.nPrmRecog = 37, stTID.nRstPrm = 38;

    vsWndName[stTID.nRgbOrg] = "Original RGB";
    vsWndName[stTID.nRgbDs] = "Downsampled RGB";
    vsWndName[stTID.nDepth] = "Depth Image";
    vsWndName[stTID.nGSegm] = "Graph-based Segments";
    vsWndName[stTID.nDSegm] = "Segments";
    vsWndName[stTID.nTrack] = "Tracked Object";
    vsWndName[stTID.nProto] = "Object";
    vsWndName[stTID.nRecogOrg] = "Recognition";
    vsWndName[stTID.nRecogDs] = "Recognition down-sampled";
    vsWndName[stTID.nSIFT] = "SIFT on whole Image";
    vsWndName[stTID.nPrmSett] = "System Setting";
    vsWndName[stTID.nPrmSegm] = "Segmentation Setting";
    vsWndName[stTID.nPrmRecog] = "Recognition Setting";

    vsBtnName[stTID.nRgbOrg] = "Org";
    vsBtnName[stTID.nRgbDs] = "DS";
    vsBtnName[stTID.nDepth] = "Depth";
    vsBtnName[stTID.nInfo] = "Info";
    vsBtnName[stTID.nSegmentation] = "Segmentation";
    vsBtnName[stTID.nGSegm] = "GB Seg";
    vsBtnName[stTID.nDSegm] = "Segmts";
    vsBtnName[stTID.nTrack] = "Track";
    vsBtnName[stTID.nProto] = "Objects";
    vsBtnName[stTID.nRecognition] = "Recognition";
    vsBtnName[stTID.nRecogOrg] = "Rec Org";
    vsBtnName[stTID.nRecogDs] = "Rec DS";
    vsBtnName[stTID.nSIFT] = "SIFT W";
    vsBtnName[stTID.nRecVideo] = "REC";
    vsBtnName[stTID.nSnap] = "Snapshot";
    vsBtnName[stTID.nRecTime] = "Rec Time";
    vsBtnName[stTID.nRstTime] = "Rst Time";
    vsBtnName[stTID.nPrmInfo] = "Info";
    vsBtnName[stTID.nPrmSegm] = "Segm Prm";
    vsBtnName[stTID.nPrmRecog] = "Recog Prm";
    vsBtnName[stTID.nRstPrm] = "Reset Prm";
    vsBtnName[nTaskNrMax-1] = "Abandon Ship!";

    vbFlagTask[stTID.nInfo] = true;
    vbFlagTask[stTID.nPrmInfo] = true;

    int nWndX = 300, nWndY = 50;
    for (int i = 0; i < nTaskNrMax; i++) {if (int(strlen(vsWndName[i].data()))) {nWndX += 50; mWndPos[i][0] = nWndX; mWndPos[i][1] = nWndY;}}

    for (int i = 0; i < nTaskNrMax; i++)
        if (vbFlagWnd[i]) {cv::namedWindow(vsWndName[i]); cvMoveWindow(vsWndName[i].data(), mWndPos[i][0], mWndPos[i][1]);}


    vsTrackbarName[1]  = "Snptshot format (0-2)                 ";
    vsTrackbarName[2]  = "Depth limit (0.5-10 m)                ";
    vsTrackbarName[3]  = "Filter size (3-27 pixels)               ";
    vsTrackbarName[11] = "AAA            ";
    vsTrackbarName[12] = "BBB";
    vsTrackbarName[20] = "Tracking mode                                ";
    vsTrackbarName[21] = "Color mode (0-4)                          ";
    vsTrackbarName[22] = "Pos. dist. limit (0-1)       ";
    vsTrackbarName[23] = "Size dist. limit (0-1)       ";
    vsTrackbarName[24] = "Color dist. limit (0-1)    ";
    vsTrackbarName[25] = "Pos. dist factor (0-1)    ";
    vsTrackbarName[26] = "Size factor (0-1)              ";
    vsTrackbarName[27] = "Color dist. factor (0-1)";
    vsTrackbarName[28] = "Total dist. limit (0-5)    ";
    vsTrackbarName[38] = "Stable counter (0-5 frames) " ;
    vsTrackbarName[39] = "Vanish counter (0-2 frames) " ;
    vsTrackbarName[31] = "Objs upper limit (0-1000 mm)";
    vsTrackbarName[32] = "Objs lower limit (0-1000 mm)";
    vsTrackbarName[33] = "Objs minimum (0-1000 pixels)";
    vsTrackbarName[34] = "Objs aspect1 (0-100%)           ";
    vsTrackbarName[35] = "Objs aspect2 (0-100%)           ";
    vsTrackbarName[36] = "Color dist. thres. (0-1)               ";
    vsTrackbarName[41] = "SIFT scales (1-5)                             ";
    vsTrackbarName[42] = "SIFT init. sigma (1-2)                   ";
    vsTrackbarName[43] = "SIFT peak thres. (0-1)                  ";
    vsTrackbarName[51] = "Flann KNN (1-3)                             ";
    vsTrackbarName[52] = "Flann match fac. (0-1)                ";
    vsTrackbarName[53] = "Flann match cnt (0-100)            ";
    vsTrackbarName[54] = "Print color distance                   ";
    vsTrackbarName[17] = "GB segm. sigma (0-1)     ";
    vsTrackbarName[18] = "GB segm. gr (0-500)        ";
    vsTrackbarName[19] = "GB segm. min (0-1000)  ";

    vsTrackbarName[99] = "Selection Mode                ";

    vsTrackbarName[81] = "DG Gray Center                ";
    vsTrackbarName[82] = "DG Bandwidth1, tau1           ";
    vsTrackbarName[83] = "DG Bandwidth2, tau2           ";
    vsTrackbarName[84] = "DG adjusting Const. C         ";
}




/* Stores the position of the individual rectangles which represent a button in a matrix
 */
void SetBtnPos (int nTaskNr, int nPadSecX, int nPadSecY, int nBtnW, int nBtnH, std::vector<std::vector<int> >& mnBtnPos)
{
    mnBtnPos[nTaskNr][0] = nPadSecX;
    mnBtnPos[nTaskNr][1] = nPadSecY;
    mnBtnPos[nTaskNr][2] = nBtnW;
    mnBtnPos[nTaskNr][3] = nBtnH;
}



/* Determines the position of the individual button in the main ni-vision window
 *
 * Button are represented by rectangles. The coordinates are assigned to each button and stored. If a
 * mouse event happens a mouse listener can then evaluate in which rectangle the mouse was located
 * when the button was pushed.
 */
void SetPad(int nBtnSize, std::vector<std::vector<int> >& mnBtnPos, int &row1, int &col1, int &col2, int &height)
{
    int nTaskNr;
    int nBtnOffset = 5, nBtnW, nBtnH, nPadSecX = 0, nPadSecY = 0;
    col1 = 2*nBtnSize + 1;
    col2 = col1 + 3*nBtnSize - nBtnOffset;

    nBtnW = nBtnSize - 2*nBtnOffset;
    nBtnH = nBtnSize - 8*nBtnOffset;
    nPadSecX = nBtnOffset;
    nPadSecY = nBtnOffset*4;
    nTaskNr = stTID.nRgbOrg; SetBtnPos(nTaskNr, nPadSecX, nPadSecY, nBtnW, nBtnH, mnBtnPos);
    nPadSecX += nBtnW + 2*nBtnOffset;
    nTaskNr = stTID.nRgbDs;
    SetBtnPos(nTaskNr, nPadSecX, nPadSecY, nBtnW, nBtnH, mnBtnPos);
    nPadSecY += nBtnH + 2*nBtnOffset;

    nPadSecX = nBtnOffset;
    nPadSecY += 3*nBtnOffset;
    nBtnW = 2*nBtnSize - 2*nBtnOffset;
    nBtnH = nBtnSize - 2*nBtnOffset;
    nTaskNr = stTID.nSegmentation; SetBtnPos(nTaskNr, nPadSecX, nPadSecY, nBtnW, nBtnH, mnBtnPos);
    nPadSecY += nBtnH + 2*nBtnOffset;

    nBtnOffset = 10;
    nBtnH = nBtnSize - 4*nBtnOffset;
    nBtnW = nBtnSize - 1.5*nBtnOffset;
    nPadSecX = nBtnOffset;
    nTaskNr = stTID.nDSegm; SetBtnPos(nTaskNr, nPadSecX, nPadSecY, nBtnW, nBtnH, mnBtnPos);
    nPadSecX += nBtnW + nBtnOffset;
    nTaskNr = stTID.nTrack; SetBtnPos(nTaskNr, nPadSecX, nPadSecY, nBtnW, nBtnH, mnBtnPos);
    nPadSecY += nBtnH + nBtnOffset;
    nPadSecX = nBtnOffset;
    nTaskNr = stTID.nProto; SetBtnPos(nTaskNr, nPadSecX, nPadSecY, nBtnW, nBtnH, mnBtnPos);
    nPadSecX += nBtnW + nBtnOffset;
    nTaskNr = stTID.nDepth; SetBtnPos(nTaskNr, nPadSecX, nPadSecY, nBtnW, nBtnH, mnBtnPos);
    nPadSecY += nBtnH + nBtnOffset;
    nPadSecX = nBtnOffset;
    nTaskNr = stTID.nGSegm; SetBtnPos(nTaskNr, nPadSecX, nPadSecY, nBtnW, nBtnH, mnBtnPos);
    nPadSecY += nBtnH + 3*nBtnOffset;


    nBtnOffset = 5;
    nBtnW = 2*nBtnSize - 2*nBtnOffset;
    nBtnH = nBtnSize - 2*nBtnOffset;
    nPadSecX = nBtnOffset;
    nTaskNr = stTID.nRecognition; SetBtnPos(nTaskNr, nPadSecX, nPadSecY, nBtnW, nBtnH, mnBtnPos);
    nPadSecY += nBtnH + 2*nBtnOffset;

    nBtnOffset = 10;
    nBtnH = nBtnSize - 4*nBtnOffset;
    nBtnW = nBtnSize - 1.5*nBtnOffset;
    nPadSecX = nBtnOffset;
    nTaskNr = stTID.nRecogOrg; SetBtnPos(nTaskNr, nPadSecX, nPadSecY, nBtnW, nBtnH, mnBtnPos);
    nPadSecX += nBtnW + nBtnOffset;
    nTaskNr = stTID.nRecogDs; SetBtnPos(nTaskNr, nPadSecX, nPadSecY, nBtnW, nBtnH, mnBtnPos);
    nPadSecY += nBtnH + nBtnOffset;
    nPadSecX = nBtnOffset;
    nTaskNr = stTID.nSIFT; SetBtnPos(nTaskNr, nPadSecX, nPadSecY, nBtnW, nBtnH, mnBtnPos);
    nPadSecY += nBtnH + 4*nBtnOffset;

    row1 = nPadSecY + 15;

    nPadSecX = nBtnOffset;
    nPadSecY = row1 + nBtnOffset;
    nTaskNr = stTID.nRecVideo; SetBtnPos(nTaskNr, nPadSecX, nPadSecY, nBtnW, nBtnH, mnBtnPos);
    nPadSecX += nBtnW + nBtnOffset;
    nTaskNr = stTID.nSnap; SetBtnPos(nTaskNr, nPadSecX, nPadSecY, nBtnW, nBtnH, mnBtnPos);
    nPadSecY += nBtnH + nBtnOffset;

    nPadSecX = nBtnOffset;
    nBtnW = 2*nBtnSize - 2*nBtnOffset;
    nBtnH = nBtnSize - 2*nBtnOffset;
    nTaskNr = nTaskNrMax-1; SetBtnPos(nTaskNr, nPadSecX, nPadSecY, nBtnW, nBtnH, mnBtnPos);
    nPadSecY += nBtnH + nBtnOffset;

    height = nPadSecY;


    nBtnH = nBtnSize - 4*nBtnOffset;
    nBtnW = nBtnSize - 1.5*nBtnOffset;
    nPadSecY = row1 + nBtnOffset;
    nPadSecX = col1 + nBtnOffset;
    nTaskNr = stTID.nInfo; SetBtnPos(nTaskNr, nPadSecX, nPadSecY, nBtnW, nBtnH, mnBtnPos);
    nPadSecX += nBtnW + nBtnOffset;
    nTaskNr = stTID.nRecTime; SetBtnPos(nTaskNr, nPadSecX, nPadSecY, nBtnW, nBtnH, mnBtnPos);
    nPadSecX += nBtnW + nBtnOffset;
    nTaskNr = stTID.nRstTime; SetBtnPos(nTaskNr, nPadSecX, nPadSecY, nBtnW, nBtnH, mnBtnPos);

    nPadSecY = row1 + nBtnOffset;
    nPadSecX = col2 + nBtnOffset;
    nTaskNr = stTID.nPrmInfo; SetBtnPos(nTaskNr, nPadSecX, nPadSecY, nBtnW, nBtnH, mnBtnPos);
    nPadSecX += nBtnW + nBtnOffset;
    nBtnW = nBtnSize;
    nTaskNr = stTID.nPrmSegm; SetBtnPos(nTaskNr, nPadSecX, nPadSecY, nBtnW, nBtnH, mnBtnPos);
    nPadSecX += nBtnW + nBtnOffset;
    nTaskNr = stTID.nPrmRecog; SetBtnPos(nTaskNr, nPadSecX, nPadSecY, nBtnW, nBtnH, mnBtnPos);
    nPadSecX += nBtnW + nBtnOffset;
    nTaskNr = stTID.nRstPrm; SetBtnPos(nTaskNr, nPadSecX, nPadSecY, nBtnW, nBtnH, mnBtnPos);
}




/* Draws the gui.
 */
void DrawPad(cv::Mat &cvm_input, std::vector<int> vnBtnProp, int nPadRow1, int nPadCol1, int nPadCol2)
{
    int x1 = 0, y1 = 0, x2 = 0, y2 = 0;
    int thickness;

    int nNrSegmentation = 10, nNrRecSegm = 19;
    int nNrRecognition = 20, nNrRecRecog = 29;
    int nNrRecVideo = 7;

    cv::Scalar c_color = c_gray;
    cv::line(cvm_input, cv::Point(0, nPadRow1), cv::Point(cvm_input.cols, nPadRow1), c_color, 1);
    cv::line(cvm_input, cv::Point(0, nPadRow1), cv::Point(cvm_input.cols, nPadRow1), c_color, 1);
    cv::line(cvm_input, cv::Point(nPadCol1, 0), cv::Point(nPadCol1, cvm_input.rows), c_color, 1);
    cv::line(cvm_input, cv::Point(nPadCol2, 0), cv::Point(nPadCol2, cvm_input.rows), c_color, 1);

    int i;
    for (i = 0; i < nTaskNrMax; i++){
        if (!mnBtnPos[i][2]) continue;
        thickness = 1;

        if (vbFlagTask[i]) {
            if(vbFlagWnd[i]) c_color = c_lightgreen;
            else c_color = c_white;
            if (vnBtnProp[i] == 20) c_color = c_lemon;
            if ((i == nNrRecSegm && vbFlagTask[nNrSegmentation]) || (i == nNrRecRecog && vbFlagTask[nNrRecognition]) || (i == nNrRecVideo)) c_color = c_red;
        }
        else c_color = c_gray;

        if (i == nTaskNrMax - 1) c_color = c_cyan;

        x1 = mnBtnPos[i][0]; y1 = mnBtnPos[i][1]; x2 = x1 + mnBtnPos[i][2]; y2 = y1 + mnBtnPos[i][3];
        cv::rectangle(cvm_input, cv::Point(x1, y1), cv::Point(x2 ,y2), c_color, thickness);
        cv::rectangle(cvm_input, cv::Point(x1, y1), cv::Point(x2 ,y2), c_color, thickness);

        if (vnBtnProp[i] == 20) {
            thickness = CV_FILLED;
            cv::rectangle(cvm_input, cv::Point(x1+3, y1+3), cv::Point(x2-3 ,y2-3), c_color, thickness);
            c_color = c_black;
        }

        cv::putText(cvm_input, vsBtnName[i], cv::Point(x1 + (mnBtnPos[i][2] - strlen(vsBtnName[i].data())*18*nBtnFontSize)/2, y1 + mnBtnPos[i][3]/2 + 10*nBtnFontSize), nFont, nBtnFontSize, c_color, 1);
        if ((i == nNrSegmentation && vbFlagTask[nNrSegmentation] && vbFlagTask[nNrRecSegm]) || (i == nNrRecognition && vbFlagTask[nNrRecognition] && vbFlagTask[nNrRecRecog]) || (i == nNrRecVideo && vbFlagTask[nNrRecVideo])) {
            cv::putText(cvm_input, vsBtnName[i], cv::Point(x1 + (mnBtnPos[i][2] - strlen(vsBtnName[i].data())*18*nBtnFontSize)/2, y1 + mnBtnPos[i][3]/2 + 10*nBtnFontSize), nFont, nBtnFontSize, c_red, 1);
        }
    }
}




/* Draws information about time measurements on the left side of the gui.
 */
void DrawInfo(cv::Mat &cvm_input, int nSegNr, int nPadRow1, int nPadCol1, int nCntFrame_tmp, double nTimeTotal,
              double nTimePre, double nTimeDepth, double nTimeBlur, double nTimePre_avr, double nTimeDepth_avr, double nTimeBlur_avr,
              double nTimeSegm, double nTimeTrack, double nTimeAtt, double nTimeRec, double nTimeSift, double nTimeFlann,
              double nTimeSegm_avr, double nTimeTrack_avr, double nTimeAtt_avr, double nTimeRec_avr, double nTimeSift_avr, double nTimeFlann_avr,
              double nTimeRecCycle, int nFoundCnt, double nTimeRecFound, double nTimeRecCycle_avr, double nTimeRecFound_avr,
              double nTimeGbSegm, double nTimeSiftWhole, double nTimeSiftDrawWhole, double nTimeFlannWhole, double nTimeGbSegm_avr, double nTimeSiftWhole_avr, double nTimeFlannWhole_avr)
{
    char sText[128];
    int nPosY = 25, nPosX = nPadCol1 + 10, nTab = nPosX + 140;

    sprintf(sText, "Elapsed time in ms"); cv::putText(cvm_input, sText, cv::Point(nPosX, nPosY), nFont, nFontSize, c_cyan, 1);
    sprintf(sText, "Segmentation:"); cv::putText(cvm_input, sText, cv::Point(nPosX, (nPosY+=15)), nFont, nFontSize, c_green, 1);
    sprintf(sText, "%8.2f", nTimeSegm); cv::putText(cvm_input, sText, cv::Point(nTab, nPosY), nFont, nFontSize, c_green, 1);
    sprintf(sText, "Tracking:"); cv::putText(cvm_input, sText, cv::Point(nPosX,  (nPosY+=15)), nFont, nFontSize, c_green, 1);
    sprintf(sText, "%8.2f", nTimeTrack); cv::putText(cvm_input, sText, cv::Point(nTab, nPosY), nFont, nFontSize, c_green, 1);
    sprintf(sText, "Selective attention:"); cv::putText(cvm_input, sText, cv::Point(nPosX,  (nPosY+=15)), nFont, nFontSize, c_green, 1);
    sprintf(sText, "%8.2f", nTimeAtt); cv::putText(cvm_input, sText, cv::Point(nTab, nPosY), nFont, nFontSize, c_green, 1);
    sprintf(sText, "Feature extraction:"); cv::putText(cvm_input, sText, cv::Point(nPosX, (nPosY+=15)), nFont, nFontSize, c_green, 1);
    sprintf(sText, "%8.2f", nTimeSift); cv::putText(cvm_input, sText, cv::Point(nTab, nPosY), nFont, nFontSize, c_green, 1);
    sprintf(sText, "Flann searching:"); cv::putText(cvm_input, sText, cv::Point(nPosX, (nPosY+=15)), nFont, nFontSize, c_green, 1);
    sprintf(sText, "%8.2f", nTimeFlann); cv::putText(cvm_input, sText, cv::Point(nTab, nPosY), nFont, nFontSize, c_green, 1);
    sprintf(sText, "Recognition:"); cv::putText(cvm_input, sText, cv::Point(nPosX, (nPosY+=15)), nFont, nFontSize, c_darkgreen, 1);
    sprintf(sText, "%8.2f", nTimeRec); cv::putText(cvm_input, sText, cv::Point(nTab, nPosY), nFont, nFontSize, c_darkgreen, 1);

    sprintf(sText, "Graph based Segm:"); cv::putText(cvm_input, sText, cv::Point(nPosX, (nPosY+=25)), nFont, nFontSize, c_darkgreen, 1);
    sprintf(sText, "%8.2f", nTimeGbSegm); cv::putText(cvm_input, sText, cv::Point(nTab, nPosY), nFont, nFontSize, c_darkgreen, 1);
    sprintf(sText, "SIFT on whole image: "); cv::putText(cvm_input, sText, cv::Point(nPosX, (nPosY+=15)), nFont, nFontSize, c_darkgreen, 1);
    sprintf(sText, "%8.2f %2.0f", nTimeSiftWhole, nTimeSiftDrawWhole); cv::putText(cvm_input, sText, cv::Point(nTab, nPosY), nFont, nFontSize, c_darkgreen, 1);
    sprintf(sText, "Flann on whole image: "); cv::putText(cvm_input, sText, cv::Point(nPosX, (nPosY+=15)), nFont, nFontSize, c_darkgreen, 1);
    sprintf(sText, "%8.2f", nTimeFlannWhole); cv::putText(cvm_input, sText, cv::Point(nTab, nPosY), nFont, nFontSize, c_darkgreen, 1);

    sprintf(sText, "TOTAL:"); cv::putText(cvm_input, sText, cv::Point(nPosX, (nPosY+=25)), nFont, nFontSize, c_magenta, 1);
    sprintf(sText, "%8.2f", nTimeTotal); cv::putText(cvm_input, sText, cv::Point(nTab, nPosY), nFont, nFontSize, c_magenta, 1);

    sprintf(sText, "N0. of segments:"); cv::putText(cvm_input, sText, cv::Point(nPosX, (nPosY+=30)), nFont, nFontSize, c_green, 1);
    sprintf(sText, "%6d", nSegNr); cv::putText(cvm_input, sText, cv::Point(nTab, nPosY), nFont, nFontSize, c_green, 1);
    sprintf(sText, "Recognition cycle:"); cv::putText(cvm_input, sText, cv::Point(nPosX, (nPosY+=15)), nFont, nFontSize, c_green, 1);
    sprintf(sText, "%8.2f", nTimeRecCycle); cv::putText(cvm_input, sText, cv::Point(nTab, nPosY), nFont, nFontSize, c_green, 1);
    sprintf(sText, "Recognition Found:"); cv::putText(cvm_input, sText, cv::Point(nPosX, (nPosY+=15)), nFont, nFontSize, c_green, 1);
    if (nFoundCnt) sprintf(sText, "%8.2f %2d.", nTimeRecFound, nFoundCnt); else sprintf(sText, "%8.2f", nTimeRecFound);
    cv::putText(cvm_input, sText, cv::Point(nTab, nPosY), nFont, nFontSize, c_green, 1);


    sprintf(sText, "Average elapsed time in ms"); cv::putText(cvm_input, sText, cv::Point(nPosX, (nPosY+=45)), nFont, nFontSize, c_cyan, 1);
    sprintf(sText, "Initialization:"); cv::putText(cvm_input, sText, cv::Point(nPosX, (nPosY+=15)), nFont, nFontSize, c_darkgreen, 1);
    sprintf(sText, "%8.2f", nTimePre_avr); cv::putText(cvm_input, sText, cv::Point(nTab, nPosY), nFont, nFontSize, c_darkgreen, 1);
    sprintf(sText, "Depth Processing:"); cv::putText(cvm_input, sText, cv::Point(nPosX, (nPosY+=15)), nFont, nFontSize, c_darkgreen, 1);
    sprintf(sText, "%8.2f", nTimeDepth_avr); cv::putText(cvm_input, sText, cv::Point(nTab, nPosY), nFont, nFontSize, c_darkgreen, 1);
    sprintf(sText, "Smoothing:"); cv::putText(cvm_input, sText, cv::Point(nPosX, (nPosY+=15)), nFont, nFontSize, c_darkgreen, 1);
    sprintf(sText, "%8.2f", nTimeBlur_avr); cv::putText(cvm_input, sText, cv::Point(nTab, nPosY), nFont, nFontSize, c_darkgreen, 1);

    sprintf(sText, "Segmentation:"); cv::putText(cvm_input, sText, cv::Point(nPosX, (nPosY+=15)), nFont, nFontSize, c_green, 1);
    sprintf(sText, "%8.2f", nTimeSegm_avr); cv::putText(cvm_input, sText, cv::Point(nTab, nPosY), nFont, nFontSize, c_green, 1);
    sprintf(sText, "Tracking:"); cv::putText(cvm_input, sText, cv::Point(nPosX, (nPosY+=15)), nFont, nFontSize, c_green, 1);
    sprintf(sText, "%8.2f", nTimeTrack_avr); cv::putText(cvm_input, sText, cv::Point(nTab, nPosY), nFont, nFontSize, c_green, 1);
    sprintf(sText, "Selective attention:"); cv::putText(cvm_input, sText, cv::Point(nPosX, (nPosY+=15)), nFont, nFontSize, c_green, 1);
    sprintf(sText, "%8.2f", nTimeAtt_avr); cv::putText(cvm_input, sText, cv::Point(nTab, nPosY), nFont, nFontSize, c_green, 1);
    sprintf(sText, "Feature extraction:"); cv::putText(cvm_input, sText, cv::Point(nPosX, (nPosY+=15)), nFont, nFontSize, c_green, 1);
    sprintf(sText, "%8.2f", nTimeSift_avr); cv::putText(cvm_input, sText, cv::Point(nTab, nPosY), nFont, nFontSize, c_green, 1);
    sprintf(sText, "Flann searching:"); cv::putText(cvm_input, sText, cv::Point(nPosX, (nPosY+=15)), nFont, nFontSize, c_green, 1);
    sprintf(sText, "%8.2f", nTimeFlann_avr); cv::putText(cvm_input, sText, cv::Point(nTab, nPosY), nFont, nFontSize, c_green, 1);
    sprintf(sText, "(Recognition:"); cv::putText(cvm_input, sText, cv::Point(nPosX, (nPosY+=15)), nFont, nFontSize, c_darkgreen, 1);
    sprintf(sText, "%8.2f)", nTimeRec_avr); cv::putText(cvm_input, sText, cv::Point(nTab, nPosY), nFont, nFontSize, c_darkgreen, 1);

    sprintf(sText, "Graph based Segm:"); cv::putText(cvm_input, sText, cv::Point(nPosX, (nPosY+=25)), nFont, nFontSize, c_darkgreen, 1);
    sprintf(sText, "%8.2f", nTimeGbSegm_avr); cv::putText(cvm_input, sText, cv::Point(nTab, nPosY), nFont, nFontSize, c_darkgreen, 1);
    sprintf(sText, "SIFT on whole image: "); cv::putText(cvm_input, sText, cv::Point(nPosX, (nPosY+=15)), nFont, nFontSize, c_darkgreen, 1);
    sprintf(sText, "%8.2f", nTimeSiftWhole_avr); cv::putText(cvm_input, sText, cv::Point(nTab, nPosY), nFont, nFontSize, c_darkgreen, 1);
    sprintf(sText, "Flann on whole image: "); cv::putText(cvm_input, sText, cv::Point(nPosX, (nPosY+=15)), nFont, nFontSize, c_darkgreen, 1);
    sprintf(sText, "%8.2f", nTimeFlannWhole_avr); cv::putText(cvm_input, sText, cv::Point(nTab, nPosY), nFont, nFontSize, c_darkgreen, 1);

}



/* Draws information about different settings on the right side of the gui.
 */
void DrawSettings(cv::Mat &cvm_input, int IX, int nSnapFormat, double nDLimit, int nDGradFilterSize,
                  TrackProp stTrack1,
                  int nAttSizeMax, int nAttSizeMin, int nAttPtsMin, int nAttAspect1, int nAttAspect2, int nAttTDMode,
                  double nRecogDClr, int nSiftScales, double nSiftInitSigma, double nSiftPeakThrs, int nFlannKnn, double nFlannMatchFac, int nFlannMatchCnt,
                  double nGSegmSigma, int nGSegmGrThs, int nGSegmMinSize)
{

    CvFont font;
    char sText[128];

    int nPosY = 25, nPosX = IX + 10, nTab = nPosX + 200;
    cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 0.4, 0.4, 0, 1, CV_AA);

    sprintf(sText, "Parameter Settings");
    cv::putText(cvm_input, sText, cv::Point(nPosX, nPosY), nFont, nFontSize, c_cyan, 1);

    sprintf(sText, "Snapshoot format:");
    cv::putText(cvm_input, sText, cv::Point(nPosX, (nPosY+=25)), nFont, nFontSize, c_lemon, 1);
    switch (nSnapFormat) {case 0: sprintf(sText, "bmp"); break; case 1: sprintf(sText, "tif"); break; case 2: sprintf(sText, "jpg"); break;}
    cv::putText(cvm_input, sText, cv::Point(nTab+25, nPosY), nFont, nFontSize, c_lemon, 1);


    sprintf(sText, "Segmentation"); cv::putText(cvm_input, sText, cv::Point(nPosX, (nPosY+=25)), nFont, nFontSize, c_magenta, 1);
    sprintf(sText, "Graph-based segm. Sigma:"); cv::putText(cvm_input, sText, cv::Point(nPosX, (nPosY+=15)), nFont, nFontSize, c_lemon, 1);
    sprintf(sText, "%7.1f", nGSegmSigma); cv::putText(cvm_input, sText, cv::Point(nTab, nPosY), nFont, nFontSize, c_lemon, 1);
    sprintf(sText, "Graph-based segm. Gr:"); cv::putText(cvm_input, sText, cv::Point(nPosX, (nPosY+=15)), nFont, nFontSize, c_lemon, 1);
    sprintf(sText, "%5d", nGSegmGrThs); cv::putText(cvm_input, sText, cv::Point(nTab, nPosY), nFont, nFontSize, c_lemon, 1);
    sprintf(sText, "Graph-based segm. min size:"); cv::putText(cvm_input, sText, cv::Point(nPosX, (nPosY+=15)), nFont, nFontSize, c_lemon, 1);
    sprintf(sText, "%5d", nGSegmMinSize); cv::putText(cvm_input, sText, cv::Point(nTab, nPosY), nFont, nFontSize, c_lemon, 1);
    sprintf(sText, "Depth limit:"); cv::putText(cvm_input, sText, cv::Point(nPosX, (nPosY+=15)), nFont, nFontSize, c_lemon, 1);
    sprintf(sText, "%8.2f m", nDLimit); cv::putText(cvm_input, sText, cv::Point(nTab, nPosY), nFont, nFontSize, c_lemon, 1);
    sprintf(sText, "Filter size:"); cv::putText(cvm_input, sText, cv::Point(nPosX, (nPosY+=15)), nFont, nFontSize, c_lemon, 1);
    sprintf(sText, "%5d pixels", nDGradFilterSize); cv::putText(cvm_input, sText, cv::Point(nTab, nPosY), nFont, nFontSize, c_lemon, 1);


    sprintf(sText, "Tracking"); cv::putText(cvm_input, sText, cv::Point(nPosX, (nPosY+=25)), nFont, nFontSize, c_magenta, 1);
    sprintf(sText, "Tracking mode:"); cv::putText(cvm_input, sText, cv::Point(nPosX, (nPosY+=15)), nFont, nFontSize, c_lemon, 1);
    if (stTrack.Mode) sprintf(sText, "normal"); else sprintf(sText, "optimized"); cv::putText(cvm_input, sText, cv::Point(nTab+25, nPosY), nFont, nFontSize, c_lemon, 1);
    sprintf(sText, "Color mode:"); cv::putText(cvm_input, sText, cv::Point(nPosX, (nPosY+=15)), nFont, nFontSize, c_lemon, 1);
    sprintf(sText, "Normalized rgb"); cv::putText(cvm_input, sText, cv::Point(nTab+25, nPosY), nFont, nFontSize, c_lemon, 1);

    sprintf(sText, "Position distance threshold:"); cv::putText(cvm_input, sText, cv::Point(nPosX, (nPosY+=15)), nFont, nFontSize, c_lemon, 1);
    sprintf(sText, "%8.2f", stTrack1.DPos); cv::putText(cvm_input, sText, cv::Point(nTab, nPosY), nFont, nFontSize, c_lemon, 1);
    sprintf(sText, "Size distance threshold:"); cv::putText(cvm_input, sText, cv::Point(nPosX, (nPosY+=15)), nFont, nFontSize, c_lemon, 1);
    sprintf(sText, "%8.2f", stTrack1.DSize); cv::putText(cvm_input, sText, cv::Point(nTab, nPosY), nFont, nFontSize, c_lemon, 1);
    sprintf(sText, "Color distance threshold:"); cv::putText(cvm_input, sText, cv::Point(nPosX, (nPosY+=15)), nFont, nFontSize, c_lemon, 1);
    sprintf(sText, "%8.2f", stTrack1.DClr); cv::putText(cvm_input, sText, cv::Point(nTab, nPosY), nFont, nFontSize, c_lemon, 1);
    sprintf(sText, "Position distancd factor:"); cv::putText(cvm_input, sText, cv::Point(nPosX, (nPosY+=15)), nFont, nFontSize, c_lemon, 1);
    sprintf(sText, "%8.2f", stTrack1.FPos); cv::putText(cvm_input, sText, cv::Point(nTab, nPosY), nFont, nFontSize, c_lemon, 1);
    sprintf(sText, "Size distancd factor:"); cv::putText(cvm_input, sText, cv::Point(nPosX, (nPosY+=15)), nFont, nFontSize, c_lemon, 1);
    sprintf(sText, "%8.2f", stTrack1.FSize); cv::putText(cvm_input, sText, cv::Point(nTab, nPosY), nFont, nFontSize, c_lemon, 1);
    sprintf(sText, "Color distancd factor:"); cv::putText(cvm_input, sText, cv::Point(nPosX, (nPosY+=15)), nFont, nFontSize, c_lemon, 1);
    sprintf(sText, "%8.2f", stTrack1.FClr); cv::putText(cvm_input, sText, cv::Point(nTab, nPosY), nFont, nFontSize, c_lemon, 1);
    sprintf(sText, "Total distance threshold:"); cv::putText(cvm_input, sText, cv::Point(nPosX, (nPosY+=15)), nFont, nFontSize, c_lemon, 1);
    sprintf(sText, "%8.2f", stTrack1.Dist); cv::putText(cvm_input, sText, cv::Point(nTab, nPosY), nFont, nFontSize, c_lemon, 1);


    sprintf(sText, "Objects upper limit:"); cv::putText(cvm_input, sText, cv::Point(nPosX, (nPosY+=15)), nFont, nFontSize, c_lemon, 1);
    sprintf(sText, "%5d mm", nAttSizeMax); cv::putText(cvm_input, sText, cv::Point(nTab, nPosY), nFont, nFontSize, c_lemon, 1);
    sprintf(sText, "Objects lower limit:"); cv::putText(cvm_input, sText, cv::Point(nPosX, (nPosY+=15)), nFont, nFontSize, c_lemon, 1);
    sprintf(sText, "%5d mm", nAttSizeMin); cv::putText(cvm_input, sText, cv::Point(nTab, nPosY), nFont, nFontSize, c_lemon, 1);
    sprintf(sText, "Objects pixel size lower limit:"); cv::putText(cvm_input, sText, cv::Point(nPosX, (nPosY+=15)), nFont, nFontSize, c_lemon, 1);
    sprintf(sText, "%5d mm", nAttPtsMin); cv::putText(cvm_input, sText, cv::Point(nTab, nPosY), nFont, nFontSize, c_lemon, 1);
    sprintf(sText, "Objects aspect ratio 1:"); cv::putText(cvm_input, sText, cv::Point(nPosX, (nPosY+=15)), nFont, nFontSize, c_lemon, 1);
    sprintf(sText, "%5d", nAttAspect1); cv::putText(cvm_input, sText, cv::Point(nTab, nPosY), nFont, nFontSize, c_lemon, 1);
    sprintf(sText, "Objects aspect ratio 2:"); cv::putText(cvm_input, sText, cv::Point(nPosX, (nPosY+=15)), nFont, nFontSize, c_lemon, 1);
    sprintf(sText, "%5d", nAttAspect2); cv::putText(cvm_input, sText, cv::Point(nTab, nPosY), nFont, nFontSize, c_lemon, 1);
    sprintf(sText, "Selection Mode:"); cv::putText(cvm_input, sText, cv::Point(nPosX, (nPosY+=15)), nFont, nFontSize, c_lemon, 1);
    switch (nAttTDMode) {
    case 0: sprintf(sText, "Top-down selection"); break;
    case 1: sprintf(sText, "no Top-down"); break;}
    cv::putText(cvm_input, sText, cv::Point(nTab+25, nPosY), nFont, nFontSize, c_lemon, 1);

    sprintf(sText, "Recognition"); cv::putText(cvm_input, sText, cv::Point(nPosX, (nPosY+=25)), nFont, nFontSize, c_magenta, 1);
    sprintf(sText, "Color distance threshold:"); cv::putText(cvm_input, sText, cv::Point(nPosX, (nPosY+=15)), nFont, nFontSize, c_lemon, 1);
    sprintf(sText, "%8.2f", nRecogDClr); cv::putText(cvm_input, sText, cv::Point(nTab, nPosY), nFont, nFontSize, c_lemon, 1);
    sprintf(sText, "SIFT scales:"); cv::putText(cvm_input, sText, cv::Point(nPosX, (nPosY+=15)), nFont, nFontSize, c_lemon, 1);
    sprintf(sText, "%5d", nSiftScales); cv::putText(cvm_input, sText, cv::Point(nTab, nPosY), nFont, nFontSize, c_lemon, 1);
    sprintf(sText, "SIFT initial sigma:"); cv::putText(cvm_input, sText, cv::Point(nPosX, (nPosY+=15)), nFont, nFontSize, c_lemon, 1);
    sprintf(sText, "%7.1f", nSiftInitSigma); cv::putText(cvm_input, sText, cv::Point(nTab, nPosY), nFont, nFontSize, c_lemon, 1);
    sprintf(sText, "SIFT peak threshold:"); cv::putText(cvm_input, sText, cv::Point(nPosX, (nPosY+=15)), nFont, nFontSize, c_lemon, 1);
    sprintf(sText, "%8.2f", nSiftPeakThrs); cv::putText(cvm_input, sText, cv::Point(nTab, nPosY), nFont, nFontSize, c_lemon, 1);
    sprintf(sText, "Flann KNN:"); cv::putText(cvm_input, sText, cv::Point(nPosX, (nPosY+=15)), nFont, nFontSize, c_lemon, 1);
    sprintf(sText, "%5d", nFlannKnn); cv::putText(cvm_input, sText, cv::Point(nTab, nPosY), nFont, nFontSize, c_lemon, 1);
    sprintf(sText, "Flann match factor:"); cv::putText(cvm_input, sText, cv::Point(nPosX, (nPosY+=15)), nFont, nFontSize, c_lemon, 1);
    sprintf(sText, "%8.2f", nFlannMatchFac); cv::putText(cvm_input, sText, cv::Point(nTab, nPosY), nFont, nFontSize, c_lemon, 1);
    sprintf(sText, "Flann match count:"); cv::putText(cvm_input, sText, cv::Point(nPosX, (nPosY+=15)), nFont, nFontSize, c_lemon, 1);
    sprintf(sText, "%5d", nFlannMatchCnt); cv::putText(cvm_input, sText, cv::Point(nTab, nPosY), nFont, nFontSize, c_lemon, 1);
}





/* For Mouse events
 *
 * Input:
 * event - information about the mouse event that occured
 * x,y - coordinates of the mouse event that occured
 */
void MouseHandler(int event, int x, int y, int flags, void *param){
    int k;
    bool flag = false;
    switch(event) {
    case CV_EVENT_LBUTTONDOWN:
        for (int i = 0; i < nTaskNrMax; i++) {
            if (y < mnBtnPos[i][1]) continue;
            if (y - mnBtnPos[i][1] > mnBtnPos[i][3]) continue;
            if (x < mnBtnPos[i][0]) continue;
            if (x - mnBtnPos[i][0] <= mnBtnPos[i][2]) {
                k = i;
                flag = true;
            }
        }
        if (flag) vbFlagTask[k] = !vbFlagTask[k];
        if (vbFlagTask[nTaskNrMax-1]) bFlagEnd = true;
        break;
    }
}





void CloseWindow (int wnd_nr) {
    if (vbFlagWnd[wnd_nr]) {
        vbFlagWnd[wnd_nr] = false;
        cv::destroyWindow(vsWndName[wnd_nr]);
    }
}

/* Sets the flag which indicates that the window is open/closed.
 */
void SetFlagWnd (int wnd_nr) {
    vbFlagWnd[wnd_nr] = true;
    cv::namedWindow(vsWndName[wnd_nr]);
    cvMoveWindow(vsWndName[wnd_nr].data(), mWndPos[wnd_nr][0], mWndPos[wnd_nr][1]);
}



void OpenWindow (int wnd_nr) {if(!vbFlagWnd[wnd_nr]) SetFlagWnd (wnd_nr);}





/* Reset time values for recognition only.
 */
void ResetRecTime () {
    nCntRec = 0; nTimeRec_acc = 0; nTimeRec_avr = 0;
    nTimeAtt_acc = 0; nTimeAtt_avr = 0;
    nCntSift = 0; nTimeSift_acc = 0; nTimeSift_avr = 0;
    nCntFlann = 0; nTimeFlann_acc = 0; nTimeFlann_avr = 0;

    nCntRecCycle = 0; nTimeRecCycle_acc = 0; nTimeRecCycle_avr = 0;
    nTimeRecFound_acc = 0; nTimeRecFound_avr = 0;

    nCntFrame_tmp = 0; nTimeTotal_acc = 0; nTimeTotal_avr = 0; nTimeFrame_acc = 0; nTimeFrame_avr = 0;

    nTimeRecCycle = 0; nTimeRecFound = 0; nTimeRecFound_max = 0; nTimeRecFound_min = 10000;
}


/* Resets all time values both displayed and not displayed.
 */
void ResetTime () {
    nCntGbSegm = 0; nTimeGbSegm_acc = 0; nTimeGbSegm_avr = 0;
    nCntSiftWhole = 0; nTimeSiftWhole_acc = 0; nTimeSiftWhole_avr = 0;
    nTimeFlannWhole_acc = 0; nTimeFlannWhole_avr = 0;

    nCntDepth = 0; nTimeDepth_acc = 0; nTimeDepth_avr = 0;
    nCntBlur = 0; nTimeBlur_acc = 0; nTimeBlur_avr = 0;

    nCntSegm = 0; nTimeSegm_acc = 0; nTimeSegm_avr = 0;
    nCntTrack = 0; nTimeTrack_acc = 0; nTimeTrack_avr = 0;

    ResetRecTime ();
}



void TrackbarHandler_none (int){}
void TrackbarHandler_ZLimit (int pos) {if (pos < 5) pos = 5; nDLimit = (float)pos/10; cvSetTrackbarPos(vsTrackbarName[2].data(), vsWndName[stTID.nPrmRecog].data(), pos);}
void TrackbarHandler_ZGradFilterSize (int) {if (nDGradFilterSize < 3) nDGradFilterSize = 3; else {if (!(nDGradFilterSize%2)) nDGradFilterSize++;} cvSetTrackbarPos(vsTrackbarName[3].data(), vsWndName[stTID.nPrmRecog].data(), nDGradFilterSize);}
void TrackbarHandler_ColorMode (int) {if (stTrack.ClrMode > 7) stTrack.ClrMode = 7; if (mnColorHistY_lib.size() == 1) stTrack.ClrMode = 1; cvSetTrackbarPos(vsTrackbarName[21].data(), vsWndName[stTID.nPrmSegm].data(), stTrack.ClrMode);}
void TrackbarHandler_DistPos (int pos) {stTrack.DPos = (float)pos/100; cvSetTrackbarPos(vsTrackbarName[22].data(), vsWndName[stTID.nPrmSegm].data(), pos);}
void TrackbarHandler_DistSize (int pos) {stTrack.DSize = (float)pos/100; cvSetTrackbarPos(vsTrackbarName[23].data(), vsWndName[stTID.nPrmSegm].data(), pos);}
void TrackbarHandler_DistClr (int pos) {stTrack.DClr = (float)pos/100; cvSetTrackbarPos(vsTrackbarName[24].data(), vsWndName[stTID.nPrmSegm].data(), pos);}
void TrackbarHandler_FacPos (int pos) {stTrack.FPos = (float)pos/100; cvSetTrackbarPos(vsTrackbarName[25].data(), vsWndName[stTID.nPrmSegm].data(), pos);}
void TrackbarHandler_FacSize (int pos) {stTrack.FSize = (float)pos/100; cvSetTrackbarPos(vsTrackbarName[26].data(), vsWndName[stTID.nPrmSegm].data(), pos);}
void TrackbarHandler_FacClr (int pos) {stTrack.FClr = (float)pos/100; cvSetTrackbarPos(vsTrackbarName[27].data(), vsWndName[stTID.nPrmSegm].data(), pos);}
void TrackbarHandler_DistTotal (int pos) {stTrack.Dist = (float)pos/100; cvSetTrackbarPos(vsTrackbarName[28].data(), vsWndName[stTID.nPrmSegm].data(), pos);}
void TrackbarHandler_ProMax (int) {if (nAttSizeMax < nAttSizeMin) nAttSizeMax = nAttSizeMin; cvSetTrackbarPos(vsTrackbarName[31].data(), vsWndName[stTID.nPrmSegm].data(), nAttSizeMax);}
void TrackbarHandler_ProMin (int) {if (nAttSizeMax < nAttSizeMin) nAttSizeMin = nAttSizeMax; cvSetTrackbarPos(vsTrackbarName[32].data(), vsWndName[stTID.nPrmSegm].data(), nAttSizeMin);}
void TrackbarHandler_ColorThres (int pos) {nRecogDClr = (float)pos/100; cvSetTrackbarPos(vsTrackbarName[36].data(), vsWndName[stTID.nPrmRecog].data(), pos);}
void TrackbarHandler_SiftScales (int) {if (nSiftScales < 1) nSiftScales = 1; cvSetTrackbarPos(vsTrackbarName[41].data(), vsWndName[stTID.nPrmRecog].data(), nSiftScales);}
void TrackbarHandler_SiftSigma (int pos) {if (pos < 10) pos = 10; nSiftInitSigma = (float)pos/10; cvSetTrackbarPos(vsTrackbarName[42].data(), vsWndName[stTID.nPrmRecog].data(), pos);}
void TrackbarHandler_SiftPeak (int pos) {if (pos > 500) pos = 500; nSiftPeakThrs = (float)pos/1000; cvSetTrackbarPos(vsTrackbarName[43].data(), vsWndName[stTID.nPrmRecog].data(), pos);}
void TrackbarHandler_FlannKnn (int) {if (nFlannKnn < 1) nFlannKnn = 1; cvSetTrackbarPos(vsTrackbarName[51].data(), vsWndName[stTID.nPrmRecog].data(), nFlannKnn);}
void TrackbarHandler_FlannMFac (int pos) {nFlannMatchFac = (float)pos/100; cvSetTrackbarPos(vsTrackbarName[52].data(), vsWndName[stTID.nPrmRecog].data(), pos);}
void TrackbarHandler_GSegmSigma (int pos) {nGSegmSigma = (float)pos/10; cvSetTrackbarPos(vsTrackbarName[17].data(), vsWndName[stTID.nGSegm].data(), pos);}
void TrackbarHandler_DGradC (int pos) {nDGradConst = (float)pos/100; cvSetTrackbarPos(vsTrackbarName[84].data(), vsWndName[stTID.nDepth].data(), pos);}
