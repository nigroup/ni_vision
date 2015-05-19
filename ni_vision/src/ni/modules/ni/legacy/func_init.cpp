#include "ni/legacy/func_init.h"

#include "ni/legacy/trackprop.h"

using namespace std;
using cv::Mat;

float* ReadFlannDataset_Color (const cv::Mat &mFeatureSet,
                               std::vector<std::vector<float> > &mnColorHistY_lib,
                               TrackProp &stTrack) {

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

float* ReadFlannDataset_SiftOnePos (const Mat &mFeatureSet,
                                    int nRecogFeature,
                                    int &nFlannLibCols_sift,
                                    vector<vector <float> > &mnSiftExtraFeatures) {

    nFlannLibCols_sift = 128;

    float *data;
    float *p;

    switch (nRecogFeature) {
    case 20: {

        data = (float*) malloc(mFeatureSet.rows * nFlannLibCols_sift * sizeof(float));
        mnSiftExtraFeatures.resize(mFeatureSet.rows, std::vector<float> (5,0));
        if (!data) {
            printf("Cannot allocate memory.\n"); exit(1);
        }

        printf("Memory allocated for FLANN: %d * float\n",mFeatureSet.rows * nFlannLibCols_sift);
        p = data;

        for (int i = 0; i < mFeatureSet.rows; ++i) {

            for (int j = 0; j < nFlannLibCols_sift; ++j) {

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

void BuildFlannIndex (int libnr,
                      const string &sLibFileName,
                      vector<vector<float> > &mnColorHistY_lib,
                      TrackProp &stTrack,
                      int &nFlannLibCols_sift,
                      FLANNParameters &FLANNParam,
                      float * &nFlannDataset,
                      int nRecogFeature,
                      vector <vector <float> > &mnSiftExtraFeatures,
                      flann_index_t &FlannIdx_Sift) {

    Mat mFeatureSet;
    cv::FileStorage fs(sLibFileName, cv::FileStorage::READ);
    fs["TestObjectFeatureVectors"] >> mFeatureSet;
    fs.release();

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
        nFlannDataset = ReadFlannDataset_Color(mFeatureSet, mnColorHistY_lib, stTrack);  //Store the Input file into memory!
        //FlannIdx_Color = flann_build_index(nFlannDataset, mFeatureSet.rows, mFeatureSet.rows, &speedup, &FLANNParam);
        break;
    case 2:
        nFlannDataset = ReadFlannDataset_SiftOnePos(mFeatureSet, nRecogFeature, nFlannLibCols_sift, mnSiftExtraFeatures);  //Store the Input file into memory!
        float speedup;
        FlannIdx_Sift = flann_build_index(nFlannDataset, mFeatureSet.rows, nFlannLibCols_sift, &speedup, &FLANNParam);
        break;
    }
}
