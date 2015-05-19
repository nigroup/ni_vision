#include "ni/legacy/func_init.h"

#include "ni/legacy/trackprop.h"

using namespace std;
using cv::Mat;

float* ReadFlannDataset_Color (const cv::Mat &mFeatureSet, std::vector<std::vector<float> > &mnColorHistY_lib, TrackProp &stTrack) {

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
