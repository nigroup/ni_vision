#ifndef _NI_LEGACY_FUNC_INIT_H_
#define _NI_LEGACY_FUNC_INIT_H_

#include <vector>

#include <opencv2/core/core.hpp>

struct TrackProp;

/**
 * @brief Build color index
 *
 * Store the feature vectors from the lib file in row major form and returns a pointer to the first address.
 *
 * @param[in] mFeatureSet matrix of features which was extracted from the lib file
 * @param[out] matrix of featueres in 2d vector format
 * @param[out] tracking properties modified in-place (members: HistoBin)
 * @return Pointer to the first address of the feature vector
 */
float* ReadFlannDataset_Color (const cv::Mat &mFeatureSet, std::vector<std::vector<float> > &mnColorHistY_lib, TrackProp &stTrack);

#endif // _NI_LEGACY_FUNC_INIT_H_
