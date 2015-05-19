#ifndef _NI_LEGACY_FUNC_INIT_H_
#define _NI_LEGACY_FUNC_INIT_H_

#include <vector>

#include <opencv2/core/core.hpp>

#include <flann/flann.h>

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
float* ReadFlannDataset_Color (const cv::Mat &mFeatureSet,
                               std::vector<std::vector<float> > &mnColorHistY_lib,
                               TrackProp &stTrack);

/**
 * @brief ReadFlannDataset_SiftOnePos
 *
 * Store the feature vectors from the lib file in row major form and returns a pointer to the first address.
 *
 * @param[in] mFeatureSet matrix of features which was extracted from the lib file
 * @param[in] nRecogFeature
 * @param[out] mnSiftExtraFeatures
 * @return Pointer to the first address of the feature vector
 */
float* ReadFlannDataset_SiftOnePos (const cv::Mat &mFeatureSet,
                                    int nRecogFeature,
                                    int &nFlannLibCols_sift,
                                    std::vector <std::vector <float> > &mnSiftExtraFeatures);

/**
 * @brief Build Flann Index
 *
 * Read the library file
 *
 * @param[in] libnr
 * @param[in] sLibFileName full path to library filename
 * @param mnColorHistY_lib
 * @param stTrack
 * @param nFlannLibCols_sift
 * @param FLANNParam
 * @param nFlannDataset
 * @param nRecogFeature
 * @param mnSiftExtraFeatures
 * @param FlannIdx_Sift
 */
void BuildFlannIndex (int libnr,
                      const std::string &sLibFileName,
                      std::vector<std::vector<float> > &mnColorHistY_lib,
                      TrackProp &stTrack,
                      int nFlannLibCols_sift,
                      FLANNParameters &FLANNParam,
                      float * &nFlannDataset,
                      int nRecogFeature,
                      std::vector <std::vector <float> > &mnSiftExtraFeatures,
                      flann_index_t &FlannIdx_Sift);

#endif // _NI_LEGACY_FUNC_INIT_H_
