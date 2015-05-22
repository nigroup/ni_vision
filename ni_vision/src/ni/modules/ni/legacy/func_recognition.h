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

/**
 * @brief Get SIFT keypoints from input image (computed by libsiftfast)
 * @param[in] input input image
 * @param[in] nSiftScales SIFT parameter, no. of scales
 * @param[in] nSiftInitSigma SIFT parameter, sigma
 * @param[in] nSiftPeakThrs SIFT parameter, peak thresholds
 * @param[in] x
 * @param[in] y
 * @param[in] width
 * @param[in] height
 * @param[out] keypts keypoints detected
 */
void GetSiftKeypoints(const cv::Mat &input,
                      int nSiftScales,
                      double nSiftInitSigma,
                      double nSiftPeakThrs,
                      int x,
                      int y,
                      int width,
                      int height,
                      Keypoint &keypts);

/**
 * @brief Calculating delta scale orientation (by Sahil)
 * @param[in] input_idx indices of input SIFT keypoints
 * @param[in] vnDeltaScale delta scale of input keypoints
 * @param[in] vnDeltaOri delta orientation of input keypoints
 * @param[out] nDeltaScale output scale (think of an "average value")
 * @param[in] nDeltaBinNo number of bins
 * @param[in] nMaxDeltaOri range of delta orientation
 * @param[in] nMinDeltaOri range of delta orientation
 * @param[in] T_orient
 * @param[in] T_scale
 * @param[out] nTruePositives
 * @param[out] output_flag
 */
void CalcDeltaScaleOri(const std::vector<int> &input_idx,
                       const std::vector<double> &vnDeltaScale,
                       const std::vector<double> &vnDeltaOri,
                       double& nDeltaScale,
                       int nDeltaBinNo,
                       double nMaxDeltaOri,
                       double nMinDeltaOri,
                       double T_orient,
                       double T_scale,
                       int& nTruePositives,
                       std::vector<bool>& output_flag);

/**
 * @brief Resetting properties of object surfaces
 * @param nObjsNrLimit maximum number of object surfaces that can be processed
 * @param nTrackHistoBin_max (number of bins)^number of channels
 * @param nRecogRtNr
 * @param[out] stMems properties of object surfaces in the Short-Term Memory
 * @param[out] nMemsCnt count of object surfaces
 * @param[out] nProtoNr number of current object surface
 * @param[out] nFoundCnt count of found object surfaces
 * @param[out] nFoundNr number of found object surface
 * @param[out] vnRecogRating recognition rating (like a priority) of object surfaces
 */
void ResetMemory (int nObjsNrLimit,
                  int nTrackHistoBin_max,
                  int nRecogRtNr,
                  SurfProp &stMems,
                  int &nMemsCnt,
                  int &nProtoNr,
                  int &nFoundCnt,
                  int &nFoundNr,
                  std::vector<int> &vnRecogRating);

/**
 * @brief Attention: Top-Down guidance - Reordering all properties of object surfaces in the memory after priority
 *
 * @param[out] vbProtoCand candidate flag for object surfaces (true, then candidate for attention)
 * @param[out] stMems properties of object surfaces in the Short-Term Memory
 * @param[in] nMemsCnt count of object surfaces
 * @param[out] veCandClrDist color differences of objects surfaces (to target object)
 */
void Attention_TopDown (std::vector<bool> &vbProtoCand,
                        SurfProp &stMems,
                        int nMemsCnt,
                        std::vector<std::pair<float, int> > &veCandClrDist);

/**
 * @brief Selection of a candidate object surface
 * @param[in] nMemsCnt count of candidate object surfaces
 * @param[in] vbProtoCand flag for candidate object surfaces
 * @param[in] vnMemsFound state of candidate; 0: not inspected & not found, 1:not inspected & found, 2: inspected & not found, 3: inspected & found
 * @param[out] nCandID ID of selected candidate object surface
 */
void Attention_Selection (int nMemsCnt,
                          const std::vector<bool> &vbProtoCand,
                          const std::vector<int> &vnMemsFound,
                          int &nCandID);

/**
 * @brief Extraction of pixel indices for the selected candidate object surface from original image
 * (before that the indices are from the downsampled image)
 *
 * @param[in] nCandID ID of selected candidate object surface
 * @param[in] nImgScale ratio between original and downsampled image
 * @param[in] nDsWidth width of downsampled image
 * @param[in] vnMemsPtsCnt count of pixel which belong to the object surface (of downsampled image)
 * @param[in] cvm_rgb_org original image
 * @param[out] cvm_cand_tmp image of candidate object surface
 * @param[in] mnMemsPtsIdx indices of pixel which belong to the object surfae (of downsampled image)
 * @param[out] vnIdxTmp indices of pixel which belong to the object surfae (of original image)
 */
void Recognition_Attention (int nCandID,
                            int nImgScale,
                            int nDsWidth,
                            const std::vector<int> &vnMemsPtsCnt,
                            const cv::Mat &cvm_rgb_org,
                            cv::Mat &cvm_cand_tmp,
                            const std::vector<std::vector<int> > &mnMemsPtsIdx,
                            std::vector<int> &vnIdxTmp);

/**
 * @brief FLANN (fast library for approximate nearest neighbor) matching SIFT keypoints to find target object
 *
 * @param[in] tcount
 * @param[in] nFlannKnn parameter of FLANN
 * @param[in] nFlannLibCols_sift dimension of library which stores the sift trajectories
 * @param[in] nFlannMatchFac merge factor (parameter of FLANN)
 * @param[in] mnSiftExtraFeatures matrix which stores sift trajectories (from training)
 * @param[in] FlannIdx_Sift FLANN index (type is from flann library)
 * @param[in] FLANNParam flann parameters (struct from flann lib)
 * @param[in] keypts input SIFT keypoints
 * @param[out] nKeyptsCnt count of matched SIFT keypoints
 * @param[out] nFlannIM count of initial matched keypoints after flann matching
 * @param[out] vnSiftMatched matched sift keypoints
 * @param[out] vnDeltaScale delta scale of keypoints
 * @param[out] vnDeltaOri delta orientation of keypoints
 * @param[out] nMaxDeltaOri max delta orientation
 * @param[out] nMinDeltaOri min delta orientation
 * @param[out] nMaxDeltaScale max Scale orientation
 * @param[out] nMinDeltaScale min Scale orientation
 */
void Recognition_Flann (int tcount,
                        int nFlannKnn,
                        int nFlannLibCols_sift,
                        double nFlannMatchFac,
                        const std::vector <std::vector <float> > &mnSiftExtraFeatures,
                        flann_index_t FlannIdx_Sift,
                        struct FLANNParameters FLANNParam, //struct...
                        Keypoint keypts,
                        int &nKeyptsCnt,
                        int &nFlannIM,
                        std::vector<int> &vnSiftMatched,
                        std::vector<double> &vnDeltaScale,
                        std::vector<double> &vnDeltaOri,
                        double &nMaxDeltaOri,
                        double &nMinDeltaOri,
                        double &nMaxDeltaScale,
                        double &nMinDeltaScale);

/**
 * @brief Recognition process of a selected candidate. Determines if a selected object is the target object
 * @param[in] nCandID ID of the candidate in the list candidate object surfaces
 * @param[in] nImgScale ratio of original and downsampled image
 * @param[in] nDsWidth
 * @param nTimeRatio ratio of milli- and nanoseconds (i.e. 10⁶)
 * @param[in] nMemsCnt number of object surfaces
 * @param[in] nTrackHistoBin_max (number of bin)^(number of channels)
 * @param[in] cvm_rgb_org original rgb image
 * @param[in] cvm_rec_org original image for recognition process
 * @param[in] cvm_rec_ds downsampled original image for recognition process
 * @param stMems
 * @param[in] nTrackCntLost threshold for vnMemsLostCnt
 * @param[out] nCandCnt number of candidate object surfaces
 * @param[out] nCandClrDist buffer of color histogram difference between the current candidate and the target object
 * @param[out] nFoundCnt count of found objects in a recognition cycle
 * @param[out] nFoundNr number of found object
 * @param[out] nFoundFrame number of frame where the object was found
 * @param[out] nCandKeyCnt  count of keypoints for the current candidate
 * @param[out] nCandRX coordinates of the 2D-bounding box for the current candidate
 * @param[out] nCandRY coordinates of the 2D-bounding box for the current candidate
 * @param[out] nCandRW coordinates of the 2D-bounding box for the current candidate
 * @param[out] nCandRH coordinates of the 2D-bounding box for the current candidate
 * @param[out] t_rec_found_start variable for time measurement
 * @param[out] t_rec_found_end variable for time measurement
 * @param[out] bSwitchRecordTime flag for time measurement
 * @param[out] nRecogRtNr count of frames to record
 * @param[out] vnRecogRating_tmp vector of results for time measurement
 * @param[out] cvm_cand image of the current candidate
 * @param bTimeSift
 * @param bTimeFlann
 * @param bRecogClrMask
 * @param stTrack
 * @param mnColorHistY_lib
 * @param nSiftScales
 * @param nSiftInitSigma
 * @param nSiftPeakThrs
 * @param nTimeSift
 * @param tcount
 * @param nFlannKnn
 * @param nFlannLibCols_sift
 * @param nFlannMatchFac
 * @param mnSiftExtraFeatures
 * @param FlannIdx_Sift
 * @param FLANNParam
 * @param T_numb
 * @param T_orient
 * @param T_scale
 * @param nDeltaBinNo
 * @param nFlannMatchCnt
 * @param nRecogDClr
 * @param nTimeRecFound
 * @param nCtrFrame
 * @param nCtrFrame_tmp
 * @param vbFlagTask
 * @param stTID
 * @param mnTimeMeas1
 * @param mnTimeMeas2
 * @param nCtrRecCycle
 * @param nTimeRecFound_max
 * @param nTimeRecFound_min
 * @param nTimeFlann
 * @param nRecordMode
 * @param c_red
 * @param c_white
 * @param c_blue
 * @param c_lemon
 */
void Recognition (int nCandID,
                  int nImgScale,
                  int nDsWidth,
                  int nTimeRatio,
                  int nMemsCnt,
                  int nTrackHistoBin_max,
                  const cv::Mat &cvm_rgb_org,
                  cv::Mat &cvm_rec_org,
                  cv::Mat &cvm_rec_ds,
                  SurfProp &stMems,
                  int nTrackCntLost,
                  int &nCandCnt,
                  float nCandClrDist,
                  int &nFoundCnt,
                  int &nFoundNr,
                  int &nFoundFrame,
                  int &nCandKeyCnt,
                  int &nCandRX,
                  int &nCandRY,
                  int &nCandRW,
                  int &nCandRH,
                  struct timespec t_rec_found_start,
                  struct timespec t_rec_found_end,
                  bool bSwitchRecordTime,
                  int nRecogRtNr,
                  std::vector<int> &vnRecogRating_tmp,
                  cv::Mat cvm_cand,
                  //
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
                  const cv::Scalar &c_lemon);

#endif // _NI_LEGACY_FUNC_RECOGNITION_H_
