/** @file Functions for Segmentation and tracking
 */
#ifndef _NI_LEGACY_FUNC_SEGMENTATION_H_
#define _NI_LEGACY_FUNC_SEGMENTATION_H_

#include <vector>

#include <opencv2/core/core.hpp>

#include "ni/legacy/surfprop.h"
#include "ni/legacy/trackprop.h"

/**
 * @brief linear search for value in vector
 * @param idx_vector list of values
 * @param value value searching for
 * @return position (negative for not found)
 */
int FuncFindPos(const std::vector<int> &idx_vector, int value);

/**
 * @brief Flattening depth-gradient map
 * @param[in] cvm_input depth-gradient map
 * @param[in] index indices of pixels of depth-gradient map which should be processed
 * @param[in] max upper bound for range of grayscale depth-gradient
 * @param[in] min lower bound for range of grayscale depth-gradient
 * @param[in] mode flattening mode
 * @param[in] scenter center bandwidth for flattening
 * @param[in] sband1 range (lower band) of bandwidth for flattening
 * @param[in] sband2 range (upper band) of bandwidth for flattening
 * @param[in] factor center, range and factor of bandwidth for flattening
 * @param[out] output_blur depth-gradient map after blurring
 * @param[out] output_ct depth-gradient map after flattening
 */
void Segm_FlatDepthGrad(const cv::Mat &cvm_input,
                        const std::vector<int> &index,
                        float max, float min,
                        int mode,
                        int scenter,
                        int sband1, int sband2,
                        float factor,
                        std::vector<float> &output_blur,
                        std::vector<float> &output_ct);

/**
 * @brief Smoothing depth-gradient map
 * @param[in] vInput depth-gradient map
 * @param[in] index indices of pixels of depth-gradient map which should be processed
 * @param[in] size size of input image
 * @param[in] max range of depth-gradient
 * @param[in] min range of depth-gradient
 * @param[in] none constant for invalid depth and depth-gradient
 * @param[in] fmode filter mode
 * @param[in] fsize size of filter
 * @param[in] smode
 * @param[in] scenter center for flattening
 * @param[in] sband1 range for flattening
 * @param[in] sband2 range for flattening
 * @param[in] clfac factor for flattening
 * @param[out] output_blur depth-gradient map after blurring
 * @param[out] output_ct depth-gradient map after flattening
 */
void Segm_SmoothDepthGrad(const std::vector<float> &vInput,
                          const std::vector<int> &index,
                          const cv::Size &size,
                          float max, float min,
                          float none,
                          int fmode,
                          int fsize,
                          int smode,
                          int scenter,
                          int sband1, int sband2,
                          float clfac,
                          std::vector<float> &output_blur,
                          std::vector<float> &output_ct);

/**
 * @brief Creating matrix about information of adjacent segments
 * @param[in] vInputMap input map
 * @param[in] input_idx indices of input map which should be processed
 * @param[in] range distance of adjacent segments
 * @param[in] width width of input map
 * @param[out] mnOut neighborhood matrix
 */
void Segm_NeighborMatrix(const std::vector<int> &vInputMap,
                         const std::vector<int> &input_idx,
                         int range,
                         int width,
                         std::vector<std::vector<bool> >& mnOut);

/**
 * @brief Creating matrix about information of adjacent segments (for saliency program)
 * @param[in] vInputMap input map
 * @param[in] input_idx indices of input map which should be processed
 * @param[in] range distance of adjacent segments
 * @param[in] width width of input map
 * @param[out] mnOut neighborhood matrix
 */
void Segm_NeighborMatrix1(const std::vector<int> &vInputMap,
                          const std::vector<int> &input_idx,
                          int range,
                          int width,
                          std::vector<std::vector<bool> >& mnOut);
         
/**
 * @brief Checking if two points are in the same segment
 * @param idx_ref reference point
 * @param idx_cand candidate point
 * @param nSurfCnt number of segments
 * @param seg_map segment map (which maps every point to the segment number it lies in)
 * @param seg_list segment buffer
 */
void Segm_MatchPoints(int idx_ref,
                      int idx_cand,
                      int nSurfCnt,
                      std::vector<int>& seg_map,
                      std::vector<int>& seg_list);

/**
 * @brief Merging two segments
 *
 * Segments are not merged in the sense of reduced to a single entry
 * but rather info of the merged segment is duplicated.
 *
 * @param ref reference segment
 * @param cand candidate segment
 * @param clear flag for merging (true, then merging)
 * @param n number segments
 * @param vnLB label buffer, contains segments labels (modified in-place)
 * @param vnSB size buffer, contains segments sizes (modified in-place)
 * @param vbCB clearing buffer, contains merging flags (indicate if a segment was merged) (modified in-place)
 */
void Segm_MergeSegments(int ref,
                        int cand,
                        bool clear,
                        int n,
                        std::vector<int> &vnLB,
                        std::vector<int> &vnSB,
                        std::vector<bool> &vbCB);

/**
 * @brief Main segmentation function
 * @param[in] vnDGrad depth-gradient map
 * @param[in] input_idx indices of the pixel which should be processed
 * @param[in] tau_s threshold for size differences
 * @param[in] nSegmGradDist threshold for segmentation (if depth-gradient between two pixel is greater than threshold, they get segmented)
 * @param[in] nDepthGradNone constant for invalid depth-gradient
 * @param[in] width width of depth-gradient map
 * @param[in] nMapSize size of depth-gradient map
 * @param[in] x_min coordinates of the area of the map which should be processed
 * @param[in] x_max
 * @param[in] y_min
 * @param[in] y_max
 * @param[out] vnLblMap segment map before post-processing
 * @param[out] vnLblMapFinal segment map
 * @param[out] nSurfCnt number of segments
 */
void Segmentation(const std::vector<float> &vnDGrad,
                  const std::vector<int> &input_idx,
                  int tau_s,
                  float nSegmGradDist,
                  float nDepthGradNone,
                  int width,
                  int nMapSize,
                  int x_min,
                  int x_max,
                  int y_min,
                  int y_max,
                  std::vector<int> &vnLblMap,
                  std::vector<int> &vnLblMapFinal,
                  int &nSurfCnt);

/**
 * @brief Pre-processing of the tracking, extracting object properties in current scene
 *
 * @param[in] nSegmCutSize minimum pixel count for objects to be processed
 * @param[in] nDsWidth width downsampled segment map
 * @param[in] nDsHeight height of downsampled segment map
 * @param[in] vnX coordinate values for point cloud
 * @param[in] vnY
 * @param[in] vnZ
 * @param[in] cvm_rgb_ds downsampled image
 * @param[in] stTrack parameters of tracking process
 * @param[out] stSurf properties for object surfaces for tracking
 * @param[out] nSurfCnt number of segments for tracking (reduced from the number of segmentes after the segmentation)
 */
void Tracking_Pre(int nSegmCutSize,
                  int nDsWidth,
                  int nDsHeight,
                  const std::vector<float> &vnX,
                  const std::vector<float> &vnY,
                  const std::vector<float> &vnZ,
                  const cv:: Mat &cvm_rgb_ds,
                  const TrackProp &stTrack,
                  SurfProp &stSurf,
                  int &nSurfCnt);


/* */
/**
 * @brief Recursive Sub-function of the pre-processing of the optimization of the tracking
 *
 * Elimination of the unique elements in Munkres-matrix

 * @param seg
 * @param j_min
 * @param nMemsCnt
 * @param nObjsNrLimit
 * @param nTrackDist
 * @param huge
 * @param vnSurfCandCnt
 * @param vnMemsCandCnt
 * @param vnMemCandMin
 * @param vnMatchedSeg
 * @param mnDistTmp
 */
void Tracking_OptPreFunc(int seg,
                         int j_min,
                         int nMemsCnt,
                         int nObjsNrLimit,
                         float nTrackDist,
                         float huge,
                         std::vector<int> &vnSurfCandCnt,
                         std::vector<int> &vnMemsCandCnt,
                         std::vector<int> &vnMemCandMin,
                         std::vector<int> &vnMatchedSeg,
                         std::vector<std::vector<float> > &mnDistTmp);

/**
 * @brief Pre-Processing of the optimization of the tracking
 *
 * Elimination of the unique elements in Munkres-matrix

 * @param[in] nMemsCnt
 * @param[in] nSurfCnt
 * @param[in] huge
 * @param[in] nObjsNrLimit
 * @param[in] stTrack
 * @param[out] mnDistTmp modified in-place
 * @param[out] vnSurfCandCnt
 * @param[out] vnSegCandMin
 * @param[out] vnMemsCandCnt
 * @param[out] vnMemCandMin
 * @param[out] vnMatchedSeg
 */
void Tracking_OptPre (int nMemsCnt,
                      int nSurfCnt,
                      int huge,
                      int nObjsNrLimit,
                      TrackProp &stTrack,
                      std::vector<std::vector<float> > &mnDistTmp,
                      std::vector<int> &vnSurfCandCnt,
                      std::vector<int> &vnSegCandMin,
                      std::vector<int> &vnMemsCandCnt,
                      std::vector<int> &vnMemCandMin,
                      std::vector<int> &vnMatchedSeg);

/**
 * @brief Postprocessing of the tracking (Part 1)
 * @param[in] nAttSizeMin
 * @param[in] nMemsCnt
 * @param[in] cnt_new
 * @param[in] objs_new_no
 * @param[in] objs_old_flag
 * @param[in] vnMemsPtsCnt
 * @param[in] mnMemsRCenter
 * @param[in] vnSurfPtsCnt
 * @param[in] mnSurfRCenter
 * @param[out] mnNewSize
 * @param[out] mnNewPos
 * @param[in] flag_mat flag for debugging
 * @param[in] framec frame count
 */
void Tracking_Post1(int nAttSizeMin,
                    int nMemsCnt,
                    int cnt_new,
                    const std::vector<int> &objs_new_no,
                    const std::vector<bool> &objs_old_flag,
                    const std::vector<int> &vnMemsPtsCnt,
                    const std::vector<std::vector<int> > &mnMemsRCenter,
                    const std::vector<int> &vnSurfPtsCnt,
                    const std::vector<std::vector<int> > &mnSurfRCenter,
                    std::vector<std::vector<float> > &mnNewSize,
                    std::vector<std::vector<float> > &mnNewPos,
                    bool flag_mat,
                    int framec);

/**
 * @brief Postprocessing of the tracking (Part 2)
 * @param[in] nMemsCnt
 * @param[in] stTrack tracking params
 * @param[in] stMemsOld
 * @param[out] vnMemsValidIdx
 * @param[out] mnMemsRelPose
 * @param[out] stMems
 * @param[in] framec frame count
 * @param[in] flag_mat flag for debugging
 */
void Tracking_Post2(int nMemsCnt,
                    const TrackProp &stTrack,
                    const SurfProp &stMemsOld,
                    std::vector<int> &vnMemsValidIdx,
                    std::vector<std::vector<float> > &mnMemsRelPose,
                    SurfProp &stMems,
                    int framec,
                    bool flag_mat);

/**
 * @brief TrackingMain tracking function
 *
 * Matching objects surfaces in current frame to object surfaces in Short-Term memory

 * @param[in] nSurfCnt number of object surfaces in current scene
 * @param[in] nObjsNrLimit maximum number of object surfaces which can be handled by the system
 * @param[in] stTrack tracking parameters
 * @param[in] bin number of bins for color histogram
 * @param[in] stSurf object properties for tracking
 * @param[out] stMems properties of object surfaces in the Short-Term Memory
 * @param[out] nMemsCnt number of object surfaces in Short-Term Memory
 * @param[out] vnMemsValidIdx
 * @param[out] mnMemsRelPose
 * @param[in] flag_mat flag for debugging
 * @param[in] framec frame count
 */
void Tracking(int nSurfCnt,
              int nObjsNrLimit,
              const TrackProp &stTrack,
              int bin,
              const SurfProp &stSurf,
              SurfProp &stMems,
              int &nMemsCnt,
              std::vector<int> &vnMemsValidIdx,
              std::vector<std::vector<float> > &mnMemsRelPose,
              bool flag_mat,
              int framec);


#endif // _NI_LEGACY_FUNC_SEGMENTATION_H_
