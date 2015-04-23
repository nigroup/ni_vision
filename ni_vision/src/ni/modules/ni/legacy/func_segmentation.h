/*
 * Functions for Segmentation and tracking
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
 * @return position (if pos==size then value not found)
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

/* Smoothing depth-gradient map
 *
 * Input:
 * vInput -
 * index - indices of pixels of depth-gradient map which should be processed
 * size - size of input image
 * max, min - range of depth-gradient
 * none - constant for invalid depth and depth-gradient
 * fmode - filter mode
 * fsize - size of filter
 * smode, scenter, sband1, sband2, clfac - mode center, range and factor for flattening
 *
 * Output:
 * output_blur - depth-gradient map after blurring
 * output_ct - depth-gradient map after flattening
 */
void Segm_SmoothDepthGrad (std::vector<float> vInput, std::vector<int> index, cv::Size size, float max, float min, float none, int fmode, int fsize, int smode, int scenter, int sband1, int sband2, float clfac, std::vector<float> &output_blur, std::vector<float> &output_ct);

/* Creating matrix about information of adjacent segments
 *
 * Input:
 * vInputMap - input map
 * input idx - indices of input map which should be processed
 * range - distance of adjacent segments
 * width - width of input map
 *
 * Output:
 * mnOut - neighborhood matrix
 */
void Segm_NeighborMatrix (std::vector<int> vInputMap, std::vector<int> input_idx, int range, int width, std::vector<std::vector<bool> >& mnOut);

/* Creating matrix about information of adjacent segments (for saliency program)
*
* Input:
* vInputMap - input map
* input idx - indices of input map which should be processed
* range - distance of adjacent segments
* width - width of input map
*
* Output:
* mnOut - neighborhood matrix
*/
void Segm_NeighborMatrix1 (std::vector<int> vInputMap, std::vector<int> input_idx, int range, int width, std::vector<std::vector<bool> >& mnOut);
         
/* Checking if two points are in the same segment
 *
 * Input:
 * idx_ref - reference point
 * idx_cand - candidate point
 * nSurfCnt - number of segments
 *
 * Output:
 * seg_map - segment map (which maps every point to the segment number it lies in)
 * seg_list - segment buffer */
void Segm_MatchPoints (int idx_ref, int idx_cand, int nSurfCnt, std::vector<int>& seg_map, std::vector<int>& seg_list);

/* Merging two segments
 * 
 * 		Segments are not merged in the sense of reduced to a single entry
 * 		but rather info of the merged segment is duplicated.
 * Input:
 * ref - reference segment
 * cand - candidate segment
 * clear - flag for merging (true, then merging)
 * n - number segments
 *
 * Output:
 * vnLB - label buffer, contains segments labels
 * vnSB - size buffer, contains segments sizes
 * vbCB - clearing buffer, contains merging flags (indicate if a segment was merged)
 */
void Segm_MergeSegments(int ref, int cand, bool clear, int n, std::vector<int> &vnLB, std::vector<int> &vnSB, std::vector<bool> &vbCB);

/* Main segmentation function
 *
 * Input:
 * vnDGrad - depth-gradient map
 * input_idx - indices of the pixel which should be processed
 * tau_s - threshold for size differences
 * nSegmGradDist - threshold for segmentation (if depth-gradient between two pixel is greater than threshold, they get segmented)
 * nDepthGradNone -  constant for invalid depth-gradient
 * width - width of depth-gradient map
 * nMapSize - size of depth-gradient map
 * x_min, x_max, y_min, y_max - coordinates of the area of the map which should be processed
 *
 * Output:
 * vnLblMap - segment map before post-processing
 * vnLblMapFinal - segment map
 * nSurfCnt - number of segments
 */
void Segmentation (std::vector<float> vnDGrad, std::vector<int> input_idx, int tau_s, float nSegmGradDist, float nDepthGradNone,
                        int width, int nMapSize, int x_min, int x_max, int y_min, int y_max, std::vector<int> &vnLblMap, std::vector<int> &vnLblMapFinal, int &nSurfCnt);

/* Pre-processing of the tracking, extracting object properties in current scene
 *
 * Input:
 * nSegmCutSize - minimum pixel count for objects to be processed
 * nDsWidth, nDsHeight - width and height of downsampled segment map
 * vnX, vnY, vnZ - coordinate values for point cloud
 * cvm_rgb_ds - downsampled image
 * stTrack - parameters of tracking process
 *
 * Output:
 * stSurf - properties for object surfaces for tracking (siehe struct SurfProp)
 * nSegmCnt - number of segments for tracking (reduced from the number of segmentes after the segmentation)
 */
void Tracking_Pre (int nSegmCutSize, int nDsWidth, int nDsHeight, std::vector<float> vnX, std::vector<float> vnY, std::vector<float> vnZ, cv:: Mat cvm_rgb_ds, TrackProp stTrack,
                  SurfProp &stSurf, int &nSurfCnt);


/* Sub-function of the pre-processing of the optimization of the tracking: elemination of the unique elements in Munkres-matrix
 */
void Tracking_OptPreFunc (int seg, int j_min, int nMemsCnt, int nObjsNrLimit, float nTrackDist, float huge, std::vector<int> &vnSurfCandCnt, std::vector<int> &vnMemsCandCnt, std::vector<int> &vnMemCandMin, std::vector<int> &vnMatchedSeg, std::vector<std::vector<float> > &mnDistTmp);

/* Pre-Processing of the optimization of the tracking: elemination of the unique elements in Munkres-matrix
 */
void Tracking_OptPre (int nMemsCnt, int nSurfCnt, int huge, int nObjsNrLimit,
                      const TrackProp &stTrack,
                      std::vector<std::vector<float> > &mnDistTmp, std::vector<int> &vnSurfCandCnt, std::vector<int> &vnSegCandMin, std::vector<int> &vnMemsCandCnt, std::vector<int> &vnMemCandMin, std::vector<int> &vnMatchedSeg);

/* Postprocessing of the tracking
  */
void Tracking_Post1(int nAttSizeMin, int nMemsCnt, int cnt_new, std::vector<int> objs_new_no, std::vector<bool> objs_old_flag,
                   std::vector<int> vnMemsPtsCnt, std::vector<std::vector<int> > mnMemsRCenter, std::vector<int> vnSurfPtsCnt, std::vector<std::vector<int> > mnSurfRCenter,
                   std::vector<std::vector<float> > &mnNewSize, std::vector<std::vector<float> > &mnNewPos, bool flag_mat, int framec);

/* Postprocessing of the tracking
  */
void Tracking_Post2(int nMemsCnt, const TrackProp &stTrack, SurfProp stMemsOld, std::vector<int> &vnMemsValidIdx, std::vector<std::vector<float> > &mnMemsRelPose, SurfProp &stMems, int framec, bool flag_mat);

/* Main tracking function - matching objects surfaces in current frame to object surfaces in Short-Term memory
 *
 * Input:
 * nSurfCnt - number of object surfaces in current scene
 * nObjsNrLimit - maximum number of object surfaces which can be handled by the system
 * dp_dia - position displacement
 * stTrack -  tracking parameters
 * bin - number of bins for color histogram
 * stSurf - object properties for tracking (siehe struct SurfProp)
 *
 * Output:
 * stMems - properties of object surfaces in the Short-Term Memory (SurfProp is self-defined struct)
 * nMemsCnt - number of object surfaces in Short-Term Memory
 */
void Tracking (int nSurfCnt, int nObjsNrLimit, TrackProp stTrack, int bin, SurfProp stSurf, SurfProp &stMems, int &nMemsCnt,
              std::vector<int> &vnMemsValidIdx, std::vector<std::vector<float> > &mnMemsRelPose, bool flag_mat, int framec);


#endif // _NI_LEGACY_FUNC_SEGMENTATION_H_
