#ifndef _NI_CORE_COLORHISTOGRAM_H_
#define _NI_CORE_COLORHISTOGRAM_H_

#include "elm/core/cv/typedefs_fwd.h"
#include "ni/core/stl/typedefs.h"

namespace ni {

/**
 * @brief compute color histogram over subset of image pixels
 *
 * This method is used to generate the object libraries both by makelib_simple and ni_vision
 * It classifies every pixel into one bin in every channel. With 3 channels and 8 bins per channel that gives
 * 512 possible values for each pixel. The result is saved in a vector of that length, which saves the count
 * of pixel which have a certain value. Later on this vector is then normalized
 * such that the sum over all entries is 1.
 *
 * Bins all gray pixels into the same bin, regardless of intensity
 *
 * @param src continuous bgr color image
 * @param indices subset of pixels to process
 * @param nb_bins no. of histogram bins per color channel
 * @param[out] dst color historam (in rgb order)
 *
 * @todo use OpenCV's calcHist(), keep everything in bgr color order
 */
void computeColorHist(const cv::Mat &src, const ni::VecI &indices, int nb_bins, cv::Mat1f &dst);

} // namespace ni

#endif // _NI_CORE_COLORHISTOGRAM_H_
