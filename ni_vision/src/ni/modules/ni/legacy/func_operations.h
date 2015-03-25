/*
 * Functions for Histogram calculation and processing of image data
 */
#ifndef _NI_LEGACY_FUNC_OPERATIONS_H_
#define _NI_LEGACY_FUNC_OPERATIONS_H_

#include <vector>

#include <opencv2/core/core.hpp>

/* Converting vector indices to matrix indices
 *
 * Input:
 * idx - vector index
 * width - number of columns in matrix
 *
 * Output:
 * Matrix indices
 */
void GetPixelPos(int idx, int width, int& x, int& y);



/*  Converting matrix indices to vector indices
 */
int GetPixelIdx(int x, int y, int width);



/* Drawing depth image. Values that are greater than max or smaller than min are set to max or min.
 *
 * Input:
 * vnInput - vector of depth values from kinect cam
 * mode - false: close regions are red, far regions are blue; true: opposite
 * max,min - values for largest and smallest value
 *
 * Output (by reference):
 * cvm_out - opencv image matrix of the depth image (as an rgb image)
 */
void DrawDepth(std::vector<float> vnInput, bool mode, float max, float min, cv::Mat &cvm_out);

/* Drawing depth image from certain values of the input vector. Values that are greater than max or smaller than min are set to max or min.
 *
 * Input:
 * vnInput - vector of depth values from kinect cam
 * index - vector of indices from which the depth image should be computed
 * mode - false: close regions are red, far regions are blue; true: opposite
 * max,min - values for largest and smallest value
 *
 * Output (by reference):
 * cvm_out - opencv image matrix of the depth image (as an rgb image)
 */
void DrawDepth(std::vector<float> vnInput, std::vector<int> index, bool mode, float max, float min, cv::Mat &cvm_out);

/* Convert from depth image to depth vector. Inverse function to Drawdepth
 *Input:
 * input - depth image
 * index - vector of indices from which the depth image should be computed
 * max,min - values for largest and smallest value
 *
 * Output (by reference):
 * vnOut - vector of depth values
 */
void CalcDepth(cv::Mat input, std::vector<int> index, float max, float min, std::vector<float>& vnOut);



/* Convert from depth-gradient image to depth-gradient vector. Inverse function to DrawdepthGrad.
 * Input:
 * input - depth-gradient image
 * index - vector of indices from which the depth-gradient image should be computed
 * max,min - values for largest and smallest value
 * none - default depth-gradient value for those values where the depth-gradient is not defined
 *
 * Output (by reference):
 * vnOut - vector of depth-gradient values
 */
void CalcDepthGrad(cv::Mat input, std::vector<int> index, float max, float min, float none, int amp, std::vector<float> &vnOut);

/* Convert from depth-gradient vector to depth-gradient image. Values that are greater than max or smaller than min are set to max or min.
 *
 * Input:
 * vnInput - vector of depth-gradient values from kinect cam
 * index - vector of indices from which the depth-gradient image should be computed
 * mode - false: close regions are red, far regions are blue; true: opposite
 * max,min - values for largest and smallest value
 * none - default depth-gradient value for those values where the depth-gradient is not defined
 *
 * Output (by reference):
 * cvm_out - opencv image matrix of the depth-gradient image (as an rgb image)
 */
void DrawDepthGrad(std::vector<float> vnInput, std::vector<int> index, bool mode, float max, float min, float none, cv::Mat &cvm_out);

// This method is used to generate the object libraries both by makelib_simple and ni_vision
// It classifies every pixel into one bin in every channel. With 3 channels and 8 bins per channel that gives
// 512 possible values for each pixel. The result is saved in a vector of that length, which saves the count
// of pixel which have a certain value. Later on this vector is then normalized
// such that the sum over all entries is 1.
void Calc3DColorHistogram(const cv::Mat cvm_input, const std::vector<int> index, int bin_base, std::vector<float> &vnOut);

/* Convert from map to objects surfaces.
 *
 * Input:
 * vnInput - map
 * obj_nr - number of object surfaces
 * nDsSize - size of the input vector
 *
 * Output:
 * mnOut - vector of surfaces
 * cnt - number of pixels of a surface
 */
void Map2Objects(std::vector<int> vnInput, int obj_nr, int nDsSize, std::vector<std::vector<int> > &mnOut, std::vector<int> &cnt);

/* Assigning color values to image data
 *
 * Input:
 * idx - index
 * color - color which should be assigned
 *
 * Output:
 * cvm_out - output image
 */
void AssignColor(int idx, cv::Scalar color, cv::Mat &cvm_out);

/* Paint all object surfaces in the image with specified colors
 *
 * Input:
 * vnInput - map
 * nDsSize - size of the input vector
 * mnColorTab - predefined colors for the object surfaces
 *
 * Output:
 * cvm_out - output image
 */
void Map2Image(std::vector<int> vnInput, int nDsSize, std::vector<cv::Scalar> mnColorTab, cv::Mat &cvm_out);

/* Paint an object surface in the image with a specified color
 *
 * Input:
 * vnInput - vector of pixels of a surface
 * color - predefined color of the image
 *
 * Output:
 * cvm_out - output image
 */
void Idx2Image(std::vector<int> vnInput, cv::Scalar color, cv::Mat &cvm_out);

#endif // _NI_LEGACY_FUNC_OPERATIONS_H_
