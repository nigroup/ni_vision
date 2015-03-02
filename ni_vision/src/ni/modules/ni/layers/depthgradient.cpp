#include "ni/layers/depthgradient.h"

#include <opencv2/highgui/highgui.hpp>

#include "elm/core/debug_utils.h"
#include "elm/core/cv/mat_utils.h"
#include "elm/core/exception.h"
#include "elm/core/layerconfig.h"
#include "elm/core/signal.h"
#include "elm/ts/layerattr_.h"

using namespace std;
using namespace cv;
using namespace elm;
using namespace ni;

/** Define parameters, defaults and I/O keys
  */
// paramters
const std::string DepthGradient::PARAM_GRAD_WEIGHT = "grad_weight";
const std::string DepthGradient::PARAM_GRAD_MAX    = "grad_max";

// defaults
const float DepthGradient::DEFAULT_GRAD_WEIGHT = 0.2f;
const float DepthGradient::DEFAULT_GRAD_MAX    = 0.04f;

// output keys
const std::string DepthGradient::KEY_OUTPUT_GRAD_X = "grad_x";
const std::string DepthGradient::KEY_OUTPUT_GRAD_Y = "grad_y";

/** @todo why does define guard lead to undefined reference error?
 */
//#ifdef __WITH_GTEST
#include <boost/assign/list_of.hpp>
template <>
elm::MapIONames LayerAttr_<DepthGradient>::io_pairs = boost::assign::map_list_of
        ELM_ADD_INPUT_PAIR(detail::BASE_SINGLE_INPUT_FEATURE_LAYER__KEY_INPUT_STIMULUS)
        ELM_ADD_OUTPUT_PAIR(DepthGradient::KEY_OUTPUT_GRAD_X)
        ELM_ADD_OUTPUT_PAIR(DepthGradient::KEY_OUTPUT_GRAD_Y)
        ;
//#endif // __WITH_GTEST

/* Create depth-gradient map from depth map
 *
 * Input:
 * vDepth - depth map
 * vnCloudIdx_d - indices of valid points
 * nDIdxCntTmp - count of valid points
 * nDGradConst - constant of the weighted depth
 * nDSegmDThres - threshold for very steep depth-gradient
 * nDGradNan - constant for the very steep depth-gradient
 * nDsWidth - width of depth map
 * nDGradXMin, nDGradXMax, nDGradYMin, nDGradYMax - max and min values of depth-gradient data
 *
 * Output:
 * vDGradX,vDGradY - depth-gradient maps
 */
//void MakeDGradMap (std::vector<float> vDepth, std::vector<int> vCloudIdx_d, int nDIdxCntTmp, float nDGradConst, float nDSegmDThres, float nDGradNan, int nDsWidth,
//                      float &nDGradXMin, float &nDGradXMax, float &nDGradYMin, float &nDGradYMax, std::vector<float> &vDGradX, std::vector<float> &vDGradY)
//{
//    /////// Processing depth map: make depth gradient map ////////////////////////////////////////////////
//    if (nDsWidth <= 320) {
//        for(int i = 0; i < nDIdxCntTmp; i++) {
//            int x, y, idx;
//            idx = vCloudIdx_d[i];
//            GetPixelPos(idx, nDsWidth, x, y);
//            if (!vDepth[idx]) continue;

//            if (y > 0) {
//                if (vDepth[idx-nDsWidth]) {
//                    //vDGradY[idx] = vDepth[idx] - vDepth[idx-nDsWidth];
//                    vDGradY[idx] = (vDepth[idx] - vDepth[idx-nDsWidth]) / (vDepth[idx] + nDGradConst);
//                    if (fabs(vDGradY[idx]) > nDSegmDThres) vDGradY[idx] = nDGradNan;
//                    else {
//                        if (vDGradY[idx] > nDGradYMax) nDGradYMax = vDGradY[idx];
//                        if (vDGradY[idx] < nDGradYMin) nDGradYMin = vDGradY[idx];
//                    }
//                }
//            }
//            if (x > 0) {
//                if (vDepth[idx-1]) {
//                    (vDGradX[idx] = vDepth[idx] - vDepth[idx-1]) / (vDepth[idx] + nDGradConst);;
//                    if (fabs(vDGradX[idx]) > nDSegmDThres) vDGradX[idx] = nDGradNan;
//                    else {
//                        if (vDGradX[idx] > nDGradXMax) nDGradXMax = vDGradX[idx];
//                        if (vDGradX[idx] < nDGradXMin) nDGradXMin = vDGradX[idx];
//                    }
//                }
//            }
//        }
//    }
//    else {
//        for(int i = 0; i < nDIdxCntTmp; i++) {
//            int x, y, idx;
//            idx = vCloudIdx_d[i];
//            GetPixelPos(idx, nDsWidth, x, y);
//            if (x%2) continue;
//            if (y%2) continue;
//            if (!vDepth[idx]) continue;

//            if (y > 0) {
//                if (vDepth[idx-2*nDsWidth]) {
//                    vDGradY[idx] = (vDepth[idx] - vDepth[idx-2*nDsWidth]) / (vDepth[idx] + nDGradConst);
//                    if (fabs(vDGradY[idx]) > nDSegmDThres) vDGradY[idx] = nDGradNan;
//                    else {
//                        if (vDGradY[idx] > nDGradYMax) nDGradYMax = vDGradY[idx];
//                        if (vDGradY[idx] < nDGradYMin) nDGradYMin = vDGradY[idx];
//                    }
//                    vDGradY[idx+1] = vDGradY[idx];
//                    vDGradY[idx+nDsWidth] = vDGradY[idx];
//                    vDGradY[idx+nDsWidth+1] = vDGradY[idx];
//                }
//            }
//            if (x > 0) {
//                if (vDepth[idx-2]) {
//                    (vDGradX[idx] = vDepth[idx] - vDepth[idx-2]) / (vDepth[idx] + nDGradConst);;
//                    if (fabs(vDGradX[idx]) > nDSegmDThres) vDGradX[idx] = nDGradNan;
//                    else {
//                        if (vDGradX[idx] > nDGradXMax) nDGradXMax = vDGradX[idx];
//                        if (vDGradX[idx] < nDGradXMin) nDGradXMin = vDGradX[idx];
//                    }
//                    vDGradX[idx+1] = vDGradX[idx];
//                    vDGradX[idx+nDsWidth] = vDGradX[idx];
//                    vDGradX[idx+nDsWidth+1] = vDGradX[idx];
//                }
//            }
//        }
//    }
//}

void DepthGradient::Clear()
{
    grad_x_ = Mat1f();
    grad_y_ = Mat1f();
}

void DepthGradient::Reset(const LayerConfig &config)
{
    Reconfigure(config);
}

void DepthGradient::Reconfigure(const LayerConfig &config)
{
    // params
    PTree params = config.Params();
    max_ = params.get<float>(PARAM_GRAD_MAX, DEFAULT_GRAD_MAX);
    w_ = params.get<float>(PARAM_GRAD_WEIGHT, DEFAULT_GRAD_WEIGHT);
}

void DepthGradient::OutputNames(const LayerOutputNames &io)
{
    name_out_grad_x_ = io.Output(KEY_OUTPUT_GRAD_X);
    name_out_grad_y_ = io.Output(KEY_OUTPUT_GRAD_Y);
}

void DepthGradient::Activate(const Signal &signal)
{
    Mat1f in = signal.MostRecentMat1f(name_input_);

    ELM_THROW_BAD_DIMS_IF(in.cols < 2,
                          "Input must have > 1 cols to compute gradient in x direction.");

    // compute gradient in x direction:
    computeDerivative(in, 0, grad_x_);

    const float NaN = std::numeric_limits<float>::quiet_NaN();

    //grad_x_.setTo(NaN, abs(grad_x_) > max_);
    hconcat(grad_x_, Mat1f(grad_x_.rows, 1, NaN), grad_x_);

    ELM_THROW_BAD_DIMS_IF(in.rows < 2,
                          "Input must have > 1 rows to compute gradient in y direction.");

    // compute gradient in y direction:
    computeDerivative(in, 1, grad_y_);
    //grad_y_.setTo(NaN, abs(grad_y_) > max_);
    vconcat(Mat1f(1, grad_y_.cols, NaN), grad_y_, grad_y_);
}

void DepthGradient::Response(Signal &signal)
{
    signal.Append(name_out_grad_x_, grad_x_);
    signal.Append(name_out_grad_y_, grad_y_);
}

DepthGradient::DepthGradient()
    : base_SingleInputFeatureLayer()
{
    Clear();
}

DepthGradient::DepthGradient(const LayerConfig& config)
    : base_SingleInputFeatureLayer(config)
{
    Clear();
    Reconfigure(config);
    IONames(config);
}

void DepthGradient::computeDerivative(const Mat1f &src, int dim, Mat1f &dst) const
{
    Mat1f in_shift;
    Mat1f diff;

    if(dim == 0) {

        // horizontal
        in_shift = src.colRange(1, src.cols);
        diff = in_shift - src.colRange(0, src.cols-1);
    }
    else if(dim == 1) {

        // vertical
        in_shift = src.rowRange(1, src.rows);
        diff = in_shift - src.rowRange(0, src.rows-1);
    }
    else {

        stringstream s;
        s << "Invalid dimension value (" << dim <<"). " <<
             "Expecting either 0 for vertical or 1 for horizontal.";
        ELM_THROW_VALUE_ERROR(s.str());
    }

    // gradient =  diff ./ (in+w)
    cv::add(in_shift, w_, in_shift, isnan(in_shift));
    divide(diff, in_shift, dst);
}
