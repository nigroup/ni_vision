#include "ni/layers/depthgradientrectify.h"

#include <boost/assign/list_of.hpp>

#include "elm/core/cv/mat_utils.h"
#include "elm/core/inputname.h"
#include "elm/core/layerconfig.h"
#include "elm/core/signal.h"
#include "elm/ts/layerattr_.h"

using namespace std;
using namespace cv;
using namespace elm;
using namespace ni;

// paramters
const string DepthGradientRectify::PARAM_MAX_GRAD = "max_grad";

// input keys
const string DepthGradientRectify::KEY_INPUT_GRAD_X = "grad_x";
const string DepthGradientRectify::KEY_INPUT_GRAD_Y = "grad_y";
const string DepthGradientRectify::KEY_INPUT_GRAD_SMOOTH = "grad_smooth";

template <>
elm::MapIONames LayerAttr_<DepthGradientRectify>::io_pairs = boost::assign::map_list_of
        ELM_ADD_INPUT_PAIR(DepthGradientRectify::KEY_INPUT_GRAD_X)
        ELM_ADD_INPUT_PAIR(DepthGradientRectify::KEY_INPUT_GRAD_Y)
        ELM_ADD_INPUT_PAIR(DepthGradientRectify::KEY_INPUT_GRAD_SMOOTH)
        ELM_ADD_OUTPUT_PAIR(detail::BASE_MATOUTPUT_LAYER__KEY_OUTPUT_RESPONSE)
        ;

DepthGradientRectify::DepthGradientRectify()
    : base_MatOutputLayer()
{
    Clear();
}

DepthGradientRectify::DepthGradientRectify(const LayerConfig &config)
    : base_MatOutputLayer(config)
{
    Reset(config);
}

void DepthGradientRectify::Clear()
{
    m_ = Mat1f();
}

void DepthGradientRectify::Reset(const LayerConfig &config)
{
    Reconfigure(config);
}

void DepthGradientRectify::Reconfigure(const LayerConfig &config)
{
    max_ = config.Params().get<float>(PARAM_MAX_GRAD);
}

void DepthGradientRectify::InputNames(const LayerInputNames &io)
{
    name_in_grad_x_ = io.Input(KEY_INPUT_GRAD_X);
    name_in_grad_y_ = io.Input(KEY_INPUT_GRAD_Y);
    name_in_grad_smooth_ = io.Input(KEY_INPUT_GRAD_SMOOTH);
}

void DepthGradientRectify::Activate(const Signal &signal)
{
    Mat1f grad_x = signal.MostRecentMat1f(name_in_grad_x_);
    Mat1f grad_y = signal.MostRecentMat1f(name_in_grad_y_);
    Mat1f grad_s = signal.MostRecentMat1f(name_in_grad_smooth_).clone();

    const float NAN_VALUE = std::numeric_limits<float>::quiet_NaN();

    // set elements that were originally NaN and > threshold to NaN
    // in smoothed gradient's y component

    Mat1b mask_nan = elm::isnan(grad_x);
    cv::bitwise_or(mask_nan, elm::isnan(grad_y), mask_nan);

    grad_s.setTo(NAN_VALUE, mask_nan);
    grad_s.setTo(NAN_VALUE, cv::abs(grad_x) > max_);
    grad_s.setTo(NAN_VALUE, cv::abs(grad_y) > max_);
    //grad_s.setTo(NAN_VALUE, cv::abs(grad_s) > max_);

    m_ = grad_s;
}
