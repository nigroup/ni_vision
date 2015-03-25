#include "ni/layers/depthgradientsmoothing.h"

#include "elm/core/cv/mat_vector_utils_inl.h"
#include "elm/core/layerconfig.h"
#include "elm/core/signal.h"

#include "ni/legacy/func_segmentation.h"

using namespace std;
using namespace cv;
using namespace elm;
using namespace ni;

const string DepthGradientSmoothing::PARAM_MAX          = "max";
const string DepthGradientSmoothing::PARAM_FILTER_MODE  = "fmode";
const string DepthGradientSmoothing::PARAM_SMOOTH_MODE  = "smode";
const string DepthGradientSmoothing::PARAM_SMOOTH_CENTER    = "scenter";
const string DepthGradientSmoothing::PARAM_BAND_1       = "band1";
const string DepthGradientSmoothing::PARAM_BAND_2       = "band2";
const string DepthGradientSmoothing::PARAM_SMOOTH_FACTOR = "factor";

DepthGradientSmoothing::DepthGradientSmoothing()
    : base_SmoothLayer()
{
    Clear();
}

DepthGradientSmoothing::DepthGradientSmoothing(const LayerConfig &cfg)
    : base_SmoothLayer(cfg)
{
    Reset(cfg);
}

void DepthGradientSmoothing::Reconfigure(const LayerConfig &cfg)
{
    base_SmoothLayer::Reconfigure(cfg); // configures aperture size

    PTree params = cfg.Params();

    max_ = params.get<float>(PARAM_MAX);
    filter_mode_ = params.get<int>(PARAM_FILTER_MODE);
    smode_      = params.get<int>(PARAM_SMOOTH_MODE);
    scenter_    = params.get<int>(PARAM_SMOOTH_CENTER);
    sband1_     = params.get<int>(PARAM_BAND_1);
    sband2_     = params.get<int>(PARAM_BAND_2);
    clfac_      = params.get<float>(PARAM_SMOOTH_FACTOR);
}

void DepthGradientSmoothing::Activate(const Signal &signal)
{
    Mat1f in = signal.MostRecentMat1f(name_input_).clone();
    VecF vec = elm::Mat_ToVec_<float>(in);

    in.setTo(1000.f, elm::isnan(in));
    in.setTo(1000.f, abs(in) > max_);

    VecI valid;

    Mat mask_valid;
    cv::bitwise_and(elm::is_not_nan(in), abs(in) <= max_, mask_valid);

    int i=0;
    for(uchar* mask_data_ptr=mask_valid.data;
        mask_data_ptr != mask_valid.dataend;
        ++mask_data_ptr, ++i) {

        if(*mask_data_ptr) {

            valid.push_back(i);
        }
    }

    VecF output_blur = vec, output_ct = vec;

    Segm_SmoothDepthGrad(vec, valid, in.size(),
                         max_, -max_,
                         1000.f,
                         filter_mode_,
                         ksize_,
                         smode_,
                         scenter_,
                         sband1_,
                         sband2_,
                         clfac_,
                         output_blur, output_ct);

    m_ = Mat1f(output_ct, true);
    m_ = m_.reshape(1, in.rows);
}
