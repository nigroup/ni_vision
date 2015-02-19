#include "ni/layers/depthsegmentation.h"

#include <opencv2/highgui/highgui.hpp>
#include "elm/core/debug_utils.h"

#include "elm/core/exception.h"
#include "elm/core/layerconfig.h"
#include "elm/core/signal.h"
#include "elm/ts/layerattr_.h"

using namespace std;
using namespace cv;
using namespace elm;
using namespace ni;

/** Define parameters, non-integral defaults and any remaining I/O keys
  */
// paramters
const string DepthSegmentation::PARAM_MAX_GRAD = "max_grad";
const string DepthSegmentation::PARAM_TAU_SIZE = "tau_size";

// defaults
const float DepthSegmentation::DEFAULT_MAX_GRAD = 0.005f;

DepthSegmentation::DepthSegmentation()
    : base_FeatureTransformationLayer()
{
    Clear();
}

DepthSegmentation::DepthSegmentation(const LayerConfig &config)
    : base_FeatureTransformationLayer(config)
{
    Clear();
    Reconfigure(config);
    IONames(config);
}

void DepthSegmentation::Clear()
{
    m_ = Mat1f();
}

void DepthSegmentation::Reset(const LayerConfig &config)
{
    Reconfigure(config);
}

void DepthSegmentation::Reconfigure(const LayerConfig &config)
{
    PTree p = config.Params();
    tau_size_ = p.get<int>(PARAM_TAU_SIZE);
    max_grad_ = p.get<float>(PARAM_MAX_GRAD);
}

void DepthSegmentation::Activate(const Signal &signal)
{
    Mat1f g = signal.MostRecent(name_input_); // weighted gradient after thresholding

    /* 1. Group pixels into surfaces based on depth discontinuities:
     *
     * row-wise iteration from top-left to bottom-right
     * for each pixel
     *  do:
     *      compare pixel to preceeding horizontal and vertical neighbors
     *      (e.g. g(r, c-1) g(r-1, c)
     *      if neither current nor neighbor is 'undefined'
     *          AND
     *          difference is < threshold
     *          then:
     *              assign pixel matching neighbors segment (e.g. merge pixels to surface)
     *              if both pixel matches to both neighbors
     *              then:
     *              merge all into same surface // todo: what about neighbors preceeding neighbors?
     *      else if could not match and current pixel is defined
     *          then:
     *          start new surface and assing pixel to it
     *      else if current pixel is undefined
     *          skip pixel
     */
}
