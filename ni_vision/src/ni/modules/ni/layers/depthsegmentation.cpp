#include "ni/layers/depthsegmentation.h"

#include <opencv2/highgui/highgui.hpp>
#include "elm/core/debug_utils.h"

#include "elm/core/cv/lut.h"
#include "elm/core/cv/mat_utils.h"
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
const float DepthSegmentation::DEFAULT_MAX_GRAD         = 0.005f;
const int DepthSegmentation::DEFAULT_TAU_SIZE           = 200;
const int DepthSegmentation::DEFAULT_LABEL_UNASSIGNED   = 0;

/** @todo why does define guard lead to undefined reference error?
 */
//#ifdef __WITH_GTEST
#include <boost/assign/list_of.hpp>
template <>
elm::MapIONames LayerAttr_<DepthSegmentation>::io_pairs = boost::assign::map_list_of
        ELM_ADD_INPUT_PAIR(detail::BASE_SINGLE_INPUT_FEATURE_LAYER__KEY_INPUT_STIMULUS)
        ELM_ADD_OUTPUT_PAIR(detail::BASE_MATOUTPUT_LAYER__KEY_OUTPUT_RESPONSE)
        ;
//#endif // __WITH_GTEST

DepthSegmentation::DepthSegmentation()
    : base_FeatureTransformationLayer()
{
    Clear();
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
    tau_size_ = p.get<int>(PARAM_TAU_SIZE, DEFAULT_TAU_SIZE);
    max_grad_ = p.get<float>(PARAM_MAX_GRAD, DEFAULT_MAX_GRAD);
}

void DepthSegmentation::Activate(const Signal &signal)
{
    Mat1f g = signal.MostRecentMat1f(name_input_); // weighted gradient after thresholding

    m_ = group(g);
}

bool DepthSegmentation::comparePixels(float current, float neighbor) const
{
    return fabs(current-neighbor) < max_grad_;
}

Mat1i DepthSegmentation::group(const Mat1f &g) const
{
    /* Group pixels into surfaces based on depth discontinuities:
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
    Mat1i surface_labels(g.size(), DEFAULT_LABEL_UNASSIGNED);

    int surface_count = DEFAULT_LABEL_UNASSIGNED;

    Mat1b not_nan = elm::is_not_nan(g);
    cv::bitwise_and(not_nan, g < 100.f, not_nan);

//    DeferredAssign_<int> deferred_substitutions;

//    Mat1f diff_v = g.rowRange(1, g.rows) - g.rowRange(0, g.rows-1);
//    vconcat(Mat1f::zeros(1, g.cols), diff_v, diff_v);

//    Mat1f diff_h = g.colRange(1, g.cols) - g.colRange(0, g.cols-1);
//    hconcat(Mat1f::zeros(1, g.rows), diff_h, diff_h);

    elm::LUT lut(g.total()+1);

    for(int r=0; r<g.rows; r++) {

        for(int c=0; c<g.cols; c++) {

            // skip for 'undefined' pixel
            if(not_nan(r, c)) {

                bool is_matched = false;
                float current = g(r, c);

                // neighbor above, skip for row 0
                if(r > 0 && not_nan(r-1, c)) {

                    if(comparePixels(current, g(r-1, c))) {

                        // current follows above
                        is_matched = true;
                        surface_labels(r, c) = surface_labels(r-1, c);
                    }
                }

                // neighbor to the left, skip for column 0
                if(c > 0 && not_nan(r, c-1)) {

                    if(comparePixels(current, g(r, c-1))) {

                        // already matched with neighbor above?
                        if(is_matched) {

                            // left follows current unless they're already equivalent
                            int surface_label_cur = surface_labels(r, c);
                            int surface_label_left = surface_labels(r, c-1);

                            if(surface_label_cur != surface_label_left) {

                                surface_labels(r, c-1) =
                                        surface_labels(r, c) = lut.update(surface_label_cur, surface_label_left);
                            }
                        }
                        else {

                            // current follows left
                            is_matched = true;
                            surface_labels(r, c) = surface_labels(r, c-1);
                        }
                    }
                }

                if(!is_matched) {

                    surface_labels(r, c) = ++surface_count; // assign to new surface
                    lut.insert(surface_count);
                }
            } // not_nan
        } // column
    } // row

    lut.apply(surface_labels);

    return surface_labels;
}

