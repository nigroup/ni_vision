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
const int DepthSegmentation::DEFAULT_TAU_SIZE   = 200;

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
    tau_size_ = p.get<int>(PARAM_TAU_SIZE, DEFAULT_TAU_SIZE);
    max_grad_ = p.get<float>(PARAM_MAX_GRAD, DEFAULT_MAX_GRAD);
}

void DepthSegmentation::Activate(const Signal &signal)
{
    Mat1f g = signal.MostRecent(name_input_); // weighted gradient after thresholding

    group(g);



}

void DepthSegmentation::group(const Mat1f g)
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
    const int LABEL_UNASSIGNED = 0;
    Mat1i surface_labels(g.size(), LABEL_UNASSIGNED);

    int surface_count = LABEL_UNASSIGNED;

    Mat1b not_nan = g == g;

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
                            if(surface_labels(r, c-1) != surface_labels(r, c)) {

                                /* propagate new assignment to top left quadrant
                                 * relative to current pixel
                                 */
                                Mat1i tl = surface_labels(Rect2i(0, 0, c+1, r+1));
                                tl.setTo(surface_labels(r, c), tl == surface_labels(r, c-1));
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
                }
            } // not_nan
        } // column
    } // row

    m_ = surface_labels;
}

bool DepthSegmentation::comparePixels(float current, float neighbor)
{
    return fabs(current-neighbor) < max_grad_;
}
