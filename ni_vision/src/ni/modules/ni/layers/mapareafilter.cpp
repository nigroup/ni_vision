#include "ni/layers/mapareafilter.h"

#include "ni/layers/depthsegmentation.h"

#include "elm/core/cv/mat_vector_utils.h"
#include "elm/core/debug_utils.h"
#include "elm/core/exception.h"
#include "elm/core/graph/graphattr.h"
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
const string MapAreaFilter::PARAM_TAU_SIZE = "tau_size";

// defaults
const int MapAreaFilter::DEFAULT_TAU_SIZE           = 200;
const int MapAreaFilter::DEFAULT_LABEL_UNASSIGNED   = 0;


#include <boost/assign/list_of.hpp>
template <>
elm::MapIONames LayerAttr_<MapAreaFilter>::io_pairs = boost::assign::map_list_of
        ELM_ADD_INPUT_PAIR(detail::BASE_SINGLE_INPUT_FEATURE_LAYER__KEY_INPUT_STIMULUS)
        ELM_ADD_OUTPUT_PAIR(detail::BASE_MATOUTPUT_LAYER__KEY_OUTPUT_RESPONSE)
        ;

MapAreaFilter::~MapAreaFilter()
{
}

MapAreaFilter::MapAreaFilter()
    : base_FeatureTransformationLayer()
{
    Clear();
}

MapAreaFilter::MapAreaFilter(const LayerConfig &config)
    : base_FeatureTransformationLayer(config)
{
    Clear();
    Reconfigure(config);
    IONames(config);
}

void MapAreaFilter::Clear()
{
    m_ = Mat1f();
}

void MapAreaFilter::Reset(const LayerConfig &config)
{
    Reconfigure(config);
}

void MapAreaFilter::Reconfigure(const LayerConfig &config)
{
    PTree p = config.Params();
    tau_size_ = p.get<int>(PARAM_TAU_SIZE, DEFAULT_TAU_SIZE);
}

cv::Mat1f sum_pixels(const cv::Mat1f& img, const cv::Mat1b &mask)
{
    return cv::Mat1f(1, 1, static_cast<float>(cv::countNonZero(mask)));
}

void MapAreaFilter::Activate(const Signal &signal)
{
    Mat1f map = signal.MostRecent(name_input_); // weighted gradient after thresholding

    GraphAttr seg_graph(map, map > DEFAULT_LABEL_UNASSIGNED);

    ELM_COUT_VAR(elm::to_string(seg_graph.VerticesIds()));


    VecMat1f x = seg_graph.applyVerticesToMap(sum_pixels);

    ELM_COUT_VAR(elm::Reshape(x));


    Mat1f adj;
    seg_graph.AdjacencyMat(adj);



    m_ = map;
}
