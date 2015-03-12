#include "ni/layers/prunesmallsegments.h"

#include <boost/assign/list_of.hpp>

#include "elm/core/cv/mat_utils.h"
#include "elm/core/cv/mat_vector_utils.h"
#include "elm/core/exception.h"
#include "elm/core/graph/graphattr.h"
#include "elm/core/layerconfig.h"
#include "elm/core/signal.h"
#include "elm/ts/layerattr_.h"

#include "ni/core/graph/vertexopsegmentsize.h"

using namespace cv;
using namespace elm;
using namespace ni;

const std::string PruneSmallSegments::PARAM_MIN_SIZE = "min";

template <>
elm::MapIONames LayerAttr_<PruneSmallSegments>::io_pairs = boost::assign::map_list_of
        ELM_ADD_INPUT_PAIR(detail::BASE_SINGLE_INPUT_FEATURE_LAYER__KEY_INPUT_STIMULUS)
        ELM_ADD_OUTPUT_PAIR(detail::BASE_MATOUTPUT_LAYER__KEY_OUTPUT_RESPONSE)
        ;

PruneSmallSegments::PruneSmallSegments()
    : base_FeatureTransformationLayer()
{
}

PruneSmallSegments::PruneSmallSegments(const LayerConfig &config)
    : base_FeatureTransformationLayer(config)
{
}

void PruneSmallSegments::Clear()
{
    m_ = Mat1f();
}

void PruneSmallSegments::Reconfigure(const LayerConfig &config)
{
    min_ = config.Params().get<float>(PARAM_MIN_SIZE);
}

void PruneSmallSegments::Reset(const LayerConfig &config)
{
    Clear();
    Reconfigure(config);
}

cv::Mat1f mask_vertex2(const cv::Mat1i& img, const cv::Mat &mask)
{
    Mat mask_inverted;
    cv::bitwise_not(mask, mask_inverted);
    return img.clone().setTo(0, mask_inverted);
}

void PruneSmallSegments::Activate(const Signal &signal)
{
    Mat1f map = signal.MostRecent(name_input_); // weighted gradient after thresholding

    GraphAttr seg_graph(map.clone(), map > 0);

    VecI seg_ids = seg_graph.VerticesIds();
    Mat1f seg_sizes = elm::Reshape(seg_graph.applyVerticesToMap(VertexOpSegmentSize::calcSize));

    // assign size to vector attributes
    for(size_t i=0; i<seg_sizes.total(); i++) {

        if(seg_sizes(i) < min_) {

            seg_graph.removeVertex(seg_ids[i]);
        }
    }

    // replace with getter to Graph's underlying map image
    VecMat1f masked_maps = seg_graph.applyVerticesToMap(mask_vertex2);
    m_ = Mat1f::zeros(map.size());
    for(size_t i=0; i<masked_maps.size(); i++) {

        m_ += masked_maps[i];
    }
}
