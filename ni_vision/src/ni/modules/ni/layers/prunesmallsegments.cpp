#include "ni/layers/prunesmallsegments.h"

#include "elm/core/cv/mat_utils.h"
#include "elm/core/cv/mat_vector_utils.h"
#include "elm/core/exception.h"
#include "elm/core/graph/graphattr.h"
#include "elm/core/layerconfig.h"
#include "elm/core/signal.h"
#include "elm/ts/layerattr_.h"

using namespace cv;
using namespace elm;
using namespace ni;

const std::string PruneSmallSegments::PARAM_MIN_SIZE = "min";

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

cv::Mat1f sum_pixels2(const cv::Mat1f& img, const cv::Mat1b &mask)
{
    return cv::Mat1f(1, 1, static_cast<float>(cv::countNonZero(mask)));
}

cv::Mat1f mask_vertex2(const cv::Mat1f& img, const cv::Mat1b &mask)
{
    Mat1b mask_inverted;
    cv::bitwise_not(mask, mask_inverted);
    return img.clone().setTo(0, mask_inverted);
}

void PruneSmallSegments::Activate(const Signal &signal)
{
    Mat1f map = signal.MostRecent(name_input_); // weighted gradient after thresholding

    GraphAttr seg_graph(map.clone(), map > 0.f);

    VecF seg_ids = seg_graph.VerticesIds();
    Mat1f seg_sizes = elm::Reshape(seg_graph.applyVerticesToMap(sum_pixels2));

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
