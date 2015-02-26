#include "ni/layers/mapneighadjacency.h"

#include "ni/layers/depthsegmentation.h"

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

// defaults
const int MapNeighAdjacency::DEFAULT_LABEL_UNASSIGNED   = 0;

#include <boost/assign/list_of.hpp>
template <>
elm::MapIONames LayerAttr_<MapNeighAdjacency>::io_pairs = boost::assign::map_list_of
        ELM_ADD_INPUT_PAIR(detail::BASE_SINGLE_INPUT_FEATURE_LAYER__KEY_INPUT_STIMULUS)
        ELM_ADD_OUTPUT_PAIR(detail::BASE_MATOUTPUT_LAYER__KEY_OUTPUT_RESPONSE)
        ;

MapNeighAdjacency::~MapNeighAdjacency()
{
}

MapNeighAdjacency::MapNeighAdjacency()
    : base_FeatureTransformationLayer()
{
    Clear();
}

MapNeighAdjacency::MapNeighAdjacency(const LayerConfig &config)
    : base_FeatureTransformationLayer(config)
{
    Clear();
    Reconfigure(config);
    IONames(config);
}

void MapNeighAdjacency::Clear()
{
    m_ = Mat1f();
}

void MapNeighAdjacency::Reset(const LayerConfig &config)
{
    Reconfigure(config);
}

void MapNeighAdjacency::Reconfigure(const LayerConfig &config)
{
    if(config.Params().size() > 0) {

        ELM_THROW_KEY_ERROR("MapNeighAdjacency does not expect any paramters.");
    }
}

void MapNeighAdjacency::Activate(const Signal &signal)
{
    Mat1f map = signal.MostRecent(name_input_); // weighted gradient after thresholding

    GraphAttr seg_graph(map, map > DEFAULT_LABEL_UNASSIGNED);

    seg_graph.AdjacencyMat(m_);
}
