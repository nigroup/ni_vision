#include "ni/layers/mapneighadjacency.h"

#include "ni/layers/depthsegmentation.h"

#include "elm/core/exception.h"
#include "elm/core/graph/graphmap.h"
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
const string MapNeighAdjacency::PARAM_TAU_SIZE = "tau_size";

// defaults
const int MapNeighAdjacency::DEFAULT_TAU_SIZE           = 200;
const int MapNeighAdjacency::DEFAULT_LABEL_UNASSIGNED   = 0;

/** @todo why does define guard lead to undefined reference error?
 */
//#ifdef __WITH_GTEST
#include <boost/assign/list_of.hpp>
template <>
elm::MapIONames LayerAttr_<MapNeighAdjacency>::io_pairs = boost::assign::map_list_of
        ELM_ADD_INPUT_PAIR(detail::BASE_SINGLE_INPUT_FEATURE_LAYER__KEY_INPUT_STIMULUS)
        ELM_ADD_OUTPUT_PAIR(detail::BASE_MATOUTPUT_LAYER__KEY_OUTPUT_RESPONSE)
        ;
//#endif // __WITH_GTEST

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
    PTree p = config.Params();
    tau_size_ = p.get<int>(PARAM_TAU_SIZE, DEFAULT_TAU_SIZE);
}

void MapNeighAdjacency::Activate(const Signal &signal)
{
    Mat1f map = signal.MostRecent(name_input_); // weighted gradient after thresholding

    GraphMap seg_graph(map, map > DEFAULT_LABEL_UNASSIGNED);

    seg_graph.AdjacencyMat(m_);
}
