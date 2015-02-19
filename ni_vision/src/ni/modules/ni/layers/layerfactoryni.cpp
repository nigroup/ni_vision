#include "ni/layers/layerfactoryni.h"

#include <boost/assign/list_of.hpp>

#include "elm/core/registor.h"

/** Whenever a new layer is imeplemented:
 *  1. include its header below
 *  2. Add it to the initialization of g_layerRegistry map.
 */
#include "elm/layers/attentionwindow.h"
#include "elm/layers/gradassignment.h"
#include "elm/layers/graphcompatibility.h"
#include "elm/layers/icp.h"
#include "elm/layers/layer_y.h"
#include "elm/layers/medianblur.h"
#include "elm/encoding/populationcode.h"
#include "elm/layers/saliencyitti.h"
#include "elm/layers/sinkhornbalancing.h"
#include "elm/layers/triangulation.h"
#include "elm/layers/weightedsum.h"

#include "ni/layers/depthgradient.h"
#include "ni/layers/depthmap.h"
#include "ni/layers/depthsegmentation.h"


using boost::assign::map_list_of;
using namespace elm;
using namespace ni;

typedef Registor_<base_Layer> LayerRegistor;
typedef Registor_<base_Layer>::Registry LayerRegistry;

/** Macros for creating individual registry pair items
 *  credit: J. Turcot, T. Senechal, http://stackoverflow.com/questions/138600/initializing-a-static-stdmapint-int-in-c
 */
#define REGISTRY_PAIR(Registor, NewInstance) (#NewInstance, &Registor::DerivedInstance<NewInstance>)
#define LAYER_REGISTRY_PAIR(NewInstance) REGISTRY_PAIR(LayerRegistor, NewInstance)

LayerRegistry g_layerRegistry = map_list_of
        LAYER_REGISTRY_PAIR( AttentionWindow )
        LAYER_REGISTRY_PAIR( DepthGradient )
        LAYER_REGISTRY_PAIR( DepthMap )
        LAYER_REGISTRY_PAIR( DepthSegmentation )
        LAYER_REGISTRY_PAIR( GradAssignment )
        LAYER_REGISTRY_PAIR( GraphCompatibility )
        LAYER_REGISTRY_PAIR( ICP )
        LAYER_REGISTRY_PAIR( LayerY )
        LAYER_REGISTRY_PAIR( MedianBlur )
        LAYER_REGISTRY_PAIR( MutexPopulationCode )
        LAYER_REGISTRY_PAIR( SaliencyItti )
        LAYER_REGISTRY_PAIR( SinkhornBalancing )
        LAYER_REGISTRY_PAIR( Triangulation )
        LAYER_REGISTRY_PAIR( WeightedSum )
        ; ///< <-- add new layer to registry here

LayerFactoryNI::LayerFactoryNI()
{
}

LayerRegistor::RegisteredTypeSharedPtr LayerFactoryNI::CreateShared(const LayerType &type)
{
    static_assert(std::is_same<LayerRegistor::RegisteredTypeSharedPtr, LayerShared >(), "Mismatching shared_ptr types.");
    return LayerRegistor::CreatePtrShared(g_layerRegistry, type);
}

LayerRegistor::RegisteredTypeSharedPtr LayerFactoryNI::CreateShared(const LayerType &type,
                                                                    const LayerConfig &config,
                                                                    const LayerIONames &io)
{
    LayerRegistor::RegisteredTypeSharedPtr ptr = LayerFactoryNI::CreateShared(type);
    ptr->Reset(config);
    ptr->IONames(io);
    return ptr;
}
