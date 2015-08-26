#include "ni/layers/layerfactoryni.h"

#include <boost/assign/list_of.hpp>

#include "elm/core/registor.h"

/** Whenever a new layer is imeplemented:
 *  1. include its header below
 *  2. Add it to the initialization of g_layerRegistryNI map.
 */
#include "elm/layers/layerfactory.h"

#include "ni/layers/attention.h"
#include "ni/layers/mapareafilter.h"
#include "ni/layers/mapneighadjacency.h"
#include "ni/layers/depthgradient.h"
#include "ni/layers/depthgradientrectify.h"
#include "ni/layers/depthgradientsmoothing.h"
#include "ni/layers/depthmap.h"
#include "ni/layers/depthsegmentation.h"
#include "ni/layers/prunesmallsegments.h"
#include "ni/layers/recognition.h"
#include "ni/layers/surfacetracking.h"

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

LayerRegistry g_layerRegistryNI = map_list_of
        LAYER_REGISTRY_PAIR( Attention )
        LAYER_REGISTRY_PAIR( DepthGradient )
        LAYER_REGISTRY_PAIR( DepthGradientRectify )
        LAYER_REGISTRY_PAIR( DepthGradientSmoothing )
        LAYER_REGISTRY_PAIR( DepthMap )
        LAYER_REGISTRY_PAIR( DepthSegmentation )
        LAYER_REGISTRY_PAIR( MapAreaFilter )
        LAYER_REGISTRY_PAIR( MapNeighAdjacency )
        LAYER_REGISTRY_PAIR( PruneSmallSegments )
        LAYER_REGISTRY_PAIR( Recognition )
        LAYER_REGISTRY_PAIR( SurfaceTracking )
        ; ///< <-- add new layer to registry here

LayerFactoryNI::LayerFactoryNI()
{
}

LayerRegistor::RegisteredTypeSharedPtr LayerFactoryNI::CreateShared(const LayerType &type)
{
    static_assert(std::is_same<LayerRegistor::RegisteredTypeSharedPtr, LayerShared >(), "Mismatching shared_ptr types.");

    if(!LayerRegistor::Find(g_layerRegistryNI, type)) {

        return LayerFactory::CreateShared(type);
    }

    return LayerRegistor::CreatePtrShared(g_layerRegistryNI, type);
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
