#ifndef _NI_LAYERS_MAPNEIGHADJACENCY_H_
#define _NI_LAYERS_MAPNEIGHADJACENCY_H_

#include "elm/layers/base_layer_derivations/base_featuretransformationlayer.h"

namespace ni {

/**
 * @brief Layer class for generating neighborhood adjacency from a 2d map
 * I/O keys defined by parent class
 */
class MapNeighAdjacency : public elm::base_FeatureTransformationLayer
{
public:
    // parameters
    static const std::string PARAM_TAU_SIZE;    ///< threshold for size differences

    // defaults
    static const int DEFAULT_TAU_SIZE;      ///< = 200 pixels

    static const int DEFAULT_LABEL_UNASSIGNED; ///< =0 value for labeling elements not assigned to any segment

    MapNeighAdjacency();

    MapNeighAdjacency(const elm::LayerConfig &config);

    void Clear();

    void Reset(const elm::LayerConfig &config);

    void Reconfigure(const elm::LayerConfig &config);

    void Activate(const elm::Signal &signal);

protected:

    // members
    int tau_size_;
};

} // namespace ni

#endif // _NI_LAYERS_MAPNEIGHADJACENCY_H_
