#ifndef _NI_LAYERS_MAPAREAFILTER_H_
#define _NI_LAYERS_MAPAREAFILTER_H_

#include "elm/layers/layers_interim/base_featuretransformationlayer.h"

#include "elm/core/typedefs_sfwd.h"

namespace elm {

class GraphAttr;

} // namespace elm

namespace ni {

class Surface;

/**
 * @brief Layer class for small map areas and merging them with largest neighbor
 * I/O keys defined by parent class
 *
 * all map segments with size <= tau will be merged with smaller and largest neighbor
 */
class MapAreaFilter : public elm::base_FeatureTransformationLayer
{
public:
    // parameters
    static const std::string PARAM_TAU_SIZE;    ///< threshold for size differences

    // defaults
    static const int DEFAULT_TAU_SIZE;          ///< = 200 pixels

    static const int DEFAULT_LABEL_UNASSIGNED;  ///< =0 value for labeling elements not assigned to any segment

    ~MapAreaFilter();

    MapAreaFilter();

    MapAreaFilter(const elm::LayerConfig &config);

    void Clear();

    void Reset(const elm::LayerConfig &config);

    void Reconfigure(const elm::LayerConfig &config);

    void Activate(const elm::Signal &signal);

protected:
    int getNeighbors(float vtx_id, const elm::GraphAttr &seg_graph, std::vector<Surface> &neighbors, elm::VecF &neigh_sizes) const;

    // members
    int tau_size_;
};

} // namespace ni

#endif // _NI_LAYERS_MAPAREAFILTER_H_

