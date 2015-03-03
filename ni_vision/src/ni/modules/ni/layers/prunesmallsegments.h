#ifndef _NI_LAYERS_PRUNESMALLSEGMENTS_H_
#define _NI_LAYERS_PRUNESMALLSEGMENTS_H_

#include "elm/layers/layers_interim/base_featuretransformationlayer.h"

namespace ni {

/**
 * @brief layer for pruning small segments
 * Reassigns their id to unassigned
 *
 * IO keys defined by parent
 */
class PruneSmallSegments :
        public elm::base_FeatureTransformationLayer
{
public:
    // params
    static const std::string PARAM_MIN_SIZE; ///< min valid size (inclusive), if size >= t then keep [pixels]

    PruneSmallSegments();

    PruneSmallSegments(const elm::LayerConfig &config);

    void Clear();

    void Reset(const elm::LayerConfig &config);

    void Reconfigure(const elm::LayerConfig &config);

    void Activate(const elm::Signal &signal);

protected:
    // members
    float min_;
};

} // namespace ni

#endif // _NI_LAYERS_PRUNESMALLSEGMENTS_H_
