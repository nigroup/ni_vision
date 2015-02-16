#ifndef _NI_LAYERS_DEPTHMAP_H_
#define _NI_LAYERS_DEPTHMAP_H_

#include "elm/layers/base_layer_derivations/base_featuretransformationlayer.h"

namespace ni {

/** class to implement depth map extraction
  *
  */
class DepthMap : public elm::base_FeatureTransformationLayer
{
public:
    virtual void Clear();

    virtual void Reconfigure(const elm::LayerConfig &config);

    virtual void Reset(const elm::LayerConfig &config);

    virtual void Activate(const elm::Signal &signal);

    /** Default constructor, still requires configurations
      * \see Reconfigure
      */
    DepthMap();

    /** Constructor with configuration
      * @param layer configuration
      */
    DepthMap(const elm::LayerConfig& config);

public:

protected:
};

} // namespace ni

#endif // _NI_LAYERS_DEPTHMAP_H_
