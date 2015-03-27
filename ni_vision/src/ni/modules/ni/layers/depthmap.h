#ifndef _NI_LAYERS_DEPTHMAP_H_
#define _NI_LAYERS_DEPTHMAP_H_

#include "elm/layers/layers_interim/base_featuretransformationlayer.h"

namespace ni {

/** @brief class to implement depth map extraction
 * layer IO names already defined in parent class:
 *
 * KEY_INPUT_STIMULUS: input point cloud
 * KEY_OUTPUT_RESPONSE: depth map matrix of all finite and valid depth (z) values
 *
 * @todo Check every z coordinate for isfinite()?
 */
class DepthMap : public elm::base_FeatureTransformationLayer
{
public:
    // layer parameters
    static const std::string PARAM_DEPTH_MAX;   ///< min. depth value

    // default values
    static const float DEFAULT_DEPTH_MAX;   ///< max. depth (5.)

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

protected:

    // members
    float depth_max_;   ///< min. allowable depth value

};

} // namespace ni

#endif // _NI_LAYERS_DEPTHMAP_H_
