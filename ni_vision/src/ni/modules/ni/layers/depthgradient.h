#ifndef _NI_LAYERS_DEPTHGRADIENT_H_
#define _NI_LAYERS_DEPTHGRADIENT_H_

#include <opencv2/core/core.hpp>

#include "elm/layers/base_layer_derivations/base_singleinputfeaturelayer.h"

namespace ni {

/** @brief class to implement depth gradient
 *
 * Input name already defined by parent
 */
class DepthGradient : public elm::base_SingleInputFeatureLayer
{
public:
    // paramters
    static const std::string PARAM_GRAD_WEIGHT; ///< constant of the weighted depth
    static const std::string PARAM_GRAD_MAX;    ///< threshold for very steep depth-gradient

    // defaults
    static const float DEFAULT_GRAD_WEIGHT;     ///< 0.2
    static const float DEFAULT_GRAD_MAX;        ///< 0.04

    // I/O keys
    static const std::string KEY_OUTPUT_GRAD_X; ///< output key for gradient in x (horizontal) direction
    static const std::string KEY_OUTPUT_GRAD_Y; ///< output key gradient in y (vertical) direction

    virtual void Clear();

    virtual void Reconfigure(const elm::LayerConfig &config);

    virtual void Reset(const elm::LayerConfig &config);

    virtual void OutputNames(const elm::LayerOutputNames &io);

    virtual void Activate(const elm::Signal &signal);

    virtual void Response(elm::Signal &signal);

    /** Default constructor, still requires configurations
      * \see Reconfigure
      */
    DepthGradient();

    /** Constructor with configuration
      * @param layer configuration
      */
    DepthGradient(const elm::LayerConfig& config);

protected:
    /**
     * @brief compute derivative of an image along a given dimension
     * using forward difference.
     *
     * @param src image
     * @param dim dimension (0 for horizontal derivative, 1 for vertical)
     * @param dst destination
     */
    virtual void computeDerivative(const cv::Mat1f &src, int dim, cv::Mat1f &dst) const;

    // members
    std::string name_out_grad_x_;
    std::string name_out_grad_y_;

    cv::Mat1f grad_x_;          ///< gradient in x (horizontal) direction
    cv::Mat1f grad_y_;          ///< gradient in y (vertical) direction

    float max_; ///< upper threshold for gradient values in either direction
    float w_;   ///< gradient weight
};

} // namespace ni

#endif // _NI_LAYERS_DEPTHGRADIENT_H_
