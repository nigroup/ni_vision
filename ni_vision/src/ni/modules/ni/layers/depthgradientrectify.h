#ifndef _NI_LAYERS_DEPTHGRADIENTRECTIFY_H_
#define _NI_LAYERS_DEPTHGRADIENTRECTIFY_H_

#include "elm/layers/layers_interim/base_matoutputlayer.h"

namespace ni {

/**
 * @brief Layer to rectify processed depth gradient
 *
 * Output keys defined by parent class
 */
class DepthGradientRectify : public elm::base_MatOutputLayer
{
public:
    // parameters
    static const std::string PARAM_MAX_GRAD;    ///< threshold for segmentation (if depth-gradient between two pixel is >= than threshold, they get segmented)

    // Input keys
    static const std::string KEY_INPUT_GRAD_X;  ///< key to raw gradient along x axis (horizontal)
    static const std::string KEY_INPUT_GRAD_Y;  ///< key to raw gradient along y axis (vertical)
    static const std::string KEY_INPUT_GRAD_SMOOTH;  ///< key to processed gradient component (i.e after smoothing)

    DepthGradientRectify();

    void Clear();

    void Reset(const elm::LayerConfig &config);

    void Reconfigure(const elm::LayerConfig &config);

    void InputNames(const elm::LayerInputNames &io);

    void Activate(const elm::Signal &signal);

protected:
    std::string name_in_grad_x_;
    std::string name_in_grad_y_;
    std::string name_in_grad_smooth_;

    float max_;
};

} // namespace ni

#endif // _NI_LAYERS_DEPTHGRADIENTRECTIFY_H_
