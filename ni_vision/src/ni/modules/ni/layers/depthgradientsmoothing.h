#ifndef _NI_LAYERS_DEPTHGRADIENTSMOOTHING_H_
#define _NI_LAYERS_DEPTHGRADIENTSMOOTHING_H_

#include "elm/layers/layers_interim/base_smoothlayer.h"

namespace ni {

class DepthGradientSmoothing
        : public elm::base_SmoothLayer
{
public:
    static const std::string PARAM_MAX;
    static const std::string PARAM_FILTER_MODE;
    static const std::string PARAM_SMOOTH_MODE;
    static const std::string PARAM_SMOOTH_CENTER;
    static const std::string PARAM_BAND_1;
    static const std::string PARAM_BAND_2;
    static const std::string PARAM_SMOOTH_FACTOR;

    DepthGradientSmoothing();

    DepthGradientSmoothing(const elm::LayerConfig &cfg);

    void Reconfigure(const elm::LayerConfig &cfg);

    void Activate(const elm::Signal &signal);

protected:
    float max_;
    int filter_mode_;
    int smode_;
    int scenter_;
    int sband1_;
    int sband2_;
    float clfac_;
};

} // namespace ni

#endif // _NI_LAYERS_DEPTHGRADIENTSMOOTHING_H_
