#ifndef _NI_LAYERS_DEPTHSEGMENTATION_H_
#define _NI_LAYERS_DEPTHSEGMENTATION_H_

#include "elm/layers/base_layer_derivations/base_featuretransformationlayer.h"

namespace ni {

/**
 * @brief Layer class for segmenting scene into surfaces
 *
 * I/O keys defined by parent class
 *
 * @cite Mohr2014 (Section 2.2  Segmentation of Surface)
 */
class DepthSegmentation : public elm::base_FeatureTransformationLayer
{
public:
    // parameters
    static const std::string PARAM_TAU_SIZE;    ///< threshold for size differences
    static const std::string PARAM_MAX_GRAD;    ///< threshold for segmentation (if depth-gradient between two pixel is >= than threshold, they get segmented)

    // defaults
    static const int DEFAULT_TAU_SIZE;      ///< = 200 @todo units?
    static const float DEFAULT_MAX_GRAD;    ///< = 0.005f;

    DepthSegmentation();

    DepthSegmentation(const elm::LayerConfig &config);

    void Clear();

    void Reset(const elm::LayerConfig &config);

    void Reconfigure(const elm::LayerConfig &config);

    void Activate(const elm::Signal &signal);

protected:

    bool comparePixels(float current, float neighbor);

    /**
     * @brief group pixels into surfaces based on depth discontinuities
     * @param g weighted gradient after thresholding
     */
    void group(const cv::Mat1f g);

    // members
    int tau_size_;
    float max_grad_;
};

} // namespace ni

#endif // _NI_LAYERS_DEPTHSEGMENTATION_H_
