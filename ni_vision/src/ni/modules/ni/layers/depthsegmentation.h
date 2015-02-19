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

    /**
     * @brief compare pixels for merging them into same surface or keeping them separate
     * @param current pixel value
     * @param neighbor neighboring pixel value (e.g. 1 row above, column to the left)
     * @return true on match to merge, false to instruct keep apart
     */
    bool comparePixels(float current, float neighbor) const;

    /**
     * @brief group pixels into surfaces based on depth discontinuities
     * @param g weighted gradient after thresholding
     * @return map of surface labels (intensity value represents surface assignment ofr that element)
     */
    cv::Mat1i group(const cv::Mat1f &g) const;



    // members
    int tau_size_;
    float max_grad_;
};

} // namespace ni

#endif // _NI_LAYERS_DEPTHSEGMENTATION_H_
