#ifndef _NI_LAYERS_ATTENTION_H_
#define _NI_LAYERS_ATTENTION_H_

#include <vector>

#include "elm/core/typedefs_fwd.h"
#include "elm/core/pcl/typedefs_fwd.h"
#include "elm/layers/layers_interim/base_matoutputlayer.h"

#include "ni/core/surface.h"
#include "ni/legacy/surfprop.h"
#include "ni/legacy/trackprop.h"

namespace ni {

/**
 * @brief layer for tracking surfaces over time,
 * by assigning newly observed surfaces to surfaces stored
 * in short-term memory (STM)
 * @cite Mohr2014
 *
 * Output keys defined by parent
 */
class Attention : public elm::base_MatOutputLayer
{
public:
    // params
    static const std::string PARAM_HIST_BINS;
    static const std::string PARAM_WEIGHT_COLOR;
    static const std::string PARAM_WEIGHT_POS;
    static const std::string PARAM_WEIGHT_SIZE;
    static const std::string PARAM_MAX_COLOR;       ///< upper threshold for similarly colored surfaces
    static const std::string PARAM_MAX_POS;         ///< upper threshold for similarly positioned surfaces
    static const std::string PARAM_MAX_SIZE;        ///< upper threshold for similarly sized surfaces
    static const std::string PARAM_MAX_DIST;        ///< upper threshold for total feature distance

    // remaining I/O keys
    static const std::string KEY_INPUT_BGR_IMAGE;
    static const std::string KEY_INPUT_CLOUD;
    static const std::string KEY_INPUT_MAP;

    static const float DISTANCE_HUGE;

    virtual ~Attention();

    Attention();

    void Clear();

    void Reset(const elm::LayerConfig &config);

    void Reconfigure(const elm::LayerConfig &config);

    void InputNames(const elm::LayerInputNames &io);

    void Activate(const elm::Signal &signal);

protected:
    void extractFeatures(const elm::CloudXYZPtr &cloud,
                         const cv::Mat &bgr,
                         const cv::Mat1f &map,
                         std::vector<Surface> &surfaces);

    void computeFeatureDistance(const std::vector<ni::Surface> &surfaces,
                                const std::vector<ni::Surface> &mem);

    void SurfPropToVecSurfaces(const SurfProp &surf_prop, std::vector<ni::Surface> &surfaces) const;
    void VecSurfacesToSurfProp(const std::vector<ni::Surface> &surfaces, SurfProp &surf_prop) const;

    // members
    std::string input_name_bgr_;
    std::string input_name_cloud_;
    std::string input_name_map_;

    int nb_bins_;           ///< no. of bins in color histogram

    cv::Mat1f dist_;        ///< weighted sum of feature distance matrices

    std::vector<ni::Surface> observed_;

    // legacy members
    SurfProp stMems;
    int nMemsCnt;
    TrackProp stTrack;
    VecI vnMemsValidIdx;
    std::vector<elm::VecF > mnMemsRelPose;
    int framec;
};

} // namespace ni

#endif // _NI_LAYERS_ATTENTION_H_
