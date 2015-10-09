#ifndef _NI_LAYERS_SURFACETRACKING_H_
#define _NI_LAYERS_SURFACETRACKING_H_

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
class SurfaceTracking : public elm::base_Layer
{
public:
    // params
    static const std::string PARAM_HIST_BINS;       ///< no. of bins in color histogram
    static const std::string PARAM_WEIGHT_COLOR;    ///< color contribution
    static const std::string PARAM_WEIGHT_POS;      ///< position contribution
    static const std::string PARAM_WEIGHT_SIZE;     ///< size contribution
    static const std::string PARAM_MAX_COLOR;       ///< upper threshold for similarly colored surfaces
    static const std::string PARAM_MAX_POS;         ///< upper threshold for similarly positioned surfaces
    static const std::string PARAM_MAX_SIZE;        ///< upper threshold for similarly sized surfaces
    static const std::string PARAM_MAX_DIST;        ///< upper threshold for total feature distance

    // remaining I/O keys
    static const std::string KEY_INPUT_BGR_IMAGE;
    static const std::string KEY_INPUT_CLOUD;
    static const std::string KEY_INPUT_MAP;
    static const std::string KEY_OUTPUT_TRACK_MAP;
    static const std::string KEY_OUTPUT_BOUNDING_BOXES;

    static const float DISTANCE_HUGE;

    virtual ~SurfaceTracking();

    SurfaceTracking();

    void Clear();

    void Reset(const elm::LayerConfig &config);

    void Reconfigure(const elm::LayerConfig &config);

    void InputNames(const elm::LayerInputNames &io);

    void OutputNames(const elm::LayerOutputNames &io);

    void Activate(const elm::Signal &signal);

    void Response(elm::Signal &signal);

protected:
    void extractFeatures(const elm::CloudXYZPtr &cloud,
                         const cv::Mat &bgr,
                         const cv::Mat1f &map,
                         std::vector<Surface> &surfaces);

    void computeFeatureDistance(const std::vector<ni::Surface> &surfaces,
                                const std::vector<ni::Surface> &mem);

    // members
    std::string input_name_bgr_;
    std::string input_name_cloud_;
    std::string input_name_map_;
    std::string name_out_trackMap_;
    std::string name_out_boundingBoxes_;


    int nb_bins_;           ///< no. of bins in color histogram

    float weight_color_;    ///< weight for distance between color histograms features h
    float weight_pos_;      ///< weight for distance between observed and tracked surface positions
    float weight_size_;     ///< weight for size distance

    float max_color_;       ///< upper threshold for similarly colored surfaces
    float max_pos_;         ///< upper threshold for similarly positioned surfaces
    float max_size_;        ///< upper threshold for similarly sized surfaces
    float max_dist_;        ///< upper threshold for total feature distance

    cv::Mat1f dist_color_;  ///< distance matrix for color features
    cv::Mat1f dist_pos_;    ///< distance matrix for position
    cv::Mat1f dist_size_;   ///< distance matrix for size
    cv::Mat1f dist_;        ///< weighted sum of feature distance matrices
    cv::Mat1f boundingBoxes_; ///< matrix with coordinates and indices of the bounding boxes
    cv::Mat1f trackMap_;

    std::vector<ni::Surface> obsereved_;
    std::vector<ni::Surface> memory_;

    // legacy members
    SurfProp stMems;
    int nMemsCnt;
    TrackProp stTrack;
    VecI vnMemsValidIdx;
    std::vector<elm::VecF > mnMemsRelPose;
    int framec;
};

} // namespace ni

#endif // _NI_LAYERS_SURFACETRACKING_H_
