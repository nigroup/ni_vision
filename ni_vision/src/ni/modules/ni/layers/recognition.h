#ifndef _NI_LAYERS_RECOGNITION_H_
#define _NI_LAYERS_RECOGNITION_H_

#include <vector>

#include <flann/flann.h>

#include "elm/core/typedefs_fwd.h"
#include "elm/core/pcl/typedefs_fwd.h"
#include "elm/layers/layers_interim/base_matoutputlayer.h"

#include "ni/core/surface.h"
#include "ni/legacy/surfprop.h"
#include "ni/legacy/trackprop.h"

namespace ni {

/**
 * @brief layer for comparing a selected surface
 * with the trained model of an object
 * @cite Mohr2014
 *
 * Output keys defined by parent
 */
class Recognition : public elm::base_Layer
{
public:
    // params
    static const std::string PARAM_HIST_BINS;
    static const std::string PARAM_SIZE_MAX;        ///< upper threshold for size (cube diagonal) [mm]
    static const std::string PARAM_SIZE_MIN;        ///< lower threshold for size (cube diagonal) [mm]
    static const std::string PARAM_PTS_MIN;         ///< lower threshold for area [pixels]
    static const std::string PARAM_PATH_COLOR;      ///< path to library file
    static const std::string PARAM_PATH_SIFT;       ///< path to library file

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

    // members
    std::string input_name_bgr_;
    std::string input_name_selHistogram_;
    std::string input_name_selBoundingBox_;
    std::string name_out_rect_;
    std::string name_out_matchFlag_;

    int nb_bins_;           ///< no. of bins in color histogram

    // legacy members
    int nAttSizeMax;        ///< upper threshold for size (cube diagonal) [mm]
    int nAttSizeMin;        ///< lower threshold for size (cube diagonal) [mm]
    int nAttPtsMin;         ///< lower threshold for area [pixels]
    std::vector<elm::VecF > mnColorHistY_lib;
    // Todo
    std::vector<elm::VecF > mnSiftFeature_lib;
    int nFlannLibCols_sift;
    FLANNParameters FLANNParam;
    float* nFlannDataset;
    std::vector <std::vector <float> > mnSiftExtraFeatures;
    flann_index_t FlannIdx_Sift;

    SurfProp stMems;
    TrackProp stTrack;
};

} // namespace ni

#endif // _NI_LAYERS_RECOGNITION_H_
