#include "ni/layers/recognition.h"

#include <opencv2/highgui/highgui.hpp>
#include "elm/core/debug_utils.h"

#include <set>

#include <boost/filesystem.hpp>

#include "elm/core/cv/mat_vector_utils.h"
#include "elm/core/cv/mat_vector_utils_inl.h"
#include "elm/core/exception.h"
#include "elm/core/featuredata.h"
#include "elm/core/inputname.h"
#include "elm/core/layerconfig.h"
#include "elm/core/layerinputnames.h"
#include "elm/core/signal.h"
#include "elm/ts/layerattr_.h"

#include "ni/core/boundingbox2d.h"
#include "ni/core/boundingbox3d.h"
#include "ni/core/colorhistogram.h"

#include "ni/legacy/func_init.h"
#include "ni/legacy/func_recognition.h"
#include "ni/legacy/surfprop_utils.h"

using namespace std;
namespace bfs=boost::filesystem;
using namespace cv;
using namespace elm;
using namespace ni;

const string Recognition::PARAM_PATH_COLOR      = "path_color";
const string Recognition::PARAM_PATH_SIFT       = "path_sift";
const string Recognition::PARAM_COLOR_THRESHOLD    = "color_threshold";
const string Recognition::PARAM_SIFT_THRESHOLD = "sift_threshold";
const string Recognition::PARAM_SIFT_SCALES = "sift_scales";
const string Recognition::PARAM_SIFT_INIT_SIGMA = "sift_init_sigma";
const string Recognition::PARAM_SIFT_PEAK_THRESHOLD = "sift_peak_threshold";
const string Recognition::PARAM_FLANN_KNN = "flann_knn";
const string Recognition::PARAM_FLANN_MATCH_FACTOR = "flann_match_factor";

const string Recognition::KEY_INPUT_BGR_IMAGE   = "bgr";
const string Recognition::KEY_INPUT_CLOUD       = "points";
const string Recognition::KEY_INPUT_MAP         = "map";
const string Recognition::KEY_INPUT_RECT = "rect";
const string Recognition::KEY_INPUT_HISTOGRAM = "hist";
const string Recognition::KEY_OUTPUT_MATCH_FLAG = "match";
const string Recognition::KEY_OUTPUT_KEYPOINTS = "keys";
const string Recognition::KEY_OUTPUT_MATCHED_KEYPOINTS = "matched_keys";

const float Recognition::DISTANCE_HUGE = 100.f;


Recognition::~Recognition()
{
}

Recognition::Recognition()
    : elm::base_Layer()
{
    Clear();
}

void Recognition::Clear()
{
}

void Recognition::Reset(const LayerConfig &config)
{
    PTree p = config.Params();

    bfs::path path_color = p.get<bfs::path>(PARAM_PATH_COLOR);

    if(!bfs::is_regular_file(path_color)) {

        stringstream s;
        s << "Invalid path to color model for recognition layer (" << path_color << ")";
        ELM_THROW_FILEIO_ERROR(s.str());
    }

    bfs::path path_sift = p.get<bfs::path>(PARAM_PATH_SIFT);

    if(!bfs::is_regular_file(path_sift)) {

        stringstream s;
        s << "Invalid path to sift model for recognition layer (" << path_sift << ")";
        ELM_THROW_FILEIO_ERROR(s.str());
    }

    mnColorHistY_lib.clear();
    mnSiftExtraFeatures.clear();

    int nRecogFeature = 20;
    BuildFlannIndex(1, path_color.string(),
                    mnColorHistY_lib,
                    stTrack,
                    nFlannLibCols_sift,
                    FLANNParam,
                    nFlannDataset,
                    nRecogFeature,
                    mnSiftExtraFeatures,
                    FlannIdx_Sift);
    BuildFlannIndex(2, path_sift.string(),
                    mnColorHistY_lib,
                    stTrack,
                    nFlannLibCols_sift,
                    FLANNParam,
                    nFlannDataset,
                    nRecogFeature,
                    mnSiftExtraFeatures,
                    FlannIdx_Sift);

    Reconfigure(config);
}


void Recognition::Reconfigure(const LayerConfig &config)
{
    PTree p = config.Params();
    colorThreshold_   = p.get<float>(PARAM_COLOR_THRESHOLD);
    siftCntThreshold_ = p.get<float>(PARAM_SIFT_THRESHOLD);
    siftScales_ = p.get<float>(PARAM_SIFT_SCALES);
    siftInitSigma_ = p.get<float>(PARAM_SIFT_INIT_SIGMA);
    siftPeakThrs_ = p.get<float>(PARAM_SIFT_PEAK_THRESHOLD);
    flannKnn_ = p.get<float>(PARAM_FLANN_KNN);
    flannMatchFac_ = p.get<float>(PARAM_FLANN_MATCH_FACTOR);
}


void Recognition::InputNames(const LayerInputNames &io)
{
    input_name_bgr_     = io.Input(KEY_INPUT_BGR_IMAGE);
    input_name_selHistogram_ = io.Input(KEY_INPUT_HISTOGRAM);
    input_name_selBoundingBox_ = io.Input(KEY_INPUT_RECT);
}


void Recognition::OutputNames(const LayerOutputNames &config)
{
    name_out_matchFlag_ = config.Output(KEY_OUTPUT_MATCH_FLAG);
    name_out_keypoints_ = config.Output(KEY_OUTPUT_KEYPOINTS);
    name_out_matchedKeypoints_ = config.Output(KEY_OUTPUT_MATCHED_KEYPOINTS);
}


void Recognition::Activate(const Signal &signal)
{
    Mat3f color         = signal.MostRecent(input_name_bgr_).get<Mat1f>();
    Mat1f selectedHistogram = signal.MostRecent(input_name_selHistogram_).get<Mat1f>();
    Mat1f selectedBoundingBox = signal.MostRecent(input_name_selBoundingBox_).get<Mat1f>();

    Mat bgr;
    color.convertTo(bgr, CV_8UC3);

    // Color histogram difference
    float colorDistance = 0;
    for (int j = 0; j < selectedHistogram.cols; j++) {
        //printf("%f %f\n", mnColorHistY_lib[0][j], selectedHistogram(j));
        colorDistance += fabs(mnColorHistY_lib[0][j] - selectedHistogram(j));
    }
    colorDistance = colorDistance / 2.f;
    printf("Hellpo World %f\n",colorThreshold_);
    if(colorDistance > colorThreshold_) {
        printf("%f\n", colorDistance);
        matchFlag_ = 0;

        keypoints_ = Mat1f();
        matchedKeypoints_ = Mat1f();
    }
    else {

        // SIFT feature comparison
        Keypoint keypts;

        int nCandRW = selectedBoundingBox(2) - selectedBoundingBox(0) + 1;
        int nCandRH = selectedBoundingBox(3) - selectedBoundingBox(1) + 1;

        // WARNING: QtCreator links this function to the wrong file
        // but the right one (in func_recognition.cpp) is compiled !!!
        GetSiftKeypoints(bgr, siftScales_, siftInitSigma_, siftPeakThrs_,
                         selectedBoundingBox(0), selectedBoundingBox(1),
                         nCandRW, nCandRH,
                         keypts);

        std::vector<int> vnSiftMatched;
        std::vector <double> vnDeltaScale;
        std::vector <double> vnDeltaOri;
        double nMaxDeltaOri = -999;
        double nMaxDeltaScale = -999;
        double nMinDeltaOri = 999;
        double nMinDeltaScale = 999;
        int keyptsCnt = 0;
        int flannIM = 0;
        // todo: correct value?
        int tcount = 1;

        Recognition_Flann (tcount,
                           flannKnn_,
                           nFlannLibCols_sift,
                           flannMatchFac_,
                           mnSiftExtraFeatures,
                           FlannIdx_Sift,
                           FLANNParam,
                           keypts, // keypoints of surface
                           keyptsCnt,
                           flannIM,
                           vnSiftMatched,
                           vnDeltaScale,
                           vnDeltaOri,
                           nMaxDeltaOri,
                           nMinDeltaOri,
                           nMaxDeltaScale,
                           nMinDeltaScale);


        // Filtering: extracting true-positives from matched keypoints
        double nDeltaScale=0;
        int flannTP=0;
        float T_orient = 0.5233333;
        float T_scale = 0.1;
        int nDeltaBinNo = 12;
    //    int tnumb = 3;

        printf("FlannIM %i\n",flannIM);
        std::vector<bool> vbSiftTP = std::vector<bool>(keyptsCnt, 0);
        if(flannIM > siftCntThreshold_) {
            CalcDeltaScaleOri(vnSiftMatched,
                              vnDeltaScale,
                              vnDeltaOri,
                              nDeltaScale,
                              nDeltaBinNo,
                              nMaxDeltaOri,
                              nMinDeltaOri,
                              T_orient,
                              T_scale,
                              flannTP,
                              vbSiftTP);
        } else {
            flannTP = 0;
        }

        printf("%i %i %i %f %f\n",keyptsCnt, flannTP,siftCntThreshold_, colorDistance, colorThreshold_);
        if (flannTP >= siftCntThreshold_) {
            matchFlag_ = 1;
        } else {
            matchFlag_ = 0;
        }

        // return siftkeypoints
        keypoints_ = Mat1f(keyptsCnt, 2);
        int counter = 0;
        while(keypts) {
            keypoints_(counter, 0) = keypts->row + selectedBoundingBox(0);
            keypoints_(counter, 1) = keypts->col + selectedBoundingBox(1);
            keypts = keypts->next;
            counter++;
        }
        matchedKeypoints_ = Mat1f(keyptsCnt, 1);
        for(int i = 0; i < keyptsCnt; i++) {
            matchedKeypoints_(i) = (int) vbSiftTP[i];
        }
    }
}

void Recognition::Response(Signal &signal)
{
    signal.Append(name_out_matchFlag_, matchFlag_);
    signal.Append(name_out_keypoints_, keypoints_);
    signal.Append(name_out_matchedKeypoints_, matchedKeypoints_);
}
