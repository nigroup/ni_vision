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

const string Recognition::KEY_INPUT_BGR_IMAGE   = "bgr";
const string Recognition::KEY_INPUT_CLOUD       = "points";
const string Recognition::KEY_INPUT_MAP         = "map";
const string Recognition::KEY_INPUT_RECT = "rect";
const string Recognition::KEY_INPUT_HISTOGRAM = "hist";
const string Recognition::KEY_OUTPUT_MATCH_FLAG = "match";


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
    // todo
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
}


void Recognition::Activate(const Signal &signal)
{
    Mat3f color         = signal.MostRecent(input_name_bgr_).get<Mat1f>();
    Mat1f selectedHistogram = signal.MostRecent(input_name_selHistogram_).get<Mat1f>();
    Mat1f selectedBoundingBox = signal.MostRecent(input_name_selBoundingBox_).get<Mat1f>();


    // Color histogram difference
    float colorDistance = 0;
    for (int j = 0; j < selectedHistogram.cols; j++) {
        printf("%i\n",selectedHistogram.cols);
        colorDistance += fabs(mnColorHistY_lib[0][j] - selectedHistogram(j));
    }
    colorDistance = colorDistance / 2.f;


    // SIFT feature comparison
    Keypoint keypts;
    // @todo: use values from the gui
    float siftScales = 3;
    float siftInitSigma = 1.6;
    float siftPeakThrs = 0.01;

    int nCandRW = selectedBoundingBox(2) - selectedBoundingBox(0) + 1;
    int nCandRH = selectedBoundingBox(3) - selectedBoundingBox(1) + 1;

    GetSiftKeypoints(color, siftScales, siftInitSigma, siftPeakThrs,
                     selectedBoundingBox(0), selectedBoundingBox(1),
                     nCandRW, nCandRH,
                     keypts);

    // todo: use values from the gui
    float flannKnn = 2;
    float flannMatchFac = 0.7;
    // @todo: find the right way to determine number of columns

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
                       flannKnn,
                       nFlannLibCols_sift,
                       flannMatchFac,
                       mnSiftExtraFeatures,
                       FlannIdx_Sift,
                       FLANNParam,
                       keypts,
                       keyptsCnt,
                       flannIM,
                       vnSiftMatched,
                       vnDeltaScale,
                       vnDeltaOri,
                       nMaxDeltaOri,
                       nMinDeltaOri,
                       nMaxDeltaScale,
                       nMinDeltaScale);

    // @todo: filtering out false-positive keypoints

    // todo: extract thresholds from gui
    int siftCntThreshold = 10;
    float colorThreshold = 0.3;

    printf("%i %i %f %f\n",keyptsCnt, siftCntThreshold, colorDistance, colorThreshold);
    // todo: (siftfeature + matched_siftfeature)
    if (keyptsCnt >= siftCntThreshold && colorDistance < colorThreshold) {
        matchFlag_ = 1;
    } else {
        matchFlag_ = 0;
    }
}

void Recognition::Response(Signal &signal)
{
    signal.Append(name_out_matchFlag_, matchFlag_);
}
