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

Attention::~Attention()
{
}

Attention::Attention()
    : elm::base_MatOutputLayer()
{
    Clear();
}

void Attention::Clear()
{
}

void Attention::Reset(const LayerConfig &config)
{
    // todo
    // @todo: load the histogram and sift feature from the learned model
    BuildFlannIndex(1, sColorLibFileName,
                    mnColorHistY_lib,
                    stTrack,
                    nFlannLibCols_sift,
                    FLANNParam,
                    nFlannDataset,
                    nRecogFeature,
                    mnSiftExtraFeatures,
                    FlannIdx_Sift);
    printf("Color FLANN Index Computed.\n\n");
    BuildFlannIndex(2, sSIFTLibFileName,
                    mnColorHistY_lib,
                    stTrack,
                    nFlannLibCols_sift,
                    FLANNParam,
                    nFlannDataset,
                    nRecogFeature,
                    mnSiftExtraFeatures,
                    FlannIdx_Sift);
}


void Attention::Reconfigure(const LayerConfig &config)
{
    // todo
}


void Attention::InputNames(const LayerInputNames &io)
{
    input_name_bgr_     = io.Input(KEY_INPUT_BGR_IMAGE);
    input_name_selHistogram_ = io.Input(KEY_INPUT_SELECTED_HISTOGRAM);
    input_name_selBoundingBox_ = io.Input(KEY_INPUT_SELECTED_BOUNDINGBOX);
}


void AttentionWindow::OutputNames(const LayerOutputNames &config)
{
    name_out_rect_ = config.Output(KEY_OUTPUT_RECT);
    name_out_matchFlag_ = config.OutputOpt(KEY_OUTPUT_MATCH_FLAG);
}


void Attention::Activate(const Signal &signal)
{
    Mat3f color         = signal.MostRecent(input_name_bgr_).get<Mat3f>();
    Mat1f selectedHistogram = signal.MostRecent(input_name_selHistogram_).get<Mat1f>();
    Mat1f selectedBoundingBox = signal.MostRecent(input_name_selBoundingBox_).get<Mat1f>();


    // Color histogram difference
    float colorDistance = 0;
    for (int j = 0; j < selectedHistogram.rows; j++) {
        colorDistance += fabs(mnColorHistY_lib[0][j] - selectedHistogram(j));
    }
    colorDistance = colorDistance / 2.f;


    // SIFT feature comparison
    Keypoint keypts;
    // @todo: use values from the gui
    float siftScales = 3;
    float siftInitSigma = 1.6;
    float siftPeakThrs = 0.01;
    GetSiftKeypoints(color, siftScales, siftInitSigma, siftPeakThrs,
                     selectedBoundingBox(0), selectedBoundingBox(1), selectedBoundingBox(2),
                     selectedBoundingBox(3), keypts);

    // todo: use values from the gui
    float flannKnn = 2;
    float flannMatchFac = 0.7;
    // @todo: find the right way to determine number of columns
    int flannLibCols_sift = mnSiftExtraFeatures[0].length;

    std::vector<int> vnSiftMatched;
    std::vector <double> vnDeltaScale;
    std::vector <double> vnDeltaOri;
    double nMaxDeltaOri = -999;
    double nMaxDeltaScale = -999;
    double nMinDeltaOri = 999;
    double nMinDeltaScale = 999;
    int keyptsCnt = 0;
    int flannIM = 0;

    Recognition_Flann (tcount,
                       flannKnn,
                       flannLibCols_sift,
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
    float siftCntThreshold = 10;
    float colorThreshold = 0.3;

    // todo: (siftfeature + matched_siftfeature)
    rect_ = selectedBoundingBox;
    if (keyptsCnt >= siftCntThreshold && colorDistance < colorThreshold) {
        matchFlag_ = 1;
    } else {
        matchFlag_ = 0;
    }
}
