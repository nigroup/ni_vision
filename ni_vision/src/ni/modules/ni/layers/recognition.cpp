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

}


void Attention::Reconfigure(const LayerConfig &config)
{

}


void Attention::InputNames(const LayerInputNames &io)
{
    input_name_bgr_     = io.Input(KEY_INPUT_BGR_IMAGE);
    input_name_attList_ = io.Input(KEY_INPUT_ATT_LIST);
}

void Attention::Activate(const Signal &signal)
{
    Mat3f color         = signal.MostRecent(input_name_bgr_).get<Mat1f>();
    Mat1f attentionList = signal.MostRecent(input_name_attList_).get<Mat1f>();
    // Todo: object properties übergeben, surface statt attention_list


    // color histogramm difference
    float dc = 0;

    for (int j = 0; j < nTrackHistoBin_tmp; j++) {
        dc += fabs(mnColorHistY_lib[0][j] - stMems.mnColorHist[i][j]);
    }
    dc /= 2.f;

    // SIFT feature comparison

}
