#include "ni/layers/attention.h"
#include <opencv2/highgui/highgui.hpp>
#include "elm/core/debug_utils.h"
#include <set>

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

#include "ni/legacy/func_segmentation.h"
#include "ni/legacy/surfprop_utils.h"

using namespace std;
using namespace cv;
using namespace elm;
using namespace ni;

const string Attention::PARAM_HIST_BINS       = "bins";
const string Attention::PARAM_WEIGHT_COLOR    = "weight_color";
const string Attention::PARAM_WEIGHT_POS      = "weight_pos";
const string Attention::PARAM_WEIGHT_SIZE     = "weight_size";
const string Attention::PARAM_MAX_COLOR       = "max_color";
const string Attention::PARAM_MAX_POS         = "max_pos";
const string Attention::PARAM_MAX_SIZE        = "max_size";
const string Attention::PARAM_MAX_DIST        = "max_dist";

const string Attention::KEY_INPUT_BGR_IMAGE   = "bgr";
const string Attention::KEY_INPUT_CLOUD       = "points";
const string Attention::KEY_INPUT_MAP         = "map";

const float Attention::DISTANCE_HUGE = 100.f;

#include <boost/assign/list_of.hpp>
template <>
elm::MapIONames LayerAttr_<Attention>::io_pairs = boost::assign::map_list_of
        ELM_ADD_INPUT_PAIR(Attention::KEY_INPUT_BGR_IMAGE)
        ELM_ADD_INPUT_PAIR(Attention::KEY_INPUT_CLOUD)
        ELM_ADD_INPUT_PAIR(Attention::KEY_INPUT_MAP)
        ELM_ADD_OUTPUT_PAIR(detail::BASE_MATOUTPUT_LAYER__KEY_OUTPUT_RESPONSE)
        ;

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
    dist_ = Mat1f();
}

void Attention::Reset(const LayerConfig &config)
{
    // reset legacy members
    int nTrackHistoBin_max = nb_bins_ * nb_bins_ * nb_bins_;
    int nObjsNrLimit = 1000;

    stMems.vnIdx.resize(nObjsNrLimit,       0);
    stMems.vnPtsCnt.resize(nObjsNrLimit,    0);

    stMems.mnPtsIdx.assign(nObjsNrLimit,    VecI(0, 0));
    stMems.mnRect.assign(nObjsNrLimit,      VecI(4, 0));
    stMems.mnRCenter.assign(nObjsNrLimit,   VecI(2, 0));
    stMems.mnCubic.assign(nObjsNrLimit,     VecF(6, 0.f));
    stMems.mnCCenter.assign(nObjsNrLimit,   VecF(3, 0.f));
    stMems.mnColorHist.resize(nObjsNrLimit);

    stMems.vnLength.resize(nObjsNrLimit, 0);
    stMems.vnMemCtr.resize(nObjsNrLimit, 0);
    stMems.vnStableCtr.resize(nObjsNrLimit, 0);
    stMems.vnLostCtr.resize(nObjsNrLimit,   0);
    stMems.vnFound.resize(nObjsNrLimit,     0);

    framec = 0;
    nMemsCnt = 0;

    Reconfigure(config);
}

void Attention::Reconfigure(const LayerConfig &config)
{
    PTree p = config.Params();
    nb_bins_ = p.get<int>(PARAM_HIST_BINS);

    // initialize legacy members
    stTrack.ClrMode = 1;
    stTrack.CntLost = 1;
    stTrack.CntMem = 10;
    stTrack.CntStable = 1;
//    stTrack.DClr = max_color_;
//    stTrack.DPos = max_pos_;
//    stTrack.DSize = max_size_;
//    stTrack.Dist = max_dist_;
//    stTrack.FClr = weight_color_;
//    stTrack.DPos = weight_pos_;
//    stTrack.FSize = weight_size_;
    stTrack.HistoBin = nb_bins_;
    stTrack.MFac = 100.;
}

void Attention::InputNames(const LayerInputNames &io)
{
    input_name_bgr_     = io.Input(KEY_INPUT_BGR_IMAGE);
    input_name_cloud_   = io.Input(KEY_INPUT_CLOUD);
    input_name_map_     = io.Input(KEY_INPUT_MAP);
}

void Attention::Activate(const Signal &signal)
{
    Mat3f color         = signal.MostRecent(input_name_bgr_).get<Mat1f>();
    CloudXYZPtr cloud   = signal.MostRecent(input_name_cloud_).get<CloudXYZPtr>();
    Mat1f map           = signal.MostRecent(input_name_map_).get<Mat1f>();

    Mat bgr;
    color.convertTo(bgr, CV_8UC3, 255.f);

    observed_.clear();
    extractFeatures(cloud, bgr, map, observed_);

    int nSurfCnt = static_cast<int>(observed_.size());

    {
        int nTrackHistoBin_max = nb_bins_ * nb_bins_ * nb_bins_;

        SurfProp stSurf;
        stSurf.vnIdx.resize(nSurfCnt, 0);
        stSurf.vnPtsCnt.resize(nSurfCnt, 0);
        stSurf.mnPtsIdx.resize(nSurfCnt,    VecI());
        stSurf.mnRect.assign(nSurfCnt,      VecI(4,0));
        stSurf.mnRCenter.assign(nSurfCnt,   VecI(2,0));
        stSurf.mnCubic.assign(nSurfCnt,     VecF(6,0));
        stSurf.mnCCenter.assign(nSurfCnt,   VecF(3,0));
        stSurf.vnLength.resize(nSurfCnt, 0);
        stSurf.mnColorHist.resize(nSurfCnt, VecF(nTrackHistoBin_max, 0));
        stSurf.vnMemCtr.resize(nSurfCnt, stTrack.CntMem - stTrack.CntStable);
        stSurf.vnStableCtr.resize(nSurfCnt, 0);
        stSurf.vnLostCtr.resize(nSurfCnt, stTrack.CntLost + 10);

        VecSurfacesToSurfProp(observed_, stSurf);


    }

    framec++;
}

void Attention::extractFeatures(
        const CloudXYZPtr &cloud,
        const Mat &bgr,
        const Mat1f &map,
        vector<Surface> &surfaces)
{    
    surfaces.clear();

    // reorder segment ids to be [1, N]
    int NB_SEGS = 0;
    double min_val, max_val;
    minMaxIdx(map, &min_val, &max_val);
    {
        const int UPPER_LIM = static_cast<int>(max_val)+1;
        vector<int> hist(UPPER_LIM, 0);

        for(size_t i=0; i<map.total(); i++) {

            int value = map(i);

            // dont bother counting zeros (a.k.a not assigned)
            if(value > 0) {

                hist[value]++;
            }
        }

        vector<int> id_lut(UPPER_LIM, 0);
        for(int i=1; i<UPPER_LIM; i++) {

            if(hist[i] > 0) {

                Surface surface;

                surface.id(++NB_SEGS);
                surface.overwritePixelCount(hist[i]);

                surfaces.push_back(surface);

                id_lut[i] = NB_SEGS;
            }
        }

        // replace segment values with new surface ids
        // and record indicies for later use
        std::vector<VecI > indicies(NB_SEGS+1, VecI());

        for(size_t i=0; i<map.total(); i++) {

            // replace
            int seg_id = id_lut[static_cast<int>(map(i))];
            //map(i) = seg_id;

            // append index
            indicies[seg_id].push_back(i);
        }

        // attach indices to surface objects
        // and record 2-d bounding boxes around each surface
        // and compute color histogram for each surface

        for (int i=0; i<NB_SEGS; i++) {

            VecI tmp = indicies[i+1];
            surfaces[i].pixelIndices(tmp); // heavy copy?

            BoundingBox2D r;

            Mat1i tmp_mat(tmp);
            Mat1i coords(tmp_mat.rows, 2); // col(0) := x, col(1) := y
            cv::divide(tmp_mat, map.cols, coords.col(1));
            Mat1i x = coords.col(0);

            // modulus operation for OpenCV Mat:
            for (int j=0; j<tmp_mat.rows; j++) {

                x(j) = tmp[j] % map.cols;
            }

            double min_x, max_x, min_y, max_y;
            cv::minMaxIdx(coords.col(0), &min_x, &max_x);
            cv::minMaxIdx(coords.col(1), &min_y, &max_y);

            r.x = static_cast<int>(min_x);
            r.y = static_cast<int>(min_y);
            r.width = static_cast<int>(max_x-min_x+1);
            r.height = static_cast<int>(max_y-min_y+1);

            surfaces[i].rect(r);

            // color histogram
            Mat1f hist;
            computeColorHist(bgr, tmp, nb_bins_, hist);

            surfaces[i].colorHistogram(hist);
        }
    }

    // sub clouds
    vector<CloudXYZPtr> sub_clouds(NB_SEGS);
    {
        for (int i=0; i<NB_SEGS; i++) {

            sub_clouds[i].reset(new CloudXYZ(*cloud, surfaces[i].pixelIndices()));
        }
    }

    // cubes
    {
        for (int i=0; i<NB_SEGS; i++) {

            BoundingBox3D cube(sub_clouds[i]);
            surfaces[i].cubeVertices(cube.cubeVertices());
            surfaces[i].cubeCenter(cube.centralPoint());
            surfaces[i].diagonal(cube.diagonal());
        }
    }
}


