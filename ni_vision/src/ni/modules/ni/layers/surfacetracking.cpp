#include "ni/layers/surfacetracking.h"

#include "elm/core/debug_utils.h"

#include "elm/core/cv/mat_vector_utils.h"
#include "elm/core/cv/mat_vector_utils_inl.h"
#include "elm/core/exception.h"
#include "elm/core/inputname.h"
#include "elm/core/layerconfig.h"
#include "elm/core/layerinputnames.h"
#include "elm/core/signal.h"
#include "elm/ts/layerattr_.h"

#include "ni/core/boundingbox2d.h"
#include "ni/core/boundingbox3d.h"
#include "ni/core/colorhistogram.h"

#include "ni/3rdparty/munkres/munkres.hpp"
#include "ni/legacy/func_segmentation.h"

using namespace std;
using namespace cv;
using namespace elm;
using namespace ni;

const string SurfaceTracking::PARAM_HIST_BINS       = "bins";
const string SurfaceTracking::PARAM_WEIGHT_COLOR    = "weight_color";
const string SurfaceTracking::PARAM_WEIGHT_POS      = "weight_pos";
const string SurfaceTracking::PARAM_WEIGHT_SIZE     = "weight_size";
const string SurfaceTracking::PARAM_MAX_COLOR       = "max_color";
const string SurfaceTracking::PARAM_MAX_POS         = "max_pos";
const string SurfaceTracking::PARAM_MAX_SIZE        = "max_size";
const string SurfaceTracking::PARAM_MAX_DIST        = "max_dist";

const string SurfaceTracking::KEY_INPUT_BGR_IMAGE   = "bgr";
const string SurfaceTracking::KEY_INPUT_CLOUD       = "points";
const string SurfaceTracking::KEY_INPUT_MAP         = "map";

const float SurfaceTracking::DISTANCE_HUGE = 100.f;

#include <boost/assign/list_of.hpp>
template <>
elm::MapIONames LayerAttr_<SurfaceTracking>::io_pairs = boost::assign::map_list_of
        ELM_ADD_INPUT_PAIR(SurfaceTracking::KEY_INPUT_BGR_IMAGE)
        ELM_ADD_INPUT_PAIR(SurfaceTracking::KEY_INPUT_CLOUD)
        ELM_ADD_INPUT_PAIR(SurfaceTracking::KEY_INPUT_MAP)
        ELM_ADD_OUTPUT_PAIR(detail::BASE_MATOUTPUT_LAYER__KEY_OUTPUT_RESPONSE)
        ;

SurfaceTracking::~SurfaceTracking()
{
}

SurfaceTracking::SurfaceTracking()
    : elm::base_MatOutputLayer()
{
    Clear();
}

SurfaceTracking::SurfaceTracking(const LayerConfig &config)
    : elm::base_MatOutputLayer(config)
{
    Reset(config);
}

void SurfaceTracking::Clear()
{
    dist_ = Mat1f();
}

void SurfaceTracking::Reset(const LayerConfig &config)
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

void SurfaceTracking::Reconfigure(const LayerConfig &config)
{
    PTree p = config.Params();
    nb_bins_ = p.get<int>(PARAM_HIST_BINS);

    weight_color_   = p.get<float>(PARAM_WEIGHT_COLOR);
    weight_pos_     = p.get<float>(PARAM_WEIGHT_POS);
    weight_size_    = p.get<float>(PARAM_WEIGHT_SIZE);

    max_color_  = p.get<float>(PARAM_MAX_COLOR);
    max_pos_    = p.get<float>(PARAM_MAX_POS);
    max_size_   = p.get<float>(PARAM_MAX_SIZE);
    max_dist_   = p.get<float>(PARAM_MAX_DIST);

    // initialize legacy members
    stTrack.ClrMode = 1;
    stTrack.CntLost = 1;
    stTrack.CntMem = 10;
    stTrack.CntStable = 1;
    stTrack.DClr = max_color_;
    stTrack.DPos = max_pos_;
    stTrack.DSize = max_size_;
    stTrack.Dist = max_dist_;
    stTrack.FClr = weight_color_;
    stTrack.DPos = weight_pos_;
    stTrack.FSize = weight_size_;
    stTrack.HistoBin = nb_bins_;
    stTrack.MFac = 100.;
}

void SurfaceTracking::InputNames(const LayerInputNames &io)
{
    input_name_bgr_     = io.Input(KEY_INPUT_BGR_IMAGE);
    input_name_cloud_   = io.Input(KEY_INPUT_CLOUD);
    input_name_map_     = io.Input(KEY_INPUT_MAP);
}

void SurfaceTracking::Activate(const Signal &signal)
{
    Mat3f color         = signal.MostRecent(input_name_bgr_).get<Mat1f>();
    CloudXYZPtr cloud   = signal.MostRecent(input_name_cloud_).get<CloudXYZPtr>();
    Mat1f map           = signal.MostRecent(input_name_map_).get<Mat1f>();

    Mat bgr;
    color.convertTo(bgr, CV_8UC3, 255.f);

    obsereved_.clear();
    extractFeatures(cloud, bgr, map, obsereved_);

    int nSurfCnt = static_cast<int>(obsereved_.size());

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

        VecSurfacesToSurfProp(obsereved_, stSurf);

        int nObjsNrLimit = 1000;

        Tracking(nSurfCnt,
                 nObjsNrLimit,
                 stTrack,
                 nTrackHistoBin_max,
                 stSurf,
                 stMems,
                 nMemsCnt,
                 vnMemsValidIdx,
                 mnMemsRelPose,
                 false,
                 framec);

        //  Making tracking map to a neighborhood matrix for surface saliencies
        int nDsSize = static_cast<int>(map.total());
        VecI vnTrkMap(nDsSize, -1);         // Tracking Map

        for (int i=0; i < nMemsCnt; i++) {

            if (stMems.vnStableCtr[i] < stTrack.CntStable ||
                    stMems.vnLostCtr[i] > stTrack.CntLost) {
                continue;
            }
            for (size_t j=0; j < stMems.mnPtsIdx[i].size(); j++) {

                vnTrkMap[stMems.mnPtsIdx[i][j]] = stMems.vnIdx[i];
            }
        }

        m_ = Mat1f(map.size(), -1.f);
        for (int i=0; i < nDsSize; i++) {

            if (vnTrkMap[i] < 0) {
                continue;
            }
            m_(i) = static_cast<float>(vnTrkMap[i]);
        }
    }

    framec++;
}

void SurfaceTracking::extractFeatures(
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

void SurfaceTracking::computeFeatureDistance(const vector<Surface> &surfaces,
                                             const vector<Surface> &mem)
{
    int nb_surfaces = static_cast<int>(surfaces.size());
    int nb_mem = static_cast<int>(mem.size());

    const int DIM = max(nb_surfaces, nb_mem);

    dist_color_ = Mat1f(DIM, DIM, DISTANCE_HUGE);
    dist_pos_   = Mat1f(DIM, DIM, DISTANCE_HUGE);
    dist_size_  = Mat1f(DIM, DIM, DISTANCE_HUGE);
    dist_       = Mat1f(DIM, DIM, DISTANCE_HUGE);

    // todo vectorize
    for (int i=0; i<nb_surfaces; i++) {

        for (int j=0; j<nb_mem; j++) {

            float dc = cv::sum(abs(surfaces[i].colorHistogram()-mem[j].colorHistogram()))[0];

            float dp = surfaces[i].distance(mem[j]);

            float diag_i = surfaces[i].diagonal();
            float diag_j = mem[j].diagonal();
            float ds = fabs(diag_i - diag_j) / max(diag_i, diag_j);

            dist_color_(i, j) = dc;
            dist_pos_(i, j) = dp;
            dist_size_(i, j) = ds;

            if (dc < max_color_ && dp < max_pos_ && ds < max_size_) {

                dist_(i, j) = weight_color_ * dc + weight_pos_ * dp + weight_size_ * ds;
            }
        } // j'th surface in STM
    } // i'th newly observed surface
}

void SurfaceTracking::SurfPropToVecSurfaces(const SurfProp &surf_prop, std::vector<Surface> &surfaces) const
{
    for(size_t i=1; i<surf_prop.vnIdx.size(); i++) {

        Surface s;

        s.id(surf_prop.vnIdx[i]);
        s.pixelIndices(surf_prop.mnPtsIdx[i]); // sets pixel count also.

        // x, y, w=xmax-x, h=ymax-y
        Rect2i r(surf_prop.mnRect[i][0],
                surf_prop.mnRect[i][1],
                surf_prop.mnRect[i][2]-r.x,
                surf_prop.mnRect[i][3]-r.y);
        s.rect(r);

        VecF tmp = surf_prop.mnCubic[i];
        s.cubeVertices(elm::Vec_ToRowMat_<float>(tmp).reshape(1, 2).clone()); // also sets cube center

        s.colorHistogram(Mat1f(surf_prop.mnColorHist[i], true));
        s.diagonal(surf_prop.vnLength[i]);

        surfaces[i-1] = s;
    }
}

void SurfaceTracking::VecSurfacesToSurfProp(const std::vector<Surface> &surfaces, SurfProp &surf_prop) const
{
    for(size_t i=0; i<surfaces.size(); i++) {

        Surface s = surfaces[i];

        surf_prop.vnIdx[i] = s.id();
        surf_prop.vnPtsCnt[i] = s.pixelCount();
        surf_prop.mnPtsIdx[i] = s.pixelIndices(); // do we need this?

        Rect2i r = s.rect();
        surf_prop.mnRect[i][0] = r.x;
        surf_prop.mnRect[i][1] = r.y;
        surf_prop.mnRect[i][2] = r.x + r.width;
        surf_prop.mnRect[i][3] = r.y + r.height;

        BoundingBox2D bbox(r);
        Mat1f pt = bbox.centralPoint();
        surf_prop.mnRCenter[i][0] = static_cast<int>(pt(0));
        surf_prop.mnRCenter[i][1] = static_cast<int>(pt(1));

        surf_prop.mnCubic[i]   = elm::Mat_ToVec_(Mat1f(s.cubeVertices()));
        surf_prop.mnCCenter[i] = elm::Mat_ToVec_(Mat1f(s.cubeCenter()));
        surf_prop.mnColorHist[i] = elm::Mat_ToVec_(s.colorHistogram());
        surf_prop.vnLength[i] = s.diagonal();
    }
}
