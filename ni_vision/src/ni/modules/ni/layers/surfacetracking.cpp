#include "ni/layers/surfacetracking.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "elm/core/debug_utils.h"
#include <set>

#include "elm/core/cv/mat_utils.h"

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

#include "ni/3rdparty/munkres/munkres.hpp"
#include "ni/legacy/func_segmentation.h"
#include "ni/legacy/surfprop_utils.h"

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
const string SurfaceTracking::KEY_OUTPUT_BOUNDING_BOXES = "boundingBoxes";
const string SurfaceTracking::KEY_OUTPUT_TRACK_MAP = "trackMap";

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
    : elm::base_Layer()
{
    Clear();
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

void SurfaceTracking::OutputNames(const LayerOutputNames &io)
{
    name_out_trackMap_ = io.Output(KEY_OUTPUT_TRACK_MAP);
    name_out_boundingBoxes_ = io.Output(KEY_OUTPUT_BOUNDING_BOXES);
}

void SurfaceTracking::Activate(const Signal &signal)
{
    Mat3f color         = signal.MostRecent(input_name_bgr_).get<Mat1f>();
    CloudXYZPtr cloud   = signal.MostRecent(input_name_cloud_).get<CloudXYZPtr>();
    Mat1f map           = signal.MostRecent(input_name_map_).get<Mat1f>();

    // Computing downsampled color-image (with dimension of depth-image)
    Mat3f colorDS;
    resize(color, colorDS, map.size());

    Mat bgr;
    colorDS.convertTo(bgr, CV_8UC3, 255.f);

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

        int validSurfaceCnt = 0;
        for (int i=0; i < nMemsCnt; i++) {

            if (stMems.vnStableCtr[i] < stTrack.CntStable ||
                    stMems.vnLostCtr[i] > stTrack.CntLost) {
                continue;
            }
            for (size_t j=0; j < stMems.mnPtsIdx[i].size(); j++) {

                vnTrkMap[stMems.mnPtsIdx[i][j]] = stMems.vnIdx[i];
            }
            validSurfaceCnt++;
        }

        trackMap_ = Mat1f(map.size(), -1.f);
        for (int i=0; i < nDsSize; i++) {

            if (vnTrkMap[i] < 0) {
                continue;
            }
            trackMap_(i) = static_cast<float>(vnTrkMap[i]);
        }

        int factor = 0;
        if(map.cols && map.rows) {
            factor = (color.cols / map.cols);
        }
        boundingBoxes_ = Mat1f(validSurfaceCnt, 5);
        {
            int tmp = 0;
            for(int i = 0; i < nMemsCnt; i++) {

                if (stMems.vnStableCtr[i] < stTrack.CntStable ||
                        stMems.vnLostCtr[i] > stTrack.CntLost) {
                    continue;
                }
                boundingBoxes_(tmp,0) = factor * stMems.mnRect[i][0];
                boundingBoxes_(tmp,1) = factor * stMems.mnRect[i][1];
                boundingBoxes_(tmp,2) = factor * stMems.mnRect[i][2];
                boundingBoxes_(tmp,3) = factor * stMems.mnRect[i][3];
                boundingBoxes_(tmp,4) = stMems.vnIdx[i];
                tmp++;
            }
            //printf("%i %i \n", tmp, validSurfaceCnt);
        }
//        cv::imshow("x", elm::ConvertTo8U(m_));
//        cv::waitKey();

//        {
//            std::set<int> snew;
//            for(size_t i=0; i<m_.total(); i++) {

//                snew.insert(static_cast<int>(m_(i)));
//            }

//            VecI vtracked;
//            std::copy(snew.begin(), snew.end(), std::back_inserter(vtracked));

//              ELM_COUT_VAR(elm::to_string(vtracked));
//        }
//        {
//            std::set<int> seg;
//            for(size_t i=0; i<map.total(); i++) {

//                seg.insert(static_cast<int>(map(i)));
//            }

//            VecI vseg;
//            std::copy(seg.begin(), seg.end(), std::back_inserter(vseg));

//            ELM_COUT_VAR(elm::to_string(vseg));
//        }

//        Mat bgr2 = bgr.clone();
//        cv::imshow("t", Mat1b::zeros(10, 10));
//        cv::waitKey(1);
//        for (int i=0; i < stMems.vnIdx.size(); i++) {

//            Point org(stMems.mnRCenter[i][0], stMems.mnRCenter[i][1]);
//            stringstream s;
//            s<<stMems.vnIdx[i];
//            cv::putText(bgr2,
//                        s.str(),
//                        org,
//                        1,
//                        1.f,
//                        Scalar(255, 255, 255));
//        }
//        cv::imshow("t", bgr2);
    }

    framec++;
}

void SurfaceTracking::Response(Signal &signal)
{
    signal.Append(name_out_trackMap_, trackMap_);
    signal.Append(name_out_boundingBoxes_, boundingBoxes_);
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


