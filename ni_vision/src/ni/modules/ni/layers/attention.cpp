#include "ni/layers/attention.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "elm/core/debug_utils.h"

#include <set>

#include <boost/filesystem.hpp>

#include "elm/core/debug_utils.h"
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
//#include "ni/core/colorhistogram.h"

#include "ni/legacy/func_init.h"
#include "ni/legacy/func_recognition.h"
#include "ni/legacy/surfprop_utils.h"
#include "ni/legacy/func_operations.h"

using namespace std;
namespace bfs=boost::filesystem;
using namespace cv;
using namespace elm;
using namespace ni;

const string Attention::PARAM_HIST_BINS       = "bins";
const string Attention::PARAM_SIZE_MAX        = "att_size_max";
const string Attention::PARAM_SIZE_MIN        = "att_size_min";
const string Attention::PARAM_PTS_MIN         = "att_pts_min";
const string Attention::PARAM_PATH_COLOR      = "path_color";
const string Attention::PARAM_PATH_SIFT       = "path_sift";

const string Attention::KEY_INPUT_BGR_IMAGE   = "bgr";
const string Attention::KEY_INPUT_CLOUD       = "points";
const string Attention::KEY_INPUT_MAP         = "map";
const string Attention::KEY_OUTPUT_HISTOGRAM  = "hist";
const string Attention::KEY_OUTPUT_RECT       = "rect";

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
    : elm::base_Layer()
{
    Clear();
}

void Attention::Clear()
{
    dist_ = Mat1f();
    inhibitionMemory = std::vector<int>();
}

void Attention::Reset(const LayerConfig &config)
{
    // reset legacy members
    //int nTrackHistoBin_max = nb_bins_ * nb_bins_ * nb_bins_;
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

    PTree p = config.Params();

    bfs::path path_color = p.get<bfs::path>(PARAM_PATH_COLOR);

    if(!bfs::is_regular_file(path_color)) {

        stringstream s;
        s << "Invalid path to color model for attention layer (" << path_color << ")";
        ELM_THROW_FILEIO_ERROR(s.str());
    }

    bfs::path path_sift = p.get<bfs::path>(PARAM_PATH_SIFT);

    if(!bfs::is_regular_file(path_sift)) {

        stringstream s;
        s << "Invalid path to sift model for attention layer (" << path_sift << ")";
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

void Attention::Reconfigure(const LayerConfig &config)
{
    PTree p = config.Params();
    nb_bins_ = p.get<int>(PARAM_HIST_BINS);

    nAttSizeMax = p.get<int>(PARAM_SIZE_MAX);
    nAttSizeMin = p.get<int>(PARAM_SIZE_MIN);
    nAttPtsMin  = p.get<int>(PARAM_PTS_MIN);

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

void Attention::OutputNames(const LayerOutputNames &io)
{
    name_out_histogram_ = io.Output(KEY_OUTPUT_HISTOGRAM);
    name_out_rect_ = io.Output(KEY_OUTPUT_RECT);
}

void Attention::Activate(const Signal &signal)
{
    Mat3f color         = signal.MostRecent(input_name_bgr_).get<Mat1f>();
    CloudXYZPtr cloud   = signal.MostRecent(input_name_cloud_).get<CloudXYZPtr>();
    Mat1f map           = signal.MostRecent(input_name_map_).get<Mat1f>();

    // Debug
    printf("%i %i\n", color.cols, color.rows);
    printf("%i %i\n", map.cols, map.rows);

    // Computing downsampled color-image (with dimension of depth-image)
    Mat3f colorDS;
    resize(color, colorDS, map.size());

    Mat bgr;
    colorDS.convertTo(bgr, CV_8UC3, 255.f);

    observed_.clear();
    extractFeatures(cloud, bgr, map, observed_);

    int nMemsCnt = static_cast<int>(observed_.size());

    int nTrackHistoBin = nb_bins_ * nb_bins_ * nb_bins_;

    stMems.vnIdx.resize(nMemsCnt, 0);
    stMems.vnPtsCnt.resize(nMemsCnt, 0);
    stMems.mnPtsIdx.resize(nMemsCnt,    VecI());
    stMems.mnRect.assign(nMemsCnt,      VecI(4,0));
    stMems.mnRCenter.assign(nMemsCnt,   VecI(2,0));
    stMems.mnCubic.assign(nMemsCnt,     VecF(6,0));
    stMems.mnCCenter.assign(nMemsCnt,   VecF(3,0));
    stMems.vnLength.resize(nMemsCnt, 0);
    stMems.mnColorHist.resize(nMemsCnt, VecF(nTrackHistoBin, 0));
    stMems.vnMemCtr.resize(nMemsCnt, stTrack.CntMem - stTrack.CntStable);
    stMems.vnStableCtr.resize(nMemsCnt, 0);
    stMems.vnLostCtr.resize(nMemsCnt, stTrack.CntLost + 10);

    VecSurfacesToSurfProp(observed_, stMems);

    // Top-down guidance

    // Sorting Objects
    int nObjsNrLimit = 1000;
    float tmp_diff = 100;
    std::vector<std::pair<float, int> > veCandClrDist(nMemsCnt);
    std::vector<bool> vbProtoCand(nObjsNrLimit, false);
    // Top-down Selection

    for (int i = 0; i < nMemsCnt; i++) {

        veCandClrDist[i].second = i;

//        if (stMems.vnStableCtr[i] < stTrack.CntStable || stMems.vnLostCtr[i] > stTrack.CntLost) {

//            veCandClrDist[i].first = tmp_diff++;
//            continue;
//        }

        if (stMems.vnLength[i]*1000 > nAttSizeMax ||
                stMems.vnLength[i]*1000 < nAttSizeMin ||
                stMems.vnPtsCnt[i] < nAttPtsMin) {

            veCandClrDist[i].first = tmp_diff++;
            continue;
        }

        vbProtoCand[i] = true;

        float dc = 0;
        for (int j = 0; j < nTrackHistoBin; j++) {
            dc += fabs(mnColorHistY_lib[0][j] - stMems.mnColorHist[i][j]);
        }
        veCandClrDist[i].first = dc/2.f;
    } // surface

    Attention_TopDown (vbProtoCand,
                       stMems,
                       nMemsCnt,
                       veCandClrDist);

    Mat1f m_ = Mat1f(1, nMemsCnt, -1.f);
    for(int i=0; i<nMemsCnt; i++) {

        m_(i) = static_cast<float>(stMems.vnIdx[i]);
    }

    // Inhibition of return
    int currentIndex = -1;
    int flag = 0;
    for(int i = 0; i < stMems.vnIdx.size(); i++) {
        flag = 0;
        for(int j = 0; j < inhibitionMemory.size(); j++) {
            if (stMems.vnIdx[i] == inhibitionMemory[j]) {
                flag = 1;
                break; // surface was already examined
            }
        }
        // surface was not examined in current cycle
        // and matches general attention criteria
        if(flag == 0
                && stMems.vnLength[i]*1000 < nAttSizeMax
                && stMems.vnLength[i]*1000 > nAttSizeMin
                && stMems.vnPtsCnt[i] > nAttPtsMin) {
            currentIndex = i;
            inhibitionMemory.push_back(stMems.vnIdx[i]);
            break;
        }
    }
    if(currentIndex == -1) { // all surfaces were examined
        currentIndex = 0;
        inhibitionMemory.clear();
        inhibitionMemory.push_back(stMems.vnIdx[0]);
    }

    // Debugging
//    for(int i = 0; i < inhibitionMemory.size(); i++) {
//        printf("%i ", inhibitionMemory[i]);
//    }
//    printf("\n");

    histogram_ = Mat1f(1, stMems.mnColorHist[currentIndex].size());
    for(int i = 0; i < stMems.mnColorHist[currentIndex].size(); i++) {
        histogram_(i) = stMems.mnColorHist[currentIndex][i];
    }


    // factor for rescaling the rgb-image, assumes same factor in x- and y-direction
    int factor = 0;
    if(map.cols && map.rows) {
        factor = (color.cols / map.cols);
    }

    rect_ = Mat1f(1, stMems.mnRect[currentIndex].size());
    for(int i = 0; i < stMems.mnRect[currentIndex].size(); i++) {
        rect_(i) = factor * stMems.mnRect[currentIndex][i];
    }

    // Debugging

    cv::Mat img(colorDS.rows, colorDS.cols, CV_8UC1);
    img.setTo(Scalar(0));
    for(int i=0; i<nMemsCnt; i++) {

        Point2i p(stMems.mnRCenter[i][0], stMems.mnRCenter[i][1]);

        stringstream s;
        s << i << " " << stMems.vnIdx[i];

        cv::putText(img, s.str(), p, CV_FONT_HERSHEY_COMPLEX, 0.4, Scalar(255, 255, 255));
    }

    cv::imshow("att", img);
    cv::waitKey(1);

    // End Debugging
}


void Attention::Response(Signal &signal)
{
    signal.Append(name_out_rect_, rect_);
    signal.Append(name_out_histogram_, histogram_);
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
            std::vector<float> hist_tmp(nb_bins_*nb_bins_*nb_bins_,0);
//            Mat1f hist;
//            computeColorHist(bgr, tmp, nb_bins_, hist);
            Calc3DColorHistogram(bgr, tmp, nb_bins_, hist_tmp);
            //for(size_t i = 0; i < hist_tmp.)
            Mat1f hist = Mat1f(1,nb_bins_*nb_bins_*nb_bins_);
            for(size_t j = 0; j < hist_tmp.size(); j++) {
                hist(j) = hist_tmp[j];
            }
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


