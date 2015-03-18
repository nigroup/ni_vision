#include "ni/layers/surfacetracking.h"

#include "elm/core/debug_utils.h"

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
}

void SurfaceTracking::InputNames(const LayerInputNames &io)
{
    input_name_bgr_     = io.Input(KEY_INPUT_BGR_IMAGE);
    input_name_cloud_   = io.Input(KEY_INPUT_CLOUD);
    input_name_map_     = io.Input(KEY_INPUT_MAP);
}

void SurfaceTracking::Activate(const Signal &signal)
{
    Mat1f color         = signal.MostRecent(input_name_bgr_).get<Mat1f>();
    CloudXYZPtr cloud   = signal.MostRecent(input_name_cloud_).get<CloudXYZPtr>();
    Mat1f map           = signal.MostRecent(input_name_map_).get<Mat1f>();

    Mat bgr;
    color.convertTo(bgr, CV_8UC3, 255.f);

    obsereved_.clear();
    extractFeatures(cloud, bgr, map, obsereved_);

    if(memory_.size() > 0) {

        computeFeatureDistance(obsereved_, memory_);

        // Elemination of rows and columns that have a unique minimum match
        int nMemsCnt = static_cast<int>(memory_.size());
        int nSurfCnt = static_cast<int>(obsereved_.size());

        vector<VecF> mnDistTmp(dist_.rows, VecF(dist_.cols));
        for(int r=0; r<dist_.rows; r++) {

            for(int c=0; c<dist_.cols; c++) {

                mnDistTmp[r][c] = dist_(r, c);
            }
        }

        int nObjsNrLimit = 1000;

        VecI surfCandCount(nSurfCnt, 0);
        VecI segCandMin(nSurfCnt, nObjsNrLimit);
        VecI memCandCount(nMemsCnt, 0);
        VecI memCandMin(nMemsCnt, nObjsNrLimit);
        VecI vnMatchedSeg(nSurfCnt, nObjsNrLimit*2);

        Tracking_OptPre(nMemsCnt, nSurfCnt,
                        nObjsNrLimit,
                        mnDistTmp,
                        surfCandCount,
                        segCandMin,
                        memCandCount,
                        memCandMin,
                        vnMatchedSeg);

        VecI vnMatchedMem(nMemsCnt, nObjsNrLimit*2);

        /* Main Optimization */
        std::vector<int> idx_seg;
        int cnt_nn = 0;
        for (int i=0; i < nSurfCnt; i++) {

            if (vnMatchedSeg[i] > nObjsNrLimit) {

                for (int j=0; j < nMemsCnt; j++) {

                    if (mnDistTmp[i][j] < max_dist_) {

                        vnMatchedMem[j] = 0;
                    }
                }
                idx_seg.resize(cnt_nn + 1);
                idx_seg[cnt_nn++] = i;
            }
        }

        if (cnt_nn > 0) {

            std::vector<int> idx_mem;
            cnt_nn = 0;
            for (int j=0; j < nMemsCnt; j++) {

                if (!vnMatchedMem[j]) {

                    idx_mem.resize(cnt_nn + 1);
                    idx_mem[cnt_nn++] = j;
                }
            }

            int munkres_huge = 100;
            int nDimMunkres = max(static_cast<int>(idx_seg.size()),
                                  static_cast<int>(idx_mem.size()));
            MunkresMatrix<double> m_MunkresIn(nDimMunkres, nDimMunkres);
            MunkresMatrix<double> m_MunkresOut(nDimMunkres, nDimMunkres);

            for (size_t i=0; i < idx_seg.size(); i++) {

                for (size_t j=0; j < idx_mem.size(); j++) {

                    m_MunkresIn(i, j) = mnDistTmp[idx_seg[i]][idx_mem[j]];
                }
            }

            if (idx_mem.size() > idx_seg.size()) {

                for (int i=(int)idx_seg.size(); i < nDimMunkres; i++) {

                    for (int j=0; j < nDimMunkres; j++) {

                        m_MunkresIn(i, j) = static_cast<double>(rand() % 10 + munkres_huge);
                    }
                }
            }
            if (idx_mem.size() < idx_seg.size()) {

                for (int j=(int)idx_mem.size(); j < nDimMunkres; j++) {

                    for (int i=0; i < nDimMunkres; i++) {

                        m_MunkresIn(i,j) = static_cast<double>(rand()% 10 + munkres_huge);
                    }
                }
            }

            m_MunkresOut = m_MunkresIn; // for in-place substitution

            Munkres m;
            m.solve(m_MunkresOut);

            // Specifying the output matrix
            for (size_t i=0; i < idx_seg.size(); i++) {

                for (size_t j=0; j < idx_mem.size(); j++) {

                    if (m_MunkresOut(i,j) == 0) {

                        vnMatchedSeg[idx_seg[i]] = idx_mem[j];
                    }
                    else {
                        mnDistTmp[idx_seg[i]][idx_mem[j]] = DISTANCE_HUGE;
                    }
                }
            }
        }
        // End of the optimization


    }
    else {

        memory_ = obsereved_;
    }

    m_ = map; // until layer produces actual output
}

void SurfaceTracking::extractFeatures(
        const CloudXYZPtr &cloud,
        const Mat &bgr,
        const Mat1f &map,
        vector<Surface> &surfaces)
{    
    surfaces.clear();
    vector<BoundingBox2D> rects;

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
            r.y = static_cast<int>(min_x);
            r.width = static_cast<int>(max_x-min_x+1);
            r.height = static_cast<int>(max_y-min_y+1);

            rects.push_back(r);

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
    vector<BoundingBox3D> cubes(NB_SEGS);
    {
        for (int i=0; i<NB_SEGS; i++) {

            BoundingBox3D cube(sub_clouds[i]);
            cubes[i] = cube;
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

void SurfaceTracking::Tracking_OptPre(int nMemsCnt, int nSurfCnt,
                                      int nObjsNrLimit,
                                      vector<VecF > &mnDistTmp,
                                      VecI &vnSurfCandCnt,
                                      VecI &vnSegCandMin,
                                      VecI &vnMemsCandCnt,
                                      VecI &vnMemCandMin,
                                      VecI &vnMatchedSeg) const
{
    float offset = 0.01;
    for (int i = 0; i < nSurfCnt; i++) {

        float j_min = DISTANCE_HUGE;
        for (int j = 0; j < nMemsCnt; j++) {

            if (mnDistTmp[i][j] >= max_dist_) {

                mnDistTmp[i][j] = DISTANCE_HUGE;
            }
            else {
                vnSurfCandCnt[i]++;

                if (mnDistTmp[i][j] > j_min) {
                    continue;
                }
                if (mnDistTmp[i][j] == j_min) {

                    mnDistTmp[i][j] += offset;
                }
                else {
                    j_min = mnDistTmp[i][j];
                    vnSegCandMin[i] = j;
                }
            }
        }
    }

    for (int j=0; j < nMemsCnt; j++) {

        float i_min = DISTANCE_HUGE;
        for (int i=0; i < nSurfCnt; i++) {

            if (mnDistTmp[i][j] >= max_dist_) {
                continue;
            }
            vnMemsCandCnt[j]++;

            if (mnDistTmp[i][j] > i_min) {
                continue;
            }
            if (mnDistTmp[i][j] == i_min) {

                mnDistTmp[i][j] += offset;
            }
            else {
                i_min = mnDistTmp[i][j];
                vnMemCandMin[j] = i;
            }
        }
    }

    for (int i=0; i < nSurfCnt; i++) {
        if (vnSurfCandCnt[i]) {

            // if no other initial elements in the column
            if (vnMemsCandCnt[vnSegCandMin[i]] < 2) {

                if (vnMatchedSeg[i] < nObjsNrLimit) {

                    printf("Error, the segment %d is already matched %d\n",
                           i, vnSegCandMin[i]);
                }
                vnMatchedSeg[i] = vnSegCandMin[i];

                if (vnSurfCandCnt[i] > 1) Tracking_OptPreFunc(i,
                                                              vnSegCandMin[i],
                                                              nMemsCnt,
                                                              nObjsNrLimit,
                                                              vnSurfCandCnt,
                                                              vnMemsCandCnt,
                                                              vnMemCandMin,
                                                              vnMatchedSeg,
                                                              mnDistTmp);
            }
        }
        else vnMatchedSeg[i] = nObjsNrLimit;
    }
}

void SurfaceTracking::Tracking_OptPreFunc(int seg,
                                          int j_min,
                                          int nMemsCnt,
                                          int nObjsNrLimit,
                                          VecI &vnSurfCandCnt,
                                          VecI &vnMemsCandCnt,
                                          VecI &vnMemCandMin,
                                          VecI &vnMatchedSeg,
                                          vector<VecF > &mnDistTmp) const
{
    for (int j=0; j < nMemsCnt; j++) {

        if (j != j_min && mnDistTmp[seg][j] <= max_dist_) {

            mnDistTmp[seg][j] = DISTANCE_HUGE;
            vnMemsCandCnt[j]--;
            vnSurfCandCnt[seg]--;
            if (!vnMemsCandCnt[j]) vnMemCandMin[j] = nObjsNrLimit;

            //// if there is only one candidate in the colum left
            if (vnMemsCandCnt[j] == 1) {

                for (int i=0; i < seg; i++) {

                    if (mnDistTmp[i][j] <= max_dist_) {

                        vnMatchedSeg[i] = j;
                        if (vnSurfCandCnt[i] > 1) {
                            Tracking_OptPreFunc(i, j,
                                                nMemsCnt, nObjsNrLimit,
                                                vnSurfCandCnt,
                                                vnMemsCandCnt,
                                                vnMemCandMin,
                                                vnMatchedSeg,
                                                mnDistTmp);
                        }
                    }
                }
            }
        }
    }
}
