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
    int nTrackHistoBin_max = nb_bins_ * nb_bins_ * nb_bins_;
    int nObjsNrLimit = 1000;

    stMems.vnIdx.resize(nObjsNrLimit,       0);
    stMems.vnPtsCnt.resize(nObjsNrLimit,    0);

    stMems.mnPtsIdx.assign(nObjsNrLimit,    VecI(0, 0));
    stMems.mnRect.assign(nObjsNrLimit,      VecI(4, 0));
    stMems.mnRCenter.assign(nObjsNrLimit,   VecI(2, 0));
    stMems.mnCubic.assign(nObjsNrLimit,     VecF(6, 0.f));
    stMems.mnCCenter.assign(nObjsNrLimit,   VecF(3, 0.f));
    stMems.mnColorHist.assign(nObjsNrLimit, VecF(nTrackHistoBin_max, 0.f));

    stMems.vnLength.resize(nObjsNrLimit, 0);
    stMems.vnMemCtr.resize(nObjsNrLimit, 0);
    stMems.vnStableCtr.resize(nObjsNrLimit, 0);
    stMems.vnLostCtr.resize(nObjsNrLimit,   0);
    stMems.vnFound.resize(nObjsNrLimit,     0);

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

    int nSurfCnt = static_cast<int>(obsereved_.size());
    int cnt_new = 0, cnt_old = 0;
    std::vector<int> objs_new_no(nSurfCnt, 0);

    if(memory_.size() > 0) {

        computeFeatureDistance(obsereved_, memory_);

        // Elemination of rows and columns that have a unique minimum match
        int nMemsCnt = static_cast<int>(memory_.size());

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

        // Postprocessing...

        // Short-Term-Memory (STM)...

        // Update properties of matched segments in the Short-Term-Memory
        std::vector<bool> objs_old_flag(nMemsCnt, false);

        // conversion: new vector of Surface objects to legacy SurfProp object
        int nTrackHistoBin_max = nb_bins_ * nb_bins_ * nb_bins_;

        SurfProp stSurf;
        stSurf.vnIdx    = VecI(nSurfCnt, 0);
        stSurf.vnPtsCnt = VecI(nSurfCnt);
        stSurf.mnPtsIdx = std::vector<VecI>(nSurfCnt, VecI());
        stSurf.mnRect.assign(nSurfCnt, std::vector<int>(4,0));
        stSurf.mnRCenter.assign(nSurfCnt, std::vector<int>(2,0));
        stSurf.mnCubic.assign(nSurfCnt, std::vector<float>(6,0));
        stSurf.mnCCenter.assign(nSurfCnt, std::vector<float>(3,0));
        stSurf.vnLength.resize(nSurfCnt, 0);
        stSurf.mnColorHist.resize(nSurfCnt, std::vector<float> (nTrackHistoBin_max, 0));
        stSurf.vnMemCtr.resize(nSurfCnt, 0);
        stSurf.vnStableCtr.resize(nSurfCnt, 0);
        stSurf.vnLostCtr.resize(nSurfCnt, 0);

        VecSurfacesToSurfProp(obsereved_, stSurf);
        for (int i=1; i < nSurfCnt; i++) {

            stSurf.vnMemCtr[i] = stTrack.CntMem - stTrack.CntStable;
            stSurf.vnStableCtr[i] = 0;
            stSurf.vnLostCtr[i] = stTrack.CntLost + 10;
        }

        // conversion done

        SurfProp stMemsOld = stMems;
        for (int i=0; i < nSurfCnt; i++) {

            //** Assign matched segments to the Short-Term-Memory **//
            int cand = 0, j_tmp = -1;
            if (vnMatchedSeg[i] < nObjsNrLimit) {

                cand = 1;
                j_tmp = vnMatchedSeg[i];
            }

            if (cand) {

                if (j_tmp < 0) {
                    printf("Tracking Error\n");
                    continue;
                }

                objs_old_flag[j_tmp] = true;
                stMems.vnPtsCnt[j_tmp] = stSurf.vnPtsCnt[i];
                stMems.mnPtsIdx[j_tmp] = stSurf.mnPtsIdx[i];
                stMems.mnRect[j_tmp] = stSurf.mnRect[i];
                stMems.mnRCenter[j_tmp] = stSurf.mnRCenter[i];
                stMems.mnCubic[j_tmp] = stSurf.mnCubic[i];
                stMems.mnCCenter[j_tmp] = stSurf.mnCCenter[i];
                stMems.mnColorHist[j_tmp] = stSurf.mnColorHist[i];
                stMems.vnLength[j_tmp] = stSurf.vnLength[i];
            }
            else {
                objs_new_no[cnt_new++] = i;
            }
        }

        // Filter stable objects from the Short-Term-Memory
        for (int memc=0; memc < nMemsCnt; memc++) {

            if (objs_old_flag[memc]) {

                stMems.vnMemCtr[memc]++;
                if (stMems.vnMemCtr[memc] < stTrack.CntMem - stTrack.CntStable + 2) stMems.vnMemCtr[memc] = stTrack.CntMem - stTrack.CntStable + 1;
                if (stMems.vnMemCtr[memc] > 100*stTrack.CntMem) stMems.vnMemCtr[memc] = 100*stTrack.CntMem;
                if (stMems.vnStableCtr[memc] < 0) stMems.vnStableCtr[memc] = 1; else stMems.vnStableCtr[memc]++;
                if (stMems.vnStableCtr[memc] > 100*(stTrack.CntStable+1)) stMems.vnStableCtr[memc] = 100*stTrack.CntStable;
                stMems.vnLostCtr[memc] = 0;
            }
            else {
                if (cnt_new) {
                    for (int trc=0; trc < cnt_new; trc++) {
                    }
                }

                stMems.vnMemCtr[memc]--;
                stMems.vnLostCtr[memc]++;
                if (stMems.vnMemCtr[memc] >= stTrack.CntMem) {

                    stMems.vnMemCtr[memc] = stTrack.CntMem;
                }

                if (stMems.vnStableCtr[memc] > 0) {
                    stMems.vnStableCtr[memc] = 0;
                }
                else {
                    stMems.vnStableCtr[memc]--;
                }

                if (stMems.vnStableCtr[memc] < -100*(stTrack.CntStable+1)) {

                    stMems.vnStableCtr[memc] = -100*stTrack.CntStable;
                }

                if (stMems.vnLostCtr[memc] > 100*(stTrack.CntLost+1)) {

                    stMems.vnLostCtr[memc] = 100*stTrack.CntLost;
                }
            }
        }

        //** Restack stable objects in the Short-Term-Memory **//
        int cnt_tmp = 0;
        for (int i=0; i < nMemsCnt; i++) {

            if (stMems.vnMemCtr[i] < 0) {

                continue;
            }
            stMems.vnIdx[cnt_tmp] = stMems.vnIdx[i];
            stMems.vnPtsCnt[cnt_tmp] = stMems.vnPtsCnt[i];
            stMems.mnPtsIdx[cnt_tmp] = stMems.mnPtsIdx[i];
            stMems.mnRect[cnt_tmp] = stMems.mnRect[i];
            stMems.mnRCenter[cnt_tmp] = stMems.mnRCenter[i];
            stMems.mnCubic[cnt_tmp] = stMems.mnCubic[i];
            stMems.mnCCenter[cnt_tmp] = stMems.mnCCenter[i];
            stMems.mnColorHist[cnt_tmp] = stMems.mnColorHist[i];
            stMems.vnLength[cnt_tmp] = stMems.vnLength[i];
            stMems.vnStableCtr[cnt_tmp] = stMems.vnStableCtr[i];
            stMems.vnLostCtr[cnt_tmp] = stMems.vnLostCtr[i];
            stMems.vnMemCtr[cnt_tmp] = stMems.vnMemCtr[i];
            stMems.vnFound[cnt_tmp] = stMems.vnFound[i];
            cnt_tmp++;
        }
        cnt_old = cnt_tmp;

        //** Reusing unused surface indeces to the new appearing surfaces **//
        std::vector<int> mems_idx(cnt_old, 0);
        std::vector<int> mems_idx_new(cnt_new, 0);

        for (int i=0; i < cnt_old; i++) {

            mems_idx[i] = stMems.vnIdx[i];
        }

        std::sort(mems_idx.begin(), mems_idx.end());

        int cnt_tmp_tmp = 0;
        if (cnt_old > 2 && cnt_new) {

            if (mems_idx[2] > mems_idx[1]) {

                for (int i=2; i < cnt_old; i++) {

                    int diff = mems_idx[i] - mems_idx[i-1];
                    if (diff > 1) {

                        for (int j=0; j < diff-1; j++) {

                            mems_idx_new[cnt_tmp_tmp++] = mems_idx[i-1] + j+1;
                            if (cnt_tmp_tmp >= cnt_new) {

                                break;
                            }
                        }
                    }
                    if (cnt_tmp_tmp >= cnt_new) {
                        break;
                    }
                }
                if (cnt_tmp_tmp < cnt_new) {

                    for (int i=0; i < cnt_new - cnt_tmp_tmp; i++) {

                        mems_idx_new[cnt_tmp_tmp + i] = mems_idx[cnt_old-1] + i+1;
                    }
                }
            }
            else {
                printf("eeeeeeeeeeee \n");
            }
        }
        else {
            for (int i=0; i < cnt_new; i++) {

                mems_idx_new[i] = i;
            }
        }

        // Pushing new objects (unmatched segments) on the Short-Term-Memory
        for (int i=0; i < cnt_new; i++) {

            stMems.vnIdx[cnt_old + i] = mems_idx_new[i];
            stMems.vnPtsCnt[cnt_old + i] = stSurf.vnPtsCnt[objs_new_no[i]];
            stMems.mnPtsIdx[cnt_old + i] = stSurf.mnPtsIdx[objs_new_no[i]];
            stMems.mnRect[cnt_old + i] = stSurf.mnRect[objs_new_no[i]];
            stMems.mnRCenter[cnt_old + i] = stSurf.mnRCenter[objs_new_no[i]];
            stMems.mnCubic[cnt_old + i] = stSurf.mnCubic[objs_new_no[i]];
            stMems.mnCCenter[cnt_old + i] = stSurf.mnCCenter[objs_new_no[i]];
            stMems.mnColorHist[cnt_old + i] = stSurf.mnColorHist[objs_new_no[i]];
            stMems.vnLength[cnt_old + i] = stSurf.vnLength[objs_new_no[i]];
            stMems.vnStableCtr[cnt_old + i] = stSurf.vnStableCtr[objs_new_no[i]];
            stMems.vnLostCtr[cnt_old + i] = stSurf.vnLostCtr[objs_new_no[i]];
            stMems.vnMemCtr[cnt_old + i] = stSurf.vnMemCtr[objs_new_no[i]];
        }
        nMemsCnt = cnt_old + cnt_new;

        if (nMemsCnt >= nObjsNrLimit) {

            printf("Object queue exceeds object no. limit %d\n", nObjsNrLimit);
        }

        //  Making tracking map to a neighborhood matrix for surface saliencies
        const int nDsSize = static_cast<int>(map.total());
        std::vector<int> vnTrkMap(nDsSize, -1);         // Tracking Map
        //std::vector<int> vnTrkMapComp(nDsSize, -1);     // for the making of the Neighbor Matrix
        for (int i = 0; i < nMemsCnt; i++) {

            if (stMems.vnStableCtr[i] < stTrack.CntStable
                    || stMems.vnLostCtr[i] > stTrack.CntLost) {
                continue;
            }

            for (size_t j=0; j < stMems.mnPtsIdx[i].size(); j++) {

                vnTrkMap[stMems.mnPtsIdx[i][j]] = stMems.vnIdx[i];
                //vnTrkMapComp[stMems.mnPtsIdx[i][j]] = i;
            }
        }

        // Show Tracking
        m_ = Mat1f::zeros(map.size());
        for (int i=0; i < nDsSize; i++) {

            if (vnTrkMap[i] < 0) {
                continue;
            }
            m_(i) = vnTrkMap[i];
        }
    }
    else {

        memory_ = obsereved_;
        m_ = map; // until layer produces actual output
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

void SurfaceTracking::SurfPropToVecSurfaces(const SurfProp &surf_prop, std::vector<Surface> &surfaces) const
{
    for(size_t i=0; i<surf_prop.vnIdx.size(); i++) {

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

        surfaces[i] = s;
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
