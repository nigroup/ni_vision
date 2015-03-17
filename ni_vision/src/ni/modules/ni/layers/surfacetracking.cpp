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

const string SurfaceTracking::KEY_INPUT_BGR_IMAGE   = "bgr";
const string SurfaceTracking::KEY_INPUT_CLOUD       = "points";
const string SurfaceTracking::KEY_INPUT_MAP         = "map";

#include <boost/assign/list_of.hpp>
template <>
elm::MapIONames LayerAttr_<SurfaceTracking>::io_pairs = boost::assign::map_list_of
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
}

void SurfaceTracking::Activate(const Signal &signal)
{

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
            computeColorHist(bgr, tmp, 8, hist);

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



