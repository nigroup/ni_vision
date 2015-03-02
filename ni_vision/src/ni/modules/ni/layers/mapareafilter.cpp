#include "ni/layers/mapareafilter.h"

#include "ni/layers/depthsegmentation.h"

#include "elm/core/cv/mat_vector_utils.h"
#include "elm/core/debug_utils.h"
#include "elm/core/exception.h"
#include "elm/core/graph/graphattr.h"
#include "elm/core/layerconfig.h"
#include "elm/core/signal.h"
#include "elm/ts/layerattr_.h"

#include "ni/core/surface.h"

using namespace std;
using namespace cv;
using namespace elm;
using namespace ni;

/** Define parameters, non-integral defaults and any remaining I/O keys
  */
// paramters
const string MapAreaFilter::PARAM_TAU_SIZE = "tau_size";

// defaults
const int MapAreaFilter::DEFAULT_TAU_SIZE           = 200;
const int MapAreaFilter::DEFAULT_LABEL_UNASSIGNED   = 0;


#include <boost/assign/list_of.hpp>
template <>
elm::MapIONames LayerAttr_<MapAreaFilter>::io_pairs = boost::assign::map_list_of
        ELM_ADD_INPUT_PAIR(detail::BASE_SINGLE_INPUT_FEATURE_LAYER__KEY_INPUT_STIMULUS)
        ELM_ADD_OUTPUT_PAIR(detail::BASE_MATOUTPUT_LAYER__KEY_OUTPUT_RESPONSE)
        ;

MapAreaFilter::~MapAreaFilter()
{
}

MapAreaFilter::MapAreaFilter()
    : base_FeatureTransformationLayer()
{
    Clear();
}

MapAreaFilter::MapAreaFilter(const LayerConfig &config)
    : base_FeatureTransformationLayer(config)
{
    Clear();
    Reconfigure(config);
    IONames(config);
}

void MapAreaFilter::Clear()
{
    m_ = Mat1f();
}

void MapAreaFilter::Reset(const LayerConfig &config)
{
    Reconfigure(config);
}

void MapAreaFilter::Reconfigure(const LayerConfig &config)
{
    PTree p = config.Params();
    tau_size_ = p.get<int>(PARAM_TAU_SIZE, DEFAULT_TAU_SIZE);
}

cv::Mat1f sum_pixels(const cv::Mat1f& img, const cv::Mat1b &mask)
{
    return cv::Mat1f(1, 1, static_cast<float>(cv::countNonZero(mask)));
}

cv::Mat1f mask_vertex(const cv::Mat1f& img, const cv::Mat1b &mask)
{
    return img.clone().setTo(0, mask == 0);
}

void MapAreaFilter::Activate(const Signal &signal)
{
    Mat1f map = signal.MostRecent(name_input_); // weighted gradient after thresholding

    GraphAttr seg_graph(map.clone(), map > DEFAULT_LABEL_UNASSIGNED);

    VecF seg_ids = seg_graph.VerticesIds();
    Mat1f seg_sizes = elm::Reshape(seg_graph.applyVerticesToMap(sum_pixels));

    for(size_t i=0; i<seg_ids.size(); i++) {

        ELM_COUT_VAR(elm::to_string(seg_graph.VerticesIds()));
        ELM_COUT_VAR(elm::Reshape(seg_graph.applyVerticesToMap(sum_pixels)));

        float cur_seg_id = seg_ids[i];
        float cur_seg_size = seg_sizes(i);

        if(cur_seg_size <= tau_size_) {

            try {

                VecF neigh_ids = seg_graph.getNeighbors(cur_seg_id);
                int nb_neighbors = static_cast<int>(neigh_ids.size());

                if(nb_neighbors < 1) {

                    continue; // nothing to do
                }

                // create list of neighbors
                std::vector<Surface> neighbors(nb_neighbors);
                VecF neigh_sizes(nb_neighbors); // keep a list of sizes for sorting

                for(int j=0; j<nb_neighbors; j++) {

                    float neigh_size = seg_sizes(
                                seg_graph.VertexIndex(neigh_ids[j]));

                    Surface neighbor;
                    neighbor.id(static_cast<int>(neigh_ids[j]));
                    neighbor.pixelIndices(VecI(static_cast<int>(neigh_size)));

                    neighbors[j] = neighbor;

                    neigh_sizes[j] = neigh_size;
                }

                // sort neighbors according to size/area of pixels covered
                Mat1i neigh_sizes_sorted_idx;
                cv::sortIdx(neigh_sizes, neigh_sizes_sorted_idx, SORT_ASCENDING);

                std::vector<Surface> neighbors_sorted(nb_neighbors);
                for(size_t j=0; j<neigh_sizes_sorted_idx.total(); j++) {
                    neighbors_sorted[neigh_sizes_sorted_idx(j)] = neighbors[j];
                }

                // merge small neighbors into current segment
                bool too_large = false;
                for(int j=0; j<nb_neighbors && !too_large; j++) {

                    // access sorted list
                    Surface neighbor = neighbors_sorted[j];

                    bool too_large = neighbor.pixelCount() > tau_size_;
                    if(!too_large) {

                        //ELM_COUT_VAR("contractEdges(" << neighbor.id() << "," << cur_seg_id << ")");
                        seg_graph.contractEdges(neighbor.id(), cur_seg_id);
                        seg_sizes(i) = cur_seg_size + neighbor.pixelCount();
                        //ELM_COUT_VAR(elm::to_string(seg_graph.VerticesIds()));
                    }
                }


                // find largest neighbor that is actually large enough

                Surface largest_neigh = neighbors_sorted[nb_neighbors-1];
                if(largest_neigh.pixelCount() > tau_size_)
                {
                    // merge current segment into larget neighbor
                    seg_graph.contractEdges(cur_seg_id, largest_neigh.id());

                    bool found = false;
                    for(size_t j=0; j<seg_sizes.total() && !found; j++) {

                        if(seg_ids[j] == largest_neigh.id()) {

                            seg_sizes(j) = static_cast<float>(largest_neigh.pixelCount());
                            found = true;
                        }
                    }
                }
            }
            catch(ExceptionKeyError &e) {

                //ELM_COUT_VAR(e.what());
                //ELM_COUT_VAR(cur_seg_id << " gone.");
            }
        } // large enough?
    } // each segment

    // replace with getter to Graph's underlying map image
    VecMat1f masked_maps = seg_graph.applyVerticesToMap(mask_vertex);
    m_ = Mat1f::zeros(map.size());
    for(size_t i=0; i<masked_maps.size(); i++) {

        m_ += masked_maps[i];
    }
}
