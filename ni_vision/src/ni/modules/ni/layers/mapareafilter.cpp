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
#include "ni/core/graph/VertexOpSegmentSize.h"

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

int MapAreaFilter::getNeighbors(float vtx_id, const GraphAttr &seg_graph, std::vector<Surface> &neighbors, VecF &neigh_sizes) const
{
    VecF neigh_ids = seg_graph.getNeighbors(vtx_id);
    int nb_neighbors = static_cast<int>(neigh_ids.size());

    // create list of neighbors
    neighbors = std::vector<Surface>(nb_neighbors);
    neigh_sizes = VecF(nb_neighbors); // keep a list of sizes for sorting

    for(int j=0; j<nb_neighbors; j++) {

        float neigh_size = seg_graph.getAttributes(neigh_ids[j])(0);

        Surface neighbor;
        neighbor.id(static_cast<int>(neigh_ids[j]));
        neighbor.pixelIndices(VecI(static_cast<int>(neigh_size)));

        neighbors[j] = neighbor;

        neigh_sizes[j] = neigh_size;
    }

    return nb_neighbors;
}

cv::Mat1f mask_vertex(const cv::Mat1f& img, const cv::Mat1b &mask)
{
    Mat1b mask_inverted;
    cv::bitwise_not(mask, mask_inverted);
    return img.clone().setTo(0, mask_inverted);
}

void MapAreaFilter::Activate(const Signal &signal)
{
    Mat1f map = signal.MostRecent(name_input_); // weighted gradient after thresholding

    GraphAttr seg_graph(map.clone(), map > DEFAULT_LABEL_UNASSIGNED);

    VecF seg_ids = seg_graph.VerticesIds();
    Mat1f seg_sizes = elm::Reshape(seg_graph.applyVerticesToMap(
                                       VertexOpSegmentSize::calcSize));

    // assign size to vector attributes
    for(size_t i=0; i<seg_sizes.total(); i++) {

        seg_graph.addAttributes(seg_ids[i], seg_sizes.col(i));
    }

    for(size_t i=0; i<seg_ids.size(); i++) {

        //ELM_COUT_VAR(elm::to_string(seg_graph.VerticesIds()));

        float cur_seg_id = seg_ids[i];

        try {

            float cur_seg_size = seg_graph.getAttributes(cur_seg_id)(0);

            if(cur_seg_size <= tau_size_) {

                // create list of neighbors
                std::vector<Surface> neighbors;
                VecF neigh_sizes; // keep a list of sizes for sorting
                int nb_neighbors = getNeighbors(cur_seg_id, seg_graph, neighbors, neigh_sizes);

                if(nb_neighbors < 1) {

                    continue; // nothing to do
                }
                //ELM_COUT_VAR(nb_neighbors);

                // sort neighbors according to size/area of pixels covered
                Mat1i neigh_sizes_sorted_idx;
                cv::sortIdx(neigh_sizes, neigh_sizes_sorted_idx, SORT_ASCENDING);

                std::vector<Surface> neighbors_sorted(nb_neighbors);
                for(size_t j=0; j<neigh_sizes_sorted_idx.total(); j++) {

                    int new_idx = static_cast<int>(neigh_sizes_sorted_idx(j));
                    neighbors_sorted[new_idx] = neighbors[j];
                }

                // merge small neighbors into current segment
                bool too_large = false;
                for(int j=0; j<nb_neighbors && !too_large; j++) {

                    // access sorted list
                    Surface neighbor = neighbors_sorted[j];

                    if(neighbor.pixelCount() <= tau_size_) {

                        //ELM_COUT_VAR("contractEdges(" << neighbor.id() << "," << cur_seg_id << ")");
                        seg_graph.contractEdges(neighbor.id(), cur_seg_id);
                        float new_cur_size = cur_seg_size + static_cast<float>(neighbor.pixelCount());
                        seg_graph.addAttributes(cur_seg_id, Mat1f(1, 1, new_cur_size));
                        //ELM_COUT_VAR(elm::to_string(seg_graph.VerticesIds()));
                    }
                }

                // find largest neighbor that is actually large enough

                Surface largest_neigh = neighbors_sorted[nb_neighbors-1];
                if(largest_neigh.pixelCount() > tau_size_)
                {
                    // merge current segment into larget neighbor
                    //ELM_COUT_VAR("contractEdges(" << cur_seg_id << "," << largest_neigh.id() << ")");

                    seg_graph.contractEdges(cur_seg_id, largest_neigh.id());

                    bool found = false;
                    for(size_t j=0; j<seg_sizes.total() && !found; j++) {

                        if(seg_ids[j] == largest_neigh.id()) {

                            //seg_sizes(j) = static_cast<float>(largest_neigh.pixelCount());
                            float new_size = cur_seg_size + static_cast<float>(largest_neigh.pixelCount());
                            seg_graph.addAttributes(largest_neigh.id(), Mat1f(1, 1, new_size));
                            found = true;
                        }
                    }
                }
                }
            } // large enough?

        catch(ExceptionKeyError &e) {

            //ELM_COUT_VAR(e.what());
            //ELM_COUT_VAR(cur_seg_id << " gone.");
        }
    } // each segment

    // replace with getter to Graph's underlying map image
    VecMat1f masked_maps = seg_graph.applyVerticesToMap(mask_vertex);
    m_ = Mat1f::zeros(map.size());
    for(size_t i=0; i<masked_maps.size(); i++) {

        m_ += masked_maps[i];
    }
}
