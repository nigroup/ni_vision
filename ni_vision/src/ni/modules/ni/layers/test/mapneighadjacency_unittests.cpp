#include "ni/layers/mapneighadjacency.h"

#include "gtest/gtest.h"

#include <memory>

#include "elm/core/exception.h"
#include "elm/core/layerconfig.h"
#include "elm/core/signal.h"
#include "elm/ts/layer_assertions.h"
#include "elm/ts/layer_feat_transf_assertions.h"
#include "elm/core/debug_utils.h"

using namespace std;
using namespace cv;
using namespace elm;
using namespace ni;

namespace {

ELM_INSTANTIATE_LAYER_TYPED_TEST_CASE_P(MapNeighAdjacency);
ELM_INSTANTIATE_LAYER_FEAT_TRANSF_TYPED_TEST_CASE_P(MapNeighAdjacency);

const std::string NAME_IN_SEG_MAP = "seg_map";
const std::string NAME_OUT_ADJ    = "adj";

class MapNeighAdjacencyTest : public ::testing::Test
{
protected:
    virtual void SetUp()
    {
        config_ = LayerConfig();

        // params
        PTree p;
        config_.Params(p);

        // IO
        config_.Input(MapNeighAdjacency::KEY_INPUT_STIMULUS, NAME_IN_SEG_MAP);
        config_.Output(MapNeighAdjacency::KEY_OUTPUT_RESPONSE, NAME_OUT_ADJ);

        to_.reset(new MapNeighAdjacency(config_));
    }

    shared_ptr<base_Layer> to_; ///< test object
    LayerConfig config_;        ///< default config for tests
};

TEST_F(MapNeighAdjacencyTest, Reset_EmptyConfig)
{
    EXPECT_NO_THROW(to_->Reset(LayerConfig())) << "All params are optional, no?";
}

TEST_F(MapNeighAdjacencyTest, SegmentNeighAdj_4_el_2_seg)
{
    // break into 4 by increasing top-right and bottom-left values
    // this way we avoid neighbors matching
    float data[4] = {1.0f, 2.0f,
                     2.0f, 2.0f};
    Mat1f in = Mat1f(2, 2, data).clone();

    Signal sig;
    sig.Append(NAME_IN_SEG_MAP, in);

    to_->Activate(sig);
    to_->Response(sig);

    Mat1i seg_neigh_adj = sig.MostRecentMat1f(NAME_OUT_ADJ);

    EXPECT_EQ(2, seg_neigh_adj.rows);
    EXPECT_EQ(2, seg_neigh_adj.cols);

    EXPECT_MAT_EQ(seg_neigh_adj, seg_neigh_adj.t()) << "adj. matrix is not symmetric";

    // verify adj. has all-zeros diagonal
    for(int r=0; r<seg_neigh_adj.rows; r++) {

        EXPECT_FLOAT_EQ(0.f, seg_neigh_adj(r, r)) << "Non-zero value in diagonal.";
    }

    // verify values in adj. matrix
    EXPECT_FLOAT_EQ(0.f, seg_neigh_adj(0, 0));
    EXPECT_FLOAT_EQ(1.f, seg_neigh_adj(0, 1));
    EXPECT_FLOAT_EQ(1.f, seg_neigh_adj(1, 0));
    EXPECT_FLOAT_EQ(0.f, seg_neigh_adj(1, 1));
}

TEST_F(MapNeighAdjacencyTest, SegmentNeighAdj_4_el_4_seg)
{
    // break into 4 by increasing top-right and bottom-left values
    // this way we avoid neighbors matching
    const int NB_SURFACES = 4;
    float data[NB_SURFACES] = {1.0f, 2.0f,
                               3.0f, 4.0f};
    Mat1f in = Mat1f(2, 2, data).clone();

    Signal sig;
    sig.Append(NAME_IN_SEG_MAP, in);

    to_->Activate(sig);
    to_->Response(sig);

    Mat1i seg_neigh_adj = sig.MostRecentMat1f(NAME_OUT_ADJ);

    EXPECT_MAT_DIMS_EQ(seg_neigh_adj, Size2i(NB_SURFACES, NB_SURFACES));

    EXPECT_MAT_EQ(seg_neigh_adj, seg_neigh_adj.t()) << "adj. matrix is not symmetric";

    // verify adj. has all-zeros diagonal
    for(int r=0; r<seg_neigh_adj.rows; r++) {

        EXPECT_FLOAT_EQ(0.f, seg_neigh_adj(r, r)) << "Non-zero value in diagonal.";
    }

    // verify each surface/segment has two neighbors
    for(int r=0; r<seg_neigh_adj.rows; r++) {

        EXPECT_FLOAT_EQ(2.f, cv::sum(seg_neigh_adj.row(r))[0]);
    }

    /* verify connectivity - we only need to verify whom the surface is not connected to
     * since the surface neighborhood adjacency describes an undirected graph
     * there is some redundancy in the assertions
     * let's keep them anyway for a better overview of what we're testing
     */
    EXPECT_FLOAT_EQ(0.f, seg_neigh_adj(0, 3));
    EXPECT_FLOAT_EQ(0.f, seg_neigh_adj(1, 2));
    EXPECT_FLOAT_EQ(0.f, seg_neigh_adj(2, 1)); // redunant assertion
    EXPECT_FLOAT_EQ(0.f, seg_neigh_adj(0, 3)); // redunant assertion

}


} // annonymous namespace for test cases and fixtures
