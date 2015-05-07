#include "ni/layers/prunesmallsegments.h"

#include "gtest/gtest.h"

#include <memory>

#include "elm/core/exception.h"
#include "elm/core/layerconfig.h"
#include "elm/core/signal.h"
#include "elm/ts/layer_assertions.h"
#include "elm/ts/layer_feat_transf_assertions.h"
#include "elm/ts/mat_assertions.h"
#include "elm/core/debug_utils.h"

using namespace std;
using namespace cv;
using namespace elm;
using namespace ni;

namespace {

ELM_INSTANTIATE_LAYER_TYPED_TEST_CASE_P(PruneSmallSegments);

const std::string NAME_IN_MAP = "map_in";
const std::string NAME_OUT_MAP = "map_out";

class PruneSmallSegmentsTest : public ::testing::Test
{
protected:
    virtual void SetUp()
    {
        config_ = LayerConfig();

        // params
        PTree p;
        p.put(PruneSmallSegments::PARAM_MIN_SIZE, 2);
        config_.Params(p);

        // IO
        LayerIONames io;
        io.Input(PruneSmallSegments::KEY_INPUT_STIMULUS, NAME_IN_MAP);
        io.Output(PruneSmallSegments::KEY_OUTPUT_RESPONSE, NAME_OUT_MAP);

        to_.reset(new PruneSmallSegments);
        to_->Reset(config_);
        to_->IONames(io);
    }

    shared_ptr<base_Layer> to_; ///< test object
    LayerConfig config_;        ///< default config for tests
};

TEST_F(PruneSmallSegmentsTest, Reset_EmptyConfig)
{
    EXPECT_THROW(to_->Reset(LayerConfig()), boost::property_tree::ptree_bad_path) << "Missing required params.";
}

TEST_F(PruneSmallSegmentsTest, Response_dims)
{
    for(int r=2; r<10; r++) {

        for(int c=2; c<10; c++) {

            Mat1f in = Mat1f(r, c);
            for(size_t i=0; i<in.total(); i++) {

                in(i) = static_cast<float>(i);
            }

            Signal sig;
            sig.Append(NAME_IN_MAP, in);

            to_->Activate(sig);
            to_->Response(sig);

            EXPECT_MAT_DIMS_EQ(sig.MostRecentMat1f(NAME_OUT_MAP), Size2i(c, r));
        }
    }
}

TEST_F(PruneSmallSegmentsTest, Map_constant)
{
    Mat1f in = Mat1f(320, 240, 50.f);

    std::vector<int> tau_values;
    tau_values.push_back(0);
    tau_values.push_back(1);
    tau_values.push_back(in.rows*2);
    tau_values.push_back(static_cast<int>(in.total()));

    for(size_t i=0; i<tau_values.size(); i++)
    {
        PTree p;
        p.put(PruneSmallSegments::PARAM_MIN_SIZE, tau_values[i]);
        config_.Params(p);
        to_->Reconfigure(config_);

        Signal sig;
        sig.Append(NAME_IN_MAP, in);

        to_->Activate(sig);
        to_->Response(sig);

        Mat1f map_filtered = sig.MostRecentMat1f(NAME_OUT_MAP);

        EXPECT_MAT_EQ(map_filtered, in);
    }
}

TEST_F(PruneSmallSegmentsTest, MapVariableAreas)
{
    // break into 4 by increasing top-right and bottom-left values
    // this way we avoid neighbors matching
    const int N = 20;
    float data[N] = {1.f, 1.f, 1.f, 2.f, 3.f,
                     1.f, 1.f, 1.f, 2.f, 4.f,
                     1.f, 1.f, 1.f, 2.f, 5.f,
                     6.f, 6.f, 6.f, 7.f, 5.f};
    Mat1f in = Mat1f(4, 5, data).clone();

    to_->Reconfigure(config_);

    Signal sig;
    sig.Append(NAME_IN_MAP, in);

    to_->Activate(sig);
    to_->Response(sig);

    Mat1f map_filtered = sig.MostRecentMat1f(NAME_OUT_MAP);

    //ELM_COUT_VAR(map_filtered);

    EXPECT_MAT_DIMS_EQ(map_filtered, in.size());

    float expected[N] = {1.f, 1.f, 1.f, 2.f, 0.f,
                         1.f, 1.f, 1.f, 2.f, 0.f,
                         1.f, 1.f, 1.f, 2.f, 5.f,
                         6.f, 6.f, 6.f, 0.f, 5.f};
    EXPECT_MAT_EQ(map_filtered, Mat1f(4, 5, expected).clone());
}

TEST_F(PruneSmallSegmentsTest, MapVariableAreas_all_large)
{
    // break into 4 by increasing top-right and bottom-left values
    // this way we avoid neighbors matching
    const int N = 20;
    float data[N] = {1.f, 1.f, 1.f, 2.f, 3.f,
                     1.f, 1.f, 1.f, 2.f, 3.f,
                     1.f, 1.f, 1.f, 2.f, 3.f,
                     6.f, 6.f, 6.f, 6.f, 3.f};
    Mat1f in = Mat1f(4, 5, data).clone();

    to_->Reconfigure(config_);

    Signal sig;
    sig.Append(NAME_IN_MAP, in);

    to_->Activate(sig);
    to_->Response(sig);

    Mat1f map_filtered = sig.MostRecentMat1f(NAME_OUT_MAP);

    //ELM_COUT_VAR(map_filtered);

    EXPECT_MAT_EQ(in, map_filtered);
}

} // annonymous namespace for test cases and fixtures
