#include "ni/layers/mapareafilter.h"

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

ELM_INSTANTIATE_LAYER_TYPED_TEST_CASE_P(MapAreaFilter);
ELM_INSTANTIATE_LAYER_FEAT_TRANSF_TYPED_TEST_CASE_P(MapAreaFilter);

const std::string NAME_IN_MAP = "map_in";
const std::string NAME_OUT_MAP = "map_out";

class MapAreaFilterTest : public ::testing::Test
{
protected:
    virtual void SetUp()
    {
        config_ = LayerConfig();

        // params
        PTree p;
        config_.Params(p);

        // IO
        config_.Input(MapAreaFilter::KEY_INPUT_STIMULUS, NAME_IN_MAP);
        config_.Output(MapAreaFilter::KEY_OUTPUT_RESPONSE, NAME_OUT_MAP);

        to_.reset(new MapAreaFilter(config_));
    }

    shared_ptr<base_Layer> to_; ///< test object
    LayerConfig config_;        ///< default config for tests
};

TEST_F(MapAreaFilterTest, Reset_EmptyConfig)
{
    EXPECT_NO_THROW(to_->Reset(LayerConfig())) << "All params are optional, no?";
}

TEST_F(MapAreaFilterTest, Response_dims)
{
    PTree p;
    p.put(MapAreaFilter::PARAM_TAU_SIZE, 1);
    config_.Params(p);

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

TEST_F(MapAreaFilterTest, Map_constant)
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
        p.put(MapAreaFilter::PARAM_TAU_SIZE, tau_values[i]);
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

TEST_F(MapAreaFilterTest, Map_4_el_4_seg)
{
    // break into 4 by increasing top-right and bottom-left values
    // this way we avoid neighbors matching
    const int NB_SURFACES = 4;
    float data[NB_SURFACES] = {1.0f, 2.0f,
                               3.0f, 4.0f};
    Mat1f in = Mat1f(2, 2, data).clone();

    PTree p;
    p.put(MapAreaFilter::PARAM_TAU_SIZE, 1);
    config_.Params(p);

    to_->Reconfigure(config_);

    Signal sig;
    sig.Append(NAME_IN_MAP, in);

    to_->Activate(sig);
    to_->Response(sig);

    Mat1f map_filtered = sig.MostRecentMat1f(NAME_OUT_MAP);

    EXPECT_MAT_EQ(map_filtered, in);
}


} // annonymous namespace for test cases and fixtures
