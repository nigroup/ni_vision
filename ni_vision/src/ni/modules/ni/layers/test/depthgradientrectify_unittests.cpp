#include "ni/layers/depthgradientrectify.h"

#include "gtest/gtest.h"

#include <memory>

#include "elm/core/exception.h"
#include "elm/core/layerconfig.h"
#include "elm/core/signal.h"
#include "elm/ts/layer_assertions.h"

using namespace std;
using namespace cv;
using namespace elm;
using namespace ni;

namespace {

ELM_INSTANTIATE_LAYER_TYPED_TEST_CASE_P(DepthGradientRectify);

const std::string NAME_GRAD_X       = "x";
const std::string NAME_GRAD_Y       = "y";
const std::string NAME_GRAD_SMOOTH  = "s";
const std::string NAME_OUT_GRAD_RECT = "out";

class DepthGradientRectifyTest : public ::testing::Test
{
protected:
    virtual void SetUp()
    {
        to_.reset(new DepthGradientRectify());

        config_ = LayerConfig();

        // params
        PTree p;
        p.add(DepthGradientRectify::PARAM_MAX_GRAD, 0.04f); // value from paper and first version
        config_.Params(p);

        // IO
        LayerIONames io;
        io.Input(DepthGradientRectify::KEY_INPUT_GRAD_X, NAME_GRAD_X);
        io.Input(DepthGradientRectify::KEY_INPUT_GRAD_Y, NAME_GRAD_Y);
        io.Input(DepthGradientRectify::KEY_INPUT_GRAD_SMOOTH, NAME_GRAD_SMOOTH);
        io.Output(DepthGradientRectify::KEY_OUTPUT_RESPONSE, NAME_OUT_GRAD_RECT);

        to_.reset(new DepthGradientRectify(config_));
        to_->IONames(io);
    }

    shared_ptr<base_Layer> to_; ///< test object
    LayerConfig config_;        ///< default config for tests
};

TEST_F(DepthGradientRectifyTest, Reset_EmptyConfig)
{
    EXPECT_THROW(to_->Reset(LayerConfig()), boost::property_tree::ptree_bad_path) << "Missing required params.";
}

TEST_F(DepthGradientRectifyTest, Response_dims)
{
    const int R=10;
    const int C=10;

    for(int r=2; r<R; r++) {

        for(int c=2; c<C; c++) {

            Signal sig;
            sig.Append(NAME_GRAD_X, Mat1f(r, c, 0.f));
            sig.Append(NAME_GRAD_Y, Mat1f(r, c, 0.f));
            sig.Append(NAME_GRAD_SMOOTH, Mat1f(r, c, 0.f));

            to_->Activate(sig);
            to_->Response(sig);

            Mat1f seg_map = sig.MostRecentMat1f(NAME_OUT_GRAD_RECT);

            EXPECT_EQ(r, seg_map.rows);
            EXPECT_EQ(c, seg_map.cols);
        }
    }
}

} // annonymous namespace for test cases and fixtures

