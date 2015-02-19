#include "ni/layers/depthsegmentation.h"

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

ELM_INSTANTIATE_LAYER_TYPED_TEST_CASE_P(DepthSegmentation);
ELM_INSTANTIATE_LAYER_FEAT_TRANSF_TYPED_TEST_CASE_P(DepthSegmentation);

const std::string NAME_GRAD_Y       = "g";
const std::string NAME_OUT_SEG_MAP  = "seg_map";

class DepthSegmentationTest : public ::testing::Test
{
protected:
    virtual void SetUp()
    {
        to_.reset(new DepthSegmentation());

        config_ = LayerConfig();

        // params
        PTree p;
        p.add(DepthSegmentation::PARAM_MAX_GRAD, 0.005f); // value from paper and first version
        config_.Params(p);

        // IO
        config_.Input(DepthSegmentation::KEY_INPUT_STIMULUS, NAME_GRAD_Y);
        config_.Output(DepthSegmentation::KEY_OUTPUT_RESPONSE, NAME_OUT_SEG_MAP);

        to_.reset(new DepthSegmentation(config_));
    }

    shared_ptr<base_Layer> to_; ///< test object
    LayerConfig config_;        ///< default config for tests
};

TEST_F(DepthSegmentationTest, Reset_EmptyConfig)
{
    EXPECT_NO_THROW(to_->Reset(LayerConfig())) << "All params are optional, no?";
}

TEST_F(DepthSegmentationTest, Response_dims)
{

}

} // annonymous namespace for test cases and fixtures
