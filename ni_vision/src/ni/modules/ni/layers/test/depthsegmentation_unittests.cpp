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
    const int R=10;
    const int C=10;

    for(int r=2; r<R; r++) {

        for(int c=2; c<C; c++) {

            Signal sig;
            sig.Append(NAME_GRAD_Y, Mat1f(r, c));

            to_->Activate(sig);
            to_->Response(sig);

            Mat1f seg_map = sig.MostRecentMat1f(NAME_OUT_SEG_MAP);

            EXPECT_EQ(r, seg_map.rows);
            EXPECT_EQ(c, seg_map.cols);
        }
    }
}

/**
 * @brief derive from DepthSegmentation and expose protected methods
 * to test individually
 */
class DepthSegmentationExposeProtected : public DepthSegmentation
{
public:
    DepthSegmentationExposeProtected()
        : DepthSegmentation()
    {
        Clear();
    }

    DepthSegmentationExposeProtected(const LayerConfig &config)
        : DepthSegmentation(config)
    {
        Clear();
        Reconfigure(config);
        IONames(config);
    }

    bool comparePixels(float current, float neighbor)
    {
        return DepthSegmentation::comparePixels(current, neighbor);
    }

    void group(const cv::Mat1f g)
    {
        DepthSegmentation::group(g);
    }
};

class DepthSegmentationProtectedTest : public ::testing::Test
{
protected:
    virtual void SetUp()
    {
        config_ = LayerConfig();

        // params
        max_grad_ = 0.005f;
        PTree p;
        p.add(DepthSegmentation::PARAM_MAX_GRAD, max_grad_); // value from paper and first version
        config_.Params(p);

        // IO
        config_.Input(DepthSegmentation::KEY_INPUT_STIMULUS, NAME_GRAD_Y);
        config_.Output(DepthSegmentation::KEY_OUTPUT_RESPONSE, NAME_OUT_SEG_MAP);

        to_.reset(new DepthSegmentationExposeProtected(config_));
    }

    shared_ptr<DepthSegmentationExposeProtected> to_; ///< test object
    LayerConfig config_;        ///< default config for tests
    float max_grad_;
};

TEST_F(DepthSegmentationProtectedTest, ComparePixels)
{
    EXPECT_TRUE(to_->comparePixels(-10.f, -10.f));
    EXPECT_TRUE(to_->comparePixels(0.f, 0.f));
    EXPECT_TRUE(to_->comparePixels(10000.f, 10000.f));

    EXPECT_FALSE(to_->comparePixels(12345.f, -12345.f));
    EXPECT_FALSE(to_->comparePixels(-12345.f, 12345.f));

    float x = 2.f;
    EXPECT_FALSE(to_->comparePixels(x, x+max_grad_));
    EXPECT_TRUE(to_->comparePixels(x, x+max_grad_-1e-5));
    EXPECT_FALSE(to_->comparePixels(x+max_grad_, x));
    EXPECT_TRUE(to_->comparePixels(x+max_grad_-1e-5, x));
}

} // annonymous namespace for test cases and fixtures
