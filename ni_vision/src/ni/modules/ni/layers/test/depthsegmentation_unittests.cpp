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

        to_.reset(new DepthSegmentation());
        to_->Reset(config_);
        to_->IONames(config_);
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

TEST_F(DepthSegmentationTest, Stimulus_single_element)
{
    Signal sig;
    EXPECT_NO_THROW(sig.Append(NAME_GRAD_Y, Mat1f(1, 1)));

    to_->Activate(sig);
    to_->Response(sig);

    Mat1f seg_map = sig.MostRecentMat1f(NAME_OUT_SEG_MAP);

    EXPECT_EQ(1, seg_map.rows);
    EXPECT_EQ(1, seg_map.cols);

    EXPECT_FLOAT_EQ(0.f, seg_map(DepthSegmentation::DEFAULT_LABEL_UNASSIGNED+1.f));
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

    bool comparePixels(float current, float neighbor) const
    {
        return DepthSegmentation::comparePixels(current, neighbor);
    }

    Mat1i group(const cv::Mat1f &g) const
    {
        return DepthSegmentation::group(g);
    }

    cv::Mat1i computeSegmentNeighAdjacency(const cv::Mat1i &segment_map) const
    {
        return DepthSegmentation::computeSegmentNeighAdjacency(segment_map);
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

        to_.reset(new DepthSegmentationExposeProtected());
        to_->Reset(config_);
        to_->IONames(config_);
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

/**
 * @brief test surface labels after initial grouping
 * initialize test input as function of max gradient threshold
 * 2 elements, 2 distinct surfaces
 */
TEST_F(DepthSegmentationProtectedTest, Grouped_surfaces_2_el_2_seg)
{
    float data[2] = {0.0f, 1.0f};
    Mat1f in = Mat1f(1, 2, data).clone();
    in *= max_grad_;

    Mat1i seg_map = to_->group(in);

    ASSERT_NE(seg_map(0), seg_map(1));
    EXPECT_FLOAT_EQ(1.f, seg_map(0));
    EXPECT_FLOAT_EQ(2.f, seg_map(1));
}

/**
 * @brief test surface labels after initial grouping
 * initialize test input as function of max gradient threshold
 * 2 elements, 1 distinct surfaces
 */
TEST_F(DepthSegmentationProtectedTest, Grouped_surfaces_2_el_1_seg)
{
    float data[2] = {0.0f, 0.8f};
    Mat1f in = Mat1f(1, 2, data).clone();
    in *= max_grad_;

    Mat1i seg_map = to_->group(in);

    ASSERT_FLOAT_EQ(seg_map(0), seg_map(1));
    EXPECT_FLOAT_EQ(1.f, seg_map(0));
    EXPECT_FLOAT_EQ(1.f, seg_map(1));
}

/**
 * @brief test surface labels after initial grouping
 * initialize test input as function of max gradient threshold
 * 4 elements, 4 distinct surfaces
 */
TEST_F(DepthSegmentationProtectedTest, Grouped_surfaces_4_el_4_seg)
{
    // break into 4 by increasing top-right and bottom-left values
    // this way we avoid neighbors matching
    float data[4] = {0.0f, 2.0f, 2.0, 0.5f};
    Mat1f in = Mat1f(2, 2, data).clone();
    in *= max_grad_;

    Mat1i seg_map = to_->group(in);

    for(size_t i=0; i<seg_map.total(); i++) {

        for(size_t j=0; j<seg_map.total(); j++) {

            if(i!=j) {

                ASSERT_NE(seg_map(i), seg_map(j));
            }
        }
    }
}

/**
 * @brief test surface labels after initial grouping
 * initialize test input as function of max gradient threshold
 * 4 elements, 2 distinct surfaces
 */
TEST_F(DepthSegmentationProtectedTest, Grouped_surfaces_4_el_2_seg)
{
    // break into 4 by increasing top-right and bottom-left values
    // this way we avoid neighbors matching
    float data[4] = {0.0f, 2.0f, 2.0, 1.2f};
    Mat1f in = Mat1f(2, 2, data).clone();
    in *= max_grad_;

    Mat1i seg_map = to_->group(in);

    // we expect first element to be assigned a surface
    // and the rest be assigned another.
    ASSERT_NE(seg_map(0), seg_map(1));

    for(size_t i=2; i<seg_map.total(); i++) {

        EXPECT_FLOAT_EQ(seg_map(1), seg_map(i));
    }
}

} // annonymous namespace for test cases and fixtures
