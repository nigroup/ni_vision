#include "ni/layers/depthgradient.h"

#include "gtest/gtest.h"

#include <memory>

#include "elm/core/cv/mat_utils.h"
#include "elm/core/exception.h"
#include "elm/core/layerconfig.h"
#include "elm/core/signal.h"
#include "elm/ts/layer_assertions.h"
#include "elm/ts/mat_assertions.h"
#include "elm/core/debug_utils.h"

using namespace std;
using namespace cv;
using namespace elm;
using namespace ni;

namespace {

ELM_INSTANTIATE_LAYER_TYPED_TEST_CASE_P(DepthGradient);

const std::string NAME_IN        = "in";
const std::string NAME_GRAD_X    = "x";
const std::string NAME_GRAD_Y    = "y";

class DepthGradientTest : public ::testing::Test
{
public:

protected:
    virtual void SetUp()
    {
        to_.reset(new DepthGradient());

        config_ = LayerConfig();

        // params
        PTree params;
        config_.Params(params);

        // IO
        config_.Input(DepthGradient::KEY_INPUT_STIMULUS, NAME_IN);
        config_.Output(DepthGradient::KEY_OUTPUT_GRAD_X, NAME_GRAD_X);
        config_.Output(DepthGradient::KEY_OUTPUT_GRAD_Y, NAME_GRAD_Y);

        to_.reset(new DepthGradient(config_));
    }

    unique_ptr<base_Layer> to_; ///< test object
    LayerConfig config_;        ///< default config for tests
};

TEST_F(DepthGradientTest, Reset_EmptyConfig)
{
    EXPECT_NO_THROW(to_->Reset(LayerConfig())) << "All params are optional, no?";
}

TEST_F(DepthGradientTest, Response_exists)
{
    Mat1f in(10, 10, 1.f);

    Signal sig;
    sig.Append(NAME_IN, in);

    to_->Activate(sig);

    EXPECT_FALSE(sig.Exists(NAME_GRAD_X));
    EXPECT_FALSE(sig.Exists(NAME_GRAD_Y));

    to_->Response(sig);

    EXPECT_TRUE(sig.Exists(NAME_GRAD_X));
    EXPECT_TRUE(sig.Exists(NAME_GRAD_Y));
}

TEST_F(DepthGradientTest, Response_dims)
{
    const int R=10;
    const int C=10;

    for(int r=2; r<R; r++) {

        for(int c=2; c<C; c++) {

            Mat1f in(r, c, 1.f);

            Signal sig;
            sig.Append(NAME_IN, in);

            to_->Activate(sig);
            to_->Response(sig);

            // once for x and repeat for y
            Mat1f gradient = sig.MostRecentMat1f(NAME_GRAD_X);

            EXPECT_EQ(r, gradient.rows);
            EXPECT_EQ(c, gradient.cols);

            gradient = sig.MostRecentMat1f(NAME_GRAD_Y);

            EXPECT_EQ(r, gradient.rows);
            EXPECT_EQ(c, gradient.cols);
        }
    }
}

TEST_F(DepthGradientTest, Invalid_input)
{
    for(int r=0; r<2; r++) {

        for(int c=0; c<2; c++) {

            Mat1f in(r, c, 1.f);

            Signal sig;
            sig.Append(NAME_IN, in);

            EXPECT_THROW(to_->Activate(sig), ExceptionBadDims);
        }
    }
}

TEST_F(DepthGradientTest, Param_default_weight_zero)
{
    Mat1f in(2, 2, 1.f);
    in.col(in.cols-1).setTo(1000.f);

    Signal sig;
    sig.Append(NAME_IN, in);

    to_->Activate(sig);
    to_->Response(sig);

    Mat1f grad_x = sig.MostRecentMat1f(NAME_GRAD_X);

    EXPECT_MAT_EQ(Mat1f(2, 1, (1000.f-1.f)/1000.f), grad_x.col(1));

    EXPECT_EQ(2, countNonZero(elm::isnan(grad_x)));

    Mat1f grad_y = sig.MostRecentMat1f(NAME_GRAD_Y);

    EXPECT_MAT_EQ(Mat1f(1, 2, 0.f), grad_y.row(1));
\
    EXPECT_EQ(2, countNonZero(elm::isnan(grad_y)));
}

TEST_F(DepthGradientTest, Param_weight)
{
    const float WEIGHT = 2.f;

    PTree params;
    params.add(DepthGradient::PARAM_GRAD_WEIGHT, WEIGHT);
    config_.Params(params);
    to_.reset(new DepthGradient(config_));

    Mat1f in(2, 2, 1.f);
    in(0) = 0.f;

    Signal sig;
    sig.Append(NAME_IN, in);

    to_->Activate(sig);
    to_->Response(sig);

    Mat1f grad_x = sig.MostRecentMat1f(NAME_GRAD_X);

    EXPECT_FLOAT_EQ(1.f/3.f, grad_x(1));
    EXPECT_EQ(2, countNonZero(elm::isnan(grad_x)));
    EXPECT_EQ(2, countNonZero(elm::isnan(grad_x.col(0))));

    Mat1f grad_y = sig.MostRecentMat1f(NAME_GRAD_Y);

    EXPECT_FLOAT_EQ(1.f/3.f, grad_y(2));
    EXPECT_EQ(2, countNonZero(elm::isnan(grad_y)));
    EXPECT_EQ(2, countNonZero(elm::isnan(grad_y.row(0))));
}

TEST_F(DepthGradientTest, Grad_Y_nans)
{
    const int ROWS=3;
    const int COLS=3;

    const float NAN_VALUE=std::numeric_limits<float>::quiet_NaN();

    float data1[ROWS*COLS] = {NAN_VALUE, NAN_VALUE, NAN_VALUE,
                              1.3740001f, 1.3790001f, 1.3850001f,
                              1.3740001f, 1.3790001f, 1.3790001f
                             };
    Mat1f depth = Mat1f(ROWS, COLS, data1).clone();

    Signal sig;
    sig.Append(NAME_IN, depth);

    to_->Activate(sig);
    to_->Response(sig);

    Mat1f grad_y = sig.MostRecentMat1f(NAME_GRAD_Y);

    Mat is_nan_grad = elm::isnan(grad_y);
    EXPECT_EQ(6, cv::countNonZero(is_nan_grad));

    float data_res[3] = {0.f, 0.f, -0.0043510091f};

    EXPECT_MAT_EQ(Mat1f(1, grad_y.cols, data_res).clone(), grad_y.row(grad_y.rows-1));

}

} // annonymous namespace for test cases and fixtures
