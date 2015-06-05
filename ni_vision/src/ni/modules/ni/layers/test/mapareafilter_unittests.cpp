#include "ni/layers/mapareafilter.h"

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

        to_.reset(new MapAreaFilter);
        to_->Reset(config_);
        to_->IONames(config_);
    }

    LayerShared to_; ///< test object
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
    tau_values.push_back(static_cast<int>(in.total())-1);
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

        if(tau_values[i] >= static_cast<int>(in.total())) {

            EXPECT_MAT_EQ(map_filtered, Mat1f::zeros(in.size()));
        }
        else {

            EXPECT_MAT_EQ(map_filtered, in);
        }
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
    p.put(MapAreaFilter::PARAM_TAU_SIZE, 0);
    config_.Params(p);

    to_->Reconfigure(config_);

    Signal sig;
    sig.Append(NAME_IN_MAP, in);

    to_->Activate(sig);
    to_->Response(sig);

    Mat1f map_filtered = sig.MostRecentMat1f(NAME_OUT_MAP);

    EXPECT_MAT_EQ(map_filtered, in);
}

TEST_F(MapAreaFilterTest, MapVariableAreas)
{
    // break into 4 by increasing top-right and bottom-left values
    // this way we avoid neighbors matching
    const int N = 20;
    float data[N] = {1.f, 1.f, 1.f, 2.f, 3.f,
                     1.f, 1.f, 1.f, 2.f, 4.f,
                     1.f, 1.f, 1.f, 2.f, 5.f,
                     6.f, 6.f, 6.f, 7.f, 5.f};
    Mat1f in = Mat1f(4, 5, data).clone();

    PTree p;
    p.put(MapAreaFilter::PARAM_TAU_SIZE, 2);
    config_.Params(p);

    to_->Reconfigure(config_);

    Signal sig;
    sig.Append(NAME_IN_MAP, in);

    to_->Activate(sig);
    to_->Response(sig);

    Mat1f map_filtered = sig.MostRecentMat1f(NAME_OUT_MAP);

    //ELM_COUT_VAR(map_filtered);

    EXPECT_MAT_DIMS_EQ(map_filtered, in.size());
    Mat1f map_filtered_expected = in.clone();
    map_filtered_expected.setTo(2.f, in == 3.f);
    map_filtered_expected.setTo(2.f, in == 4.f);
    map_filtered_expected.setTo(2.f, in == 5.f);
    map_filtered_expected.setTo(2.f, in == 7.f);

    EXPECT_MAT_EQ(map_filtered_expected, map_filtered);
}

TEST_F(MapAreaFilterTest, MapVariableAreas_no_neighbors)
{
    const int N = 20;
    float data[N] = {1.f, 1.f, 1.f, 0.f, 3.f,
                     1.f, 1.f, 1.f, 0.f, 0.f,
                     1.f, 1.f, 1.f, 2.f, 5.f,
                     6.f, 6.f, 6.f, 7.f, 5.f};
    Mat1f in = Mat1f(4, 5, data).clone();

    PTree p;
    p.put(MapAreaFilter::PARAM_TAU_SIZE, 2);
    config_.Params(p);

    to_->Reconfigure(config_);

    Signal sig;
    sig.Append(NAME_IN_MAP, in);

    to_->Activate(sig);
    to_->Response(sig);

    Mat1f map_filtered = sig.MostRecentMat1f(NAME_OUT_MAP);

    EXPECT_MAT_DIMS_EQ(map_filtered, in.size());

    Mat1f map_filtered_expected = in.clone();

    /* We expect small segments merged into segment 1
     * segment 6 is as is
     * unassigned elements are unassigned
     * segment 3 (the island) is now unassigned
     *
     * [1, 1, 1, 0, x;
     *  1, 1, 1, 0, 0;
     *  1, 1, 1, 1, 1;
     *  6, 6, 6, 1, 1]
     */

    // We expect small segments merged into segment 1
    map_filtered_expected.setTo(1.f, in == 2.f);
    map_filtered_expected.setTo(1.f, in == 5.f);
    map_filtered_expected.setTo(1.f, in == 7.f);

    // unassigned elements remain unassigned
    EXPECT_FLOAT_EQ(0.f, map_filtered(0, 3));
    EXPECT_FLOAT_EQ(0.f, map_filtered(1, 3));
    EXPECT_FLOAT_EQ(0.f, map_filtered(1, 4));

    // segment 3 (the island) is now unassigned
    EXPECT_FLOAT_EQ(0.f, map_filtered(0, 4));
    map_filtered_expected.setTo(0.f, in == 3.f);

    EXPECT_MAT_EQ(map_filtered_expected, map_filtered);
}

TEST_F(MapAreaFilterTest, MapVariableAreas_still_too_small)
{
    const int N = 20;
    float data[N] = {1.f, 1.f, 0.f, 2.f, 3.f,
                     1.f, 1.f, 0.f, 2.f, 4.f,
                     1.f, 1.f, 0.f, 0.f, 0.f,
                     1.f, 1.f, 1.f, 7.f, 5.f};
    Mat1f in = Mat1f(4, 5, data).clone();

    PTree p;
    p.put(MapAreaFilter::PARAM_TAU_SIZE, 4);
    config_.Params(p);

    to_->Reconfigure(config_);

    Signal sig;
    sig.Append(NAME_IN_MAP, in);

    to_->Activate(sig);
    to_->Response(sig);

    Mat1f map_filtered = sig.MostRecentMat1f(NAME_OUT_MAP);

    EXPECT_MAT_DIMS_EQ(map_filtered, in.size());

    Mat1f map_filtered_expected = in.clone();

    /* We expect small segments merged into segment 1
     * unassigned elements remain unassigned
     * and
     * island segments are now unassigned
     *
     * [1, 1, 0, x, x;
     *  1, 1, 0, 0, x;
     *  1, 1, 0, 0, 0;
     *  1, 1, 1, 1, 1]
     */

    // We expect small segments merged into segment 1
    map_filtered_expected.setTo(1.f, in == 5.f);
    map_filtered_expected.setTo(1.f, in == 7.f);

    // unassigned elements remain unassigned
    // and
    // island segments are now unassigned
    EXPECT_EQ(0, countNonZero(map_filtered.rowRange(0, 3).colRange(2, 5)));

    EXPECT_FLOAT_EQ(0.f, map_filtered(0, 4));
    map_filtered_expected.setTo(0.f, in == 2.f);
    map_filtered_expected.setTo(0.f, in == 3.f);
    map_filtered_expected.setTo(0.f, in == 4.f);

    EXPECT_MAT_EQ(map_filtered_expected, map_filtered);
}



TEST_F(MapAreaFilterTest, MapVariableAreas_all_large)
{
    // break into 4 by increasing top-right and bottom-left values
    // this way we avoid neighbors matching
    const int N = 20;
    float data[N] = {1.f, 1.f, 1.f, 2.f, 3.f,
                     1.f, 1.f, 1.f, 2.f, 3.f,
                     1.f, 1.f, 1.f, 2.f, 3.f,
                     6.f, 6.f, 6.f, 6.f, 3.f};
    Mat1f in = Mat1f(4, 5, data).clone();

    PTree p;
    p.put(MapAreaFilter::PARAM_TAU_SIZE, 2);
    config_.Params(p);

    to_->Reconfigure(config_);

    Signal sig;
    sig.Append(NAME_IN_MAP, in);

    to_->Activate(sig);
    to_->Response(sig);

    Mat1f map_filtered = sig.MostRecentMat1f(NAME_OUT_MAP);

    //ELM_COUT_VAR(map_filtered);

    EXPECT_MAT_EQ(in, map_filtered);
}

TEST_F(MapAreaFilterTest, DISABLED_MapVariableAreas_which_neighbor)
{
    // break into 4 by increasing top-right and bottom-left values
    // this way we avoid neighbors matching
    const int N = 7;
    float data[N] = {1.f, 1.f, 1.f, 2.f, 3.f, 3.f, 3.f};
    Mat1f in = Mat1f(1, 7, data).clone();

    PTree p;
    p.put(MapAreaFilter::PARAM_TAU_SIZE, 2);
    config_.Params(p);

    to_->Reconfigure(config_);

    Signal sig;
    sig.Append(NAME_IN_MAP, in);

    to_->Activate(sig);
    to_->Response(sig);

    Mat1f map_filtered = sig.MostRecentMat1f(NAME_OUT_MAP);

    Mat1f map_filtered_expected = in.clone();
    map_filtered_expected.setTo(3.f, in == 2.f);
    EXPECT_MAT_EQ(map_filtered_expected, map_filtered);
}

} // annonymous namespace for test cases and fixtures
