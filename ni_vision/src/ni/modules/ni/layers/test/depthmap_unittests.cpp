#include "ni/layers/depthmap.h"

#include "gtest/gtest.h"

#include <memory>

#include "elm/core/exception.h"
#include "elm/core/layerconfig.h"
#include "elm/core/signal.h"
#include "elm/ts/layer_assertions.h"

using namespace std;
using namespace cv;
using namespace pcl;
using namespace elm;
using namespace ni;

namespace {

ELM_INSTANTIATE_LAYER_TYPED_TEST_CASE_P(DepthMap);

const std::string NAME_CLOUD        = "in";
const std::string NAME_DEPTH_MAP    = "out";

class DepthMapTest : public ::testing::Test
{
public:

protected:
    virtual void SetUp()
    {
        to_.reset(new DepthMap());

        config_ = LayerConfig();

        // params
        PTree params;
        config_.Params(params);

        config_.Input(DepthMap::KEY_INPUT_STIMULUS, NAME_CLOUD);
        config_.Output(DepthMap::KEY_OUTPUT_RESPONSE, NAME_DEPTH_MAP);

        // IO
        to_.reset(new DepthMap(config_));
    }

    shared_ptr<base_Layer> to_; ///< test object
    LayerConfig config_;        ///< default config for tests
};

TEST_F(DepthMapTest, Reset_EmptyConfig)
{
    EXPECT_NO_THROW(to_->Reset(LayerConfig())) << "All params are optional, no?";
}

TEST_F(DepthMapTest, Response_exists)
{
    CloudXYZPtr cld(new CloudXYZ());

    cld->push_back(PointXYZ(-1.f, -2.f, 1.f));
    cld->push_back(PointXYZ(-1.f, -2.f, 2.f));

    Signal sig;
    sig.Append(NAME_CLOUD, cld);

    to_->Activate(sig);

    EXPECT_FALSE(sig.Exists(NAME_DEPTH_MAP));

    to_->Response(sig);

    EXPECT_TRUE(sig.Exists(NAME_DEPTH_MAP));
}

TEST_F(DepthMapTest, Depth_larger_max)
{
    CloudXYZPtr cld(new CloudXYZ());

    cld->push_back(PointXYZ(-1.f, -2.f, 1.f));
    cld->push_back(PointXYZ(-1.f, -2.f, 2.f));
    cld->push_back(PointXYZ(-1.f, -2.f, 3.f));
    cld->push_back(PointXYZ(-1.f, -2.f, 4.f));
    cld->push_back(PointXYZ(-1.f, -2.f, 5.f));

    // sanity check to vrify effectiveness of fake point cloud data
    Mat1f tmp = PointCloud2Mat_(cld);
    ASSERT_GT(static_cast<int>(tmp.total()),
              countNonZero(tmp <= DepthMap::DEFAULT_DEPTH_MAX));

    Signal sig;
    sig.Append(NAME_CLOUD, cld);

    to_->Activate(sig);
    to_->Response(sig);

    Mat1f depth_map = sig.MostRecentMat1f(NAME_DEPTH_MAP);

    EXPECT_EQ(static_cast<int>(depth_map.total()),
              countNonZero(depth_map <= DepthMap::DEFAULT_DEPTH_MAX));

}

TEST_F(DepthMapTest, DepthMapDims)
{
    const int R=10;
    const int C=10;

    for(int r=1; r<R; r++) {

        for(int c=1; c<C; c++) {

            CloudXYZPtr cld(new CloudXYZ(c, r));

            Signal sig;
            sig.Append(NAME_CLOUD, cld);

            to_->Activate(sig);
            to_->Response(sig);

            Mat1f depth_map = sig.MostRecentMat1f(NAME_DEPTH_MAP);

            EXPECT_EQ(r, depth_map.rows);
            EXPECT_EQ(c, depth_map.cols);

        }
    }
}

TEST_F(DepthMapTest, DepthMap)
{
    CloudXYZPtr cld(new CloudXYZ(3, 2));

    float z_value = 0.f;
    for(pcl::PointCloud<PointXYZ>::iterator itr=cld->begin();
            itr != cld->end(); ++itr) {

        PointXYZ p(-1.f, -2.f, z_value);
        z_value += 1.f;
        *itr = p;
    }

    Signal sig;
    sig.Append(NAME_CLOUD, cld);

    to_->Activate(sig);
    to_->Response(sig);

    Mat1f depth_map = sig.MostRecentMat1f(NAME_DEPTH_MAP);

    EXPECT_EQ(static_cast<int>(cld->height), depth_map.rows);
    EXPECT_EQ(static_cast<int>(cld->width), depth_map.cols);

    for(int i=0; i<depth_map.rows; i++) {

        for(int j=0; j<depth_map.cols; j++) {

            if(cld->at(j, i).z > DepthMap::DEFAULT_DEPTH_MAX) {

                EXPECT_FLOAT_EQ(DepthMap::DEFAULT_DEPTH_MAX, depth_map(i, j));
            }
            else {

                EXPECT_FLOAT_EQ(cld->at(j, i).z, depth_map(i, j));
            }
        }
    }
}

} // annonymous namespace for test cases and fixtures
