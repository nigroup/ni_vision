#include "ni/layers/surfacetracking.h"

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include "elm/core/debug_utils.h"
#include "elm/core/exception.h"
#include "elm/core/layerconfig.h"
#include "elm/core/inputname.h"
#include "elm/core/signal.h"
#include "elm/ts/layer_assertions.h"

using namespace cv;
using namespace pcl;
using namespace elm;
using namespace ni;

namespace {

ELM_INSTANTIATE_LAYER_TYPED_TEST_CASE_P(SurfaceTracking);

const std::string NAME_IN_BGR       = "bgr";
const std::string NAME_IN_POINTS    = "points";
const std::string NAME_IN_MAP       = "map";
const std::string NAME_OUT_MAP_TRACKED = "map_tracked";

class SurfaceTrackingTest : public ::testing::Test
{
protected:
    virtual void SetUp()
    {
        const int ROWS = 3;
        const int COLS = 4;

        // map data
        float data[ROWS*COLS] = {1.f, 1.f, 2.f, 3.f,
                                 1.f, 1.f, 2.f, 3.f,
                                 0.f, 1.f, 2.f, 5.f};
        map_ = Mat1f(ROWS, COLS, data).clone();

        // cloud data
        cloud_.reset(new CloudXYZ(COLS, ROWS));

        float x = -1.f;
        float y = -2.f;
        float delta_x = 0.5f;
        float delta_y = 0.5f;

        for(int r=0; r<ROWS; r++, y+=delta_y) {

            for(int c=0; c<COLS; c++, x+=delta_x) {

                float z = map_(r, c)/10.f; // use map value to generate depth values
                cloud_->at(c, r) = PointXYZ(x, y, z);
            }
        }

        // color image data
        bgr_ = Mat(ROWS, COLS, CV_8UC3);
        for(int r=0; r<ROWS; r++, y+=delta_y) {

            for(int c=0; c<COLS; c++, x+=delta_x) {

                // user map values to generate colors
                Vec3b bgr_pixel;
                bgr_pixel[0] = static_cast<uchar>(map_(r, c)*10.f);
                bgr_pixel[1] = static_cast<uchar>(map_(r, c)*10.f) + static_cast<uchar>(r);
                bgr_pixel[2] = static_cast<uchar>(map_(r, c)*10.f) + static_cast<uchar>(c);

                bgr_.at<Vec3b>(r, c) = bgr_pixel;
            }
        }

        // layer initialization
        PTree params;
        params.put(SurfaceTracking::PARAM_HIST_BINS, 4);
        params.put(SurfaceTracking::PARAM_MAX_COLOR, 4.f);
        params.put(SurfaceTracking::PARAM_MAX_POS,  4.f);
        params.put(SurfaceTracking::PARAM_MAX_SIZE, 4.f);
        params.put(SurfaceTracking::PARAM_WEIGHT_COLOR, 4.f);
        params.put(SurfaceTracking::PARAM_WEIGHT_POS,   4.f);
        params.put(SurfaceTracking::PARAM_WEIGHT_SIZE,  4.f);

        config_.Params(params);

        LayerIONames io;
        io.Input(SurfaceTracking::KEY_INPUT_BGR_IMAGE,  NAME_IN_BGR);
        io.Input(SurfaceTracking::KEY_INPUT_CLOUD,      NAME_IN_POINTS);
        io.Input(SurfaceTracking::KEY_INPUT_MAP,        NAME_IN_MAP);
        io.Output(SurfaceTracking::KEY_OUTPUT_RESPONSE, NAME_OUT_MAP_TRACKED);

        to_.reset(new SurfaceTracking(config_));
        to_->IONames(io);

        // Signal
        sig_.Clear();

        Mat3f tmp;
        bgr_.convertTo(tmp, CV_32FC3, 1.f/255.f);

        sig_.Append(NAME_IN_BGR, static_cast<Mat1f>(tmp));
        sig_.Append(NAME_IN_MAP, map_);
        sig_.Append(NAME_IN_POINTS, cloud_);
    }

    // members
    std::shared_ptr<elm::base_Layer > to_;  ///< test object

    LayerConfig config_;
    Signal sig_;

    CloudXYZPtr cloud_;
    Mat1f map_;
    Mat bgr_;
};

TEST_F(SurfaceTrackingTest, ColorImageAsMat1f)
{
    Mat3f tmp;
    bgr_.convertTo(tmp, CV_32FC3);

    Mat1f color_floats = static_cast<Mat1f>(tmp);

    EXPECT_MAT_DIMS_EQ(color_floats, Size2i(3*bgr_.cols, bgr_.rows));

    for(int r=0; r<bgr_.rows; r++) {

        for(int c=0; c<bgr_.cols; c++) {

            // user map values to generate colors
            Vec3b bgr_pixel = bgr_.at<Vec3b>(r, c);

            float blue = static_cast<float>(bgr_pixel[0]);
            float green = static_cast<float>(bgr_pixel[1]);
            float red = static_cast<float>(bgr_pixel[2]);

            int col_offset = c*3;

            EXPECT_FLOAT_EQ(blue, color_floats(r, col_offset))      << "at (" << r << "," << c << ")";
            EXPECT_FLOAT_EQ(green, color_floats(r, col_offset+1))   << "at (" << r << "," << c << ")";
            EXPECT_FLOAT_EQ(red, color_floats(r, col_offset+2))     << "at (" << r << "," << c << ")";
        }
    }
}

TEST_F(SurfaceTrackingTest, Response_exists)
{
    ASSERT_FALSE(sig_.Exists(NAME_OUT_MAP_TRACKED));
    to_->Activate(sig_);
    to_->Response(sig_);
    EXPECT_TRUE(sig_.Exists(NAME_OUT_MAP_TRACKED));
}

TEST_F(SurfaceTrackingTest, Response_dims)
{
    to_->Activate(sig_);
    to_->Response(sig_);
    EXPECT_MAT_DIMS_EQ(sig_.MostRecentMat1f(NAME_OUT_MAP_TRACKED), map_);
}

class SurfaceTrackingProtected : public SurfaceTracking
{
public:
    SurfaceTrackingProtected()
        : SurfaceTracking()
    {}

    SurfaceTrackingProtected(const LayerConfig &cfg)
        : SurfaceTracking(cfg)
    {
        Reset(cfg);
    }

    vector<Surface> getObserved() const
    {
        return obsereved_;
    }

    cv::Mat1f distance() const
    {
        return dist_;
    }
};

class SurfaceTrackingProtectedTest : public SurfaceTrackingTest
{
protected:
    virtual void SetUp()
    {
        SurfaceTrackingTest::SetUp();

        LayerIONames io;
        io.Input(SurfaceTracking::KEY_INPUT_BGR_IMAGE,  NAME_IN_BGR);
        io.Input(SurfaceTracking::KEY_INPUT_CLOUD,      NAME_IN_POINTS);
        io.Input(SurfaceTracking::KEY_INPUT_MAP,        NAME_IN_MAP);
        io.Output(SurfaceTracking::KEY_OUTPUT_RESPONSE, NAME_OUT_MAP_TRACKED);

        top_.Reset(config_);
        top_.IONames(io);

        top_.Activate(sig_);
        top_.Response(sig_);
    }

    // members
    SurfaceTrackingProtected top_; ///< test object with access to protected methods
};

TEST_F(SurfaceTrackingProtectedTest, Observed_how_many)
{
    vector<Surface> obs = top_.getObserved();

    EXPECT_SIZE(4, obs) << "Unexpected number of surfaces.";
}

TEST_F(SurfaceTrackingProtectedTest, Observed_ids)
{
    vector<Surface> obs = top_.getObserved();

    // Ids
    VecI ids(obs.size());
    for (size_t i=0; i<obs.size(); i++) {

        ids[i] = obs[i].id();
    }

    EXPECT_EQ("1, 2, 3, 4", elm::to_string(ids)) << "Unexpected surface ids";
}

TEST_F(SurfaceTrackingProtectedTest, Observed_pixelInfo)
{
    vector<Surface> obs = top_.getObserved();

    /*
     * float data[ROWS*COLS] = {1.f, 1.f, 2.f, 3.f,
                                1.f, 1.f, 2.f, 3.f,
                                0.f, 1.f, 2.f, 5.f};
     */
    EXPECT_SIZE(5, obs[0].pixelIndices());
    EXPECT_SIZE(3, obs[1].pixelIndices());
    EXPECT_SIZE(2, obs[2].pixelIndices());
    EXPECT_SIZE(1, obs[3].pixelIndices());

    EXPECT_EQ("0, 1, 4, 5, 9",  elm::to_string(obs[0].pixelIndices()));
    EXPECT_EQ("2, 6, 10",       elm::to_string(obs[1].pixelIndices()));
    EXPECT_EQ("3, 7",           elm::to_string(obs[2].pixelIndices()));
    EXPECT_EQ("11",             elm::to_string(obs[3].pixelIndices()));

    // pixel count
    EXPECT_EQ(5, obs[0].pixelCount());
    EXPECT_EQ(3, obs[1].pixelCount());
    EXPECT_EQ(2, obs[2].pixelCount());
    EXPECT_EQ(1, obs[3].pixelCount());
}

TEST_F(SurfaceTrackingProtectedTest, Observed_histogram_dims)
{
    vector<Surface> obs = top_.getObserved();

    int bins = config_.Params().get<int>(SurfaceTracking::PARAM_HIST_BINS);

    for (size_t i=0; i<obs.size(); i++) {

        EXPECT_MAT_DIMS_EQ(obs[i].colorHistogram(), cv::Size2i(bins*bins*bins, 1));
    }
}

TEST_F(SurfaceTrackingProtectedTest, Distance_dims)
{
    vector<Surface> obs = top_.getObserved();
    top_.Activate(sig_);

    const int N = static_cast<int>(obs.size());

    ASSERT_GT(N, 0);

    EXPECT_MAT_EQ(Mat1f(N, N, SurfaceTracking::DISTANCE_HUGE), top_.distance());
}

} // annonymous namespace for testing
