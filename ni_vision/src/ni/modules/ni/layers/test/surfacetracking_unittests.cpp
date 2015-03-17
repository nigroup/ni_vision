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
    }

    // members
    std::shared_ptr<elm::base_Layer > to_;  ///< test object

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

} // annonymous namespace for testing
