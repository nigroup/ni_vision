#include "ni/core/color_utils.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "elm/core/debug_utils.h"
#include "elm/core/cv/mat_type_utils.h"
#include "elm/ts/mat_assertions.h"

using namespace cv;
using namespace ni;

namespace {

class ColorNormalizationTest : public ::testing::Test
{
protected:
    virtual void SetUp()
    {

    }
};

TEST_F(ColorNormalizationTest, Empty) {

    Mat src, dst;
    normalizeColors(src, dst);
    EXPECT_TRUE(dst.empty());
}

TEST_F(ColorNormalizationTest, SingleChannel) {

    Mat1b src(2, 3);
    for(size_t i=0; i<src.total(); i++) {

        src(i) = static_cast<uchar>(i);
    }

    ASSERT_GT(countNonZero(src), 0);

    Mat dst;
    normalizeColors(src, dst);

    EXPECT_MAT_EQ(dst, Mat1f::ones(2, 3));
}

TEST_F(ColorNormalizationTest, Dims) {

    for(int r=1; r<11; r++) {

        for(int c=1; c<11; c++) {

            Mat1b src = Mat1b::ones(r, c);

            Mat dst;
            normalizeColors(src, dst);

            EXPECT_MAT_DIMS_EQ(dst, Size2i(c, r));
        }
    }
}

TEST_F(ColorNormalizationTest, Single_color) {

    Mat src = Mat(2, 3, CV_8UC3);
    {
        // blue
        src = Scalar(255, 0, 0);
        Mat dst;
        normalizeColors(src, dst);

        std::vector<Mat1f> ch;
        cv::split(dst, ch);

        EXPECT_MAT_EQ(ch[0], Mat1f(src.size(), 1.f));
        EXPECT_MAT_EQ(ch[1], Mat1f(src.size(), 0.f));
        EXPECT_MAT_EQ(ch[2], Mat1f(src.size(), 0.f));
    }
    {
        // green
        src = Scalar(0, 255, 0);

        Mat dst;
        normalizeColors(src, dst);

        std::vector<Mat1f> ch;
        cv::split(dst, ch);

        EXPECT_MAT_EQ(ch[0], Mat1f(src.size(), 0.f));
        EXPECT_MAT_EQ(ch[1], Mat1f(src.size(), 1.f));
        EXPECT_MAT_EQ(ch[2], Mat1f(src.size(), 0.f));
    }
    {
        // red
        src = Scalar(0, 0, 255);

        Mat dst;
        normalizeColors(src, dst);

        std::vector<Mat1f> ch;
        cv::split(dst, ch);

        EXPECT_MAT_EQ(ch[0], Mat1f(src.size(), 0.f));
        EXPECT_MAT_EQ(ch[1], Mat1f(src.size(), 0.f));
        EXPECT_MAT_EQ(ch[2], Mat1f(src.size(), 1.f));
    }
}

TEST_F(ColorNormalizationTest, normalizeColors) {

    Mat src = Mat(2, 3, CV_8UC3);
    src = Scalar(0, 0, 0);
    {
        Vec3b v;
        v.all(0);
        v[1] = v[2] = 255; // yellow
        src.at<Vec3b>(0) = v;
    }
    {
        Vec3b v;
        v.all(0);
        v[0] = v[2] = 255; // purple
        src.at<Vec3b>(1) = v;
    }
    {
        Vec3b v;
        v.all(0);
        v[0] = v[1] = v[2] = 128; // gray
        src.at<Vec3b>(2) = v;
    }

    Mat dst;
    normalizeColors(src, dst);

    std::vector<Mat1f> ch;
    cv::split(dst, ch);

    cv::Mat1f expected_blue(src.size(), 0.f);
    expected_blue(1) = 0.5f;
    expected_blue(2) = 1.f/3.f;
    cv::Mat1f expected_green(src.size(), 0.f);
    expected_green(0) = 0.5f;
    expected_green(2) = 1.f/3.f;
    cv::Mat1f expected_red(src.size(), 0.f);
    expected_red(0) = 0.5f;
    expected_red(1) = 0.5f;
    expected_red(2) = 1.f/3.f;

    EXPECT_MAT_EQ(ch[0], expected_blue);
    EXPECT_MAT_EQ(ch[1], expected_green);
    EXPECT_MAT_EQ(ch[2], expected_red);
}

} // annonymous namespace for tests

