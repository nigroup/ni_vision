#include "ni/core/colorhistogram.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "elm/core/debug_utils.h"
#include "elm/core/cv/mat_utils.h"
#include "elm/core/cv/mat_utils_inl.h"
#include "elm/core/cv/mat_vector_utils_inl.h"
#include "elm/ts/mat_assertions.h"

using namespace std;
using namespace cv;
using namespace elm;
using namespace ni;

namespace {

class ColorHistogramTest : public ::testing::Test
{
protected:
    virtual void SetUp()
    {

    }
};

TEST_F(ColorHistogramTest, Black_8u)
{
    Mat in = Mat::zeros(2, 3, CV_8UC3);
    Mat1f hist;

    VecI indices = Mat_ToVec_<int>(ARange_<int>(0, static_cast<int>(in.total()), 1));

    computeColorHist(in, indices, 8, hist);

    Mat1f expected_hist = Mat1f::zeros(hist.size());
    expected_hist(8/3*64+8/3*8+8/3) = 1;
    EXPECT_MAT_EQ(hist, expected_hist);
}

TEST_F(ColorHistogramTest, White_8u)
{
    Mat in = Mat(2, 3, CV_8UC3);
    in = Scalar(255, 255, 255);
    Mat1f hist;

    VecI indices = Mat_ToVec_<int>(ARange_<int>(0, static_cast<int>(in.total()), 1));

    computeColorHist(in, indices, 8, hist);

    Mat1f expected_hist = Mat1f::zeros(hist.size());
    expected_hist(8/3*64+8/3*8+8/3) = 1;
    EXPECT_MAT_EQ(hist, expected_hist);
}

TEST_F(ColorHistogramTest, Single_color_8u)
{
    Mat in = Mat(2, 3, CV_8UC3);
    {
        // blue
        in = Scalar(255, 0, 0);
        Mat1f hist;

        VecI indices = Mat_ToVec_<int>(ARange_<int>(0, static_cast<int>(in.total()), 1));

        computeColorHist(in, indices, 8, hist);

        Mat1f expected_hist = Mat1f::zeros(hist.size());
        expected_hist(0*64+0*8+8-1) = 1;
        EXPECT_MAT_EQ(hist, expected_hist);
    }
    {
        // green
        in = Scalar(0, 255, 0);
        Mat1f hist;

        VecI indices = Mat_ToVec_<int>(ARange_<int>(0, static_cast<int>(in.total()), 1));

        computeColorHist(in, indices, 8, hist);

        Mat1f expected_hist = Mat1f::zeros(hist.size());
        expected_hist(0*64+(8-1)*8+0) = 1;
        EXPECT_MAT_EQ(hist, expected_hist);
    }
    {
        // red
        in = Scalar(0, 0, 255);
        Mat1f hist;

        VecI indices = Mat_ToVec_<int>(ARange_<int>(0, static_cast<int>(in.total()), 1));

        computeColorHist(in, indices, 8, hist);

        Mat1f expected_hist = Mat1f::zeros(hist.size());
        expected_hist((8-1)*64+0*8+0) = 1;
        EXPECT_MAT_EQ(hist, expected_hist);
    }
}

TEST_F(ColorHistogramTest, Color_8u)
{
    Mat in = Mat(2, 3, CV_8UC3);
    in = Scalar(0, 0, 0);

    {
        Vec3b v;
        v.all(0);
        v[1] = v[2] = 255; // yellow
        in.at<Vec3b>(0) = v;
    }
    {
        Vec3b v;
        v.all(0);
        v[0] = v[2] = 255; // purple
        in.at<Vec3b>(1) = v;
    }
    {
        Vec3b v;
        v.all(0);
        v[0] = v[1] = v[2] = 128; // gray
        in.at<Vec3b>(2) = v;
    }

    Mat1f hist;

    VecI indices = Mat_ToVec_<int>(ARange_<int>(0, static_cast<int>(in.total()), 1));

    computeColorHist(in, indices, 8, hist);

    Mat1f expected_hist = Mat1f::zeros(hist.size());

    float denom = static_cast<float>(in.total());

    expected_hist((8-1)*64+(8-1)*8+0) = 1.f/denom;            // 1x yellow
    expected_hist((8-1)*64+0*8+(8-1)) = 1.f/denom;            // 1x purple
    expected_hist(8/3*64+8/3*8+8/3) = (in.total()-2)/denom; // several black and gray
    EXPECT_MAT_EQ(hist, expected_hist);
}

TEST_F(ColorHistogramTest, Color_8u_masked)
{
    Mat in = Mat(2, 3, CV_8UC3);
    in = Scalar(0, 0, 0);

    {
        Vec3b v;
        v.all(0);
        v[1] = v[2] = 255; // yellow
        in.at<Vec3b>(0) = v;
    }
    {
        Vec3b v;
        v.all(0);
        v[0] = v[2] = 255; // purple
        in.at<Vec3b>(1) = v;
    }
    {
        Vec3b v;
        v.all(0);
        v[0] = v[1] = v[2] = 128; // gray
        in.at<Vec3b>(2) = v;
    }

    Mat1f hist;

    VecI indices = Mat_ToVec_<int>(ARange_<int>(1, static_cast<int>(in.total()), 1));

    computeColorHist(in, indices, 8, hist);

    Mat1f expected_hist = Mat1f::zeros(hist.size());

    float denom = static_cast<float>(indices.size());

    //ELM_COUT_VAR(hist);

    expected_hist((8-1)*64+0*8+(8-1)) = 1.f/denom;            // 1x purple
    expected_hist(8/3*64+8/3*8+8/3) = (indices.size()-1)/denom; // several black and gray
    EXPECT_MAT_EQ(hist, expected_hist);
    EXPECT_FLOAT_EQ(1.f, cv::sum(hist)[0]);
}

TEST_F(ColorHistogramTest, HistogramSum)
{
    Mat in = Mat(2, 3, CV_8UC3);
    in = Scalar(0, 0, 0);

    {
        Vec3b v;
        v.all(0);
        v[1] = v[2] = 255; // yellow
        in.at<Vec3b>(0) = v;
    }
    {
        Vec3b v;
        v.all(0);
        v[0] = v[2] = 255; // purple
        in.at<Vec3b>(1) = v;
    }
    {
        Vec3b v;
        v.all(0);
        v[0] = v[1] = v[2] = 128; // gray
        in.at<Vec3b>(2) = v;
    }

    Mat1f hist;

    VecI indices = Mat_ToVec_<int>(ARange_<int>(0, static_cast<int>(in.total()), 1));

    computeColorHist(in, indices, 8, hist);

    EXPECT_FLOAT_EQ(1.f, cv::sum(hist)[0]);
}

TEST_F(ColorHistogramTest, No_indices)
{
    Mat in = Mat(2, 3, CV_8UC3);
    in = Scalar(0, 0, 0);

    {
        Vec3b v;
        v.all(0);
        v[1] = v[2] = 255; // yellow
        in.at<Vec3b>(0) = v;
    }
    {
        Vec3b v;
        v.all(0);
        v[0] = v[2] = 255; // purple
        in.at<Vec3b>(1) = v;
    }
    {
        Vec3b v;
        v.all(0);
        v[0] = v[1] = v[2] = 128; // gray
        in.at<Vec3b>(2) = v;
    }

    Mat1f hist;
    computeColorHist(in, VecI(), 8, hist);

    EXPECT_MAT_DIMS_EQ(hist, cv::Size2i(8*8*8, 1));
    EXPECT_FLOAT_EQ(0.f, cv::sum(hist)[0]);
}

TEST_F(ColorHistogramTest, Dims)
{
    for(int bins=4; bins<=16; bins++) {

        for(int r=1; r<11; r++) {

            for(int c=1; c<11; c++) {

                Mat in = Mat(2, 3, CV_8UC3);
                in = Scalar(0, 0, 0);

                VecI indices = Mat_ToVec_<int>(ARange_<int>(0, static_cast<int>(in.total()), 1));

                Mat1f hist;
                computeColorHist(in, indices, bins, hist);

                EXPECT_MAT_DIMS_EQ(hist, cv::Size2i(bins*bins*bins, 1));
            }
        }
    }
}

} // annonymous namespace for tests
