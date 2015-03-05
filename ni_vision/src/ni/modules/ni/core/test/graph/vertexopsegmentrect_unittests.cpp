#include "ni/core/graph/vertexopsegmentrect.h"

#include "gtest/gtest.h"

#include "elm/ts/mat_assertions.h"

using namespace cv;
using namespace ni;

namespace {

class VertexOpSegmentRectTest : public ::testing::Test
{
protected:
    virtual void SetUp()
    {
        to_ = VertexOpSegmentRect();
    }

    VertexOpSegmentRect to_;    ///< test object
};

TEST_F(VertexOpSegmentRectTest, MutableOp_single_elements)
{
    Mat1f img(2, 2);
    for(size_t i=0; i<img.total(); i++) {

        img(i) = static_cast<float>(i);
    }

    float v = 0.f;
    for(int r=0; r<img.rows; r++) {

        for(int c=0; c<img.cols; c++, v++) {

            Mat1i mat_rect = to_.mutableOp(img, img == v);
            EXPECT_EQ(c, mat_rect(0));
            EXPECT_EQ(r, mat_rect(1));
            EXPECT_EQ(c, mat_rect(2));
            EXPECT_EQ(r, mat_rect(3));
        }
    }
}

TEST_F(VertexOpSegmentRectTest, CalcRect_single_elements)
{
    Mat1f img(2, 2);
    for(size_t i=0; i<img.total(); i++) {

        img(i) = static_cast<float>(i);
    }

    float v = 0.f;
    for(int r=0; r<img.rows; r++) {

        for(int c=0; c<img.cols; c++, v++) {

            Mat1i mat_rect = VertexOpSegmentRect::calcRect(img, img == v);
            EXPECT_EQ(c, mat_rect(0));
            EXPECT_EQ(r, mat_rect(1));
            EXPECT_EQ(c, mat_rect(2));
            EXPECT_EQ(r, mat_rect(3));
        }
    }
}

TEST_F(VertexOpSegmentRectTest, MutableOp_input_empty)
{
    EXPECT_TRUE(to_.mutableOp(Mat1f(), Mat1b()).empty());
}

TEST_F(VertexOpSegmentRectTest, CalcRect_input_empty)
{
    EXPECT_TRUE(VertexOpSegmentRect::calcRect(Mat1f(), Mat1b()).empty());
}

TEST_F(VertexOpSegmentRectTest, MutableOp_dims)
{
    Mat1f in(11, 11, 123.f);
    for(int r=1; r<in.rows; r++) {

        for(int c=1; c<in.cols; c++) {

            Mat1f img = in.colRange(0, c).rowRange(0, r);
            EXPECT_MAT_DIMS_EQ(to_.mutableOp(img, img > 0.f), cv::Size2i(4, 1));
        }
    }
}

TEST_F(VertexOpSegmentRectTest, CalcRect_dims)
{
    Mat1f in(11, 11, 123.f);
    for(int r=1; r<in.rows; r++) {

        for(int c=1; c<in.cols; c++) {

            Mat1f img = in.colRange(0, c).rowRange(0, r);
            EXPECT_MAT_DIMS_EQ(VertexOpSegmentRect::calcRect(img, img > 0.f), cv::Size2i(4, 1));
        }
    }
}

TEST_F(VertexOpSegmentRectTest, MutableOp)
{
    const int ROWS=3;
    const int COLS=3;
    float data[ROWS*COLS] = {1.f, 1.0f, 2.2f,
                             3.f, 1.0f, 2.2f,
                             1.f, 11.f, 11.f};
    Mat1f img = Mat1f(ROWS, COLS, data).clone();

    Mat1i mat_rect, mat_rect_calc_rect;

    mat_rect = VertexOpSegmentRect::calcRect(img, img == 1.0f);
    EXPECT_EQ(0, mat_rect(0));
    EXPECT_EQ(0, mat_rect(1));
    EXPECT_EQ(1, mat_rect(2));
    EXPECT_EQ(2, mat_rect(3));
    mat_rect_calc_rect = VertexOpSegmentRect::calcRect(img, img == 1.0f);
    EXPECT_MAT_EQ(mat_rect, mat_rect_calc_rect);

    mat_rect = VertexOpSegmentRect::calcRect(img, img == 2.2f);
    EXPECT_EQ(2, mat_rect(0));
    EXPECT_EQ(0, mat_rect(1));
    EXPECT_EQ(2, mat_rect(2));
    EXPECT_EQ(1, mat_rect(3));
    mat_rect_calc_rect = VertexOpSegmentRect::calcRect(img, img == 2.2f);
    EXPECT_MAT_EQ(mat_rect, mat_rect_calc_rect);

    mat_rect = VertexOpSegmentRect::calcRect(img, img == 3.f);
    EXPECT_EQ(0, mat_rect(0));
    EXPECT_EQ(1, mat_rect(1));
    EXPECT_EQ(0, mat_rect(2));
    EXPECT_EQ(1, mat_rect(3));
    mat_rect_calc_rect = VertexOpSegmentRect::calcRect(img, img == 3.f);
    EXPECT_MAT_EQ(mat_rect, mat_rect_calc_rect);

    mat_rect = VertexOpSegmentRect::calcRect(img, img == 11.f);
    EXPECT_EQ(1, mat_rect(0));
    EXPECT_EQ(2, mat_rect(1));
    EXPECT_EQ(2, mat_rect(2));
    EXPECT_EQ(2, mat_rect(3));
    mat_rect_calc_rect = VertexOpSegmentRect::calcRect(img, img == 11.f);
    EXPECT_MAT_EQ(mat_rect, mat_rect_calc_rect);
}

} // annonymous namespace for unit tests
