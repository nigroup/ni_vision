#include "ni/core/graph/VertexOpSegmentSize.h"

#include "gtest/gtest.h"

#include "elm/ts/mat_assertions.h"

using namespace cv;
using namespace ni;

namespace {

class VertexOpSegmentSizeTest : public ::testing::Test
{
protected:
    virtual void SetUp()
    {
        to_ = VertexOpSegmentSize();
    }

    VertexOpSegmentSize to_;    ///< test object
};

TEST_F(VertexOpSegmentSizeTest, MutableOp)
{
    Mat1f img(2, 2);
    for(size_t i=0; i<img.total(); i++) {

        img(i) = static_cast<float>(i);
    }

    for(size_t i=0; i<img.total(); i++) {

        EXPECT_FLOAT_EQ(1.f, to_.mutableOp(img, img == static_cast<float>(i))(0));
    }
}

TEST_F(VertexOpSegmentSizeTest, MutableOp_input_empty)
{
    EXPECT_MAT_DIMS_EQ(to_.mutableOp(Mat1f(), Mat1b()), cv::Size2i(1, 1));
    EXPECT_FALSE(to_.mutableOp(Mat1f(), Mat1b()).empty());
    EXPECT_FLOAT_EQ(0.f, to_.mutableOp(Mat1f(), Mat1b())(0));
}

TEST_F(VertexOpSegmentSizeTest, MutableOp_dims)
{
    for(int r=1; r<11; r++) {

        for(int c=1; c<11; c++) {

            Mat1f img(r, c, 123.f);
            Mat1f result = to_.mutableOp(img, img > 0);
            EXPECT_MAT_DIMS_EQ(result, cv::Size2i(1, 1));
        }
    }
}

TEST_F(VertexOpSegmentSizeTest, MutableOp_mask)
{
    Mat1f img(2, 2);
    for(size_t i=0; i<img.total(); i++) {

        img(i) = static_cast<float>(i);
    }

    Mat1f result;

    for(size_t i=0; i<img.total(); i++) {

        result = to_.mutableOp(img, img != static_cast<float>(i));

        EXPECT_FLOAT_EQ(static_cast<int>(img.total())-1, result(0));
    }
}


} // annonymous namespace for unit tests
