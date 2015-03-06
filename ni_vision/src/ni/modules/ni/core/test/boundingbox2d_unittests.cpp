#include "ni/core/boundingbox2d.h"

#include "gtest/gtest.h"

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include "elm/core/cv/mat_utils.h"

using namespace cv;
using namespace pcl;
using namespace elm;
using namespace ni;

class BoundingBox2DTest : public ::testing::Test
{
protected:
    virtual void SetUp()
    {
        to_ = BoundingBox2D();
    }

    // members
    BoundingBox2D to_;  ///< test object
};

TEST_F(BoundingBox2DTest, construct_from_rect)
{
    to_ = BoundingBox2D(Rect2i(3, 6, 9, 12));

    EXPECT_EQ(3, to_.tl().x) << "Unexpected x coordinate for top-left corner.";
    EXPECT_EQ(6, to_.tl().y) << "Unexpected y coordinate for top-left corner";

    EXPECT_EQ(3+9, to_.br().x) << "Unexpected x coordinate for bottom-right corner";
    EXPECT_EQ(6+12, to_.br().y) << "Unexpected y coordinate for bottom-right corner";

    EXPECT_EQ(12, to_.height) << "Unexpected height";
    EXPECT_EQ(9, to_.width) << "Unexpected width";

    EXPECT_EQ(12*9, to_.area()) << "Unexpected area";
}

TEST_F(BoundingBox2DTest, diagonal)
{
    to_ = BoundingBox2D(Rect2i(3, 6, 9, 12));

    EXPECT_FLOAT_EQ(static_cast<float>(sqrt(12*12+81)), to_.diagonal()) << "Unexpected diagonal";

    // after translation
    to_ = BoundingBox2D(Rect2i(7, 100, 9, 12));

    EXPECT_FLOAT_EQ(static_cast<float>(sqrt(12*12+81)), to_.diagonal()) << "Unexpected diagonal after translation";
}

TEST_F(BoundingBox2DTest, diagonal_zero_area)
{
    to_ = BoundingBox2D(Rect2i(3, 6, 0, 0));

    EXPECT_FLOAT_EQ(0, to_.diagonal()) << "Unexpected diagonal for Rect with zero area";
}

TEST_F(BoundingBox2DTest, centralPoint)
{
    to_ = BoundingBox2D(Rect2i(0, 0, 6, 12));

    Point2i cp = elm::Mat2Point2i(to_.centralPoint());
    EXPECT_FLOAT_EQ(3, cp.x);
    EXPECT_FLOAT_EQ(6, cp.y);

    // after translation
    to_ = BoundingBox2D(Rect2i(7, 100, 6, 12));

    cp = elm::Mat2Point2i(to_.centralPoint());
    EXPECT_FLOAT_EQ(3+7, cp.x);
    EXPECT_FLOAT_EQ(6+100, cp.y);
}

TEST_F(BoundingBox2DTest, centralPoint_zero_area)
{
    to_ = BoundingBox2D(Rect2i(3, 6, 0, 0));

    Point2i cp = elm::Mat2Point2i(to_.centralPoint());
    EXPECT_FLOAT_EQ(to_.tl().x, cp.x);
    EXPECT_FLOAT_EQ(to_.tl().y, cp.y);
}
