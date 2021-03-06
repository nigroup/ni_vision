#include "ni/core/boundingbox3d.h"

#include "gtest/gtest.h"

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include "elm/core/exception.h"

using namespace cv;
using namespace pcl;
using namespace elm;
using namespace ni;

class BoundingBox3DTest : public ::testing::Test
{
protected:
    virtual void SetUp()
    {
        to_ = BoundingBox3D();
    }

    // members
    BoundingBox3D to_;  ///< test object
};

TEST_F(BoundingBox3DTest, construct_from_point_cloud)
{
    CloudXYZPtr cld(new CloudXYZ());

    cld->push_back(PointXYZ(0.f, 0.f, 0.f));
    cld->push_back(PointXYZ(1.f, 1.f, 1.f));
    cld->push_back(PointXYZ(2.f, 2.f, 2.f));
    cld->push_back(PointXYZ(3.f, 3.f, 3.f));

    to_ = BoundingBox3D(cld);

    cv::Mat1f cog = to_.centralPoint();

    EXPECT_EQ(1, cog.rows) << "Expecting row matrix.";
    EXPECT_EQ(3, cog.cols) << "COG is 3-dimensional, 4th column is for SSE padding.";

    for(int i=0; i<cog.cols; i++) {

        EXPECT_FLOAT_EQ(6.f/4.f, cog(i)) << "Unexpected value for coordinate i=" << i;
    }
}

TEST_F(BoundingBox3DTest, construct_from_point_cloud_empty)
{
    CloudXYZPtr cld(new CloudXYZ());

    to_ = BoundingBox3D(cld);

    cv::Mat1f cog = to_.centralPoint();

    EXPECT_TRUE(cog.empty()) << "Empty point cloud yields empty COG Mat.";
}

TEST_F(BoundingBox3DTest, diagonal)
{
    CloudXYZPtr cld(new CloudXYZ());
    cld->push_back(PointXYZ(0.f, 0.f, 0.f));
    cld->push_back(PointXYZ(1.f, 0.f, 0.f));
    cld->push_back(PointXYZ(-1.f, 0.f, 0.f));
    cld->push_back(PointXYZ(0.f, -2.f, 0.f));
    cld->push_back(PointXYZ(0.f, 2.f, 0.f));
    cld->push_back(PointXYZ(0.f, 0.f, -3.f));
    cld->push_back(PointXYZ(0.f, 0.f, 3.f));

    to_ = BoundingBox3D(cld);

    EXPECT_FLOAT_EQ(sqrt(4+16+36), to_.diagonal());
}

TEST_F(BoundingBox3DTest, volume)
{
    CloudXYZPtr cld(new CloudXYZ());
    cld->push_back(PointXYZ(0.f, 0.f, 0.f));
    cld->push_back(PointXYZ(1.f, 0.f, 0.f));
    cld->push_back(PointXYZ(-1.f, 0.f, 0.f));
    cld->push_back(PointXYZ(0.f, -2.f, 0.f));
    cld->push_back(PointXYZ(0.f, 2.f, 0.f));
    cld->push_back(PointXYZ(0.f, 0.f, -3.f));
    cld->push_back(PointXYZ(0.f, 0.f, 3.f));

    to_ = BoundingBox3D(cld);

    EXPECT_FLOAT_EQ(2*4*6, to_.volume());
}

