#include "ni/core/surface.h"

#include "elm/ts/mat_assertions.h"

using namespace std;
using namespace ni;

namespace {

class SurfaceTest : public ::testing::Test
{
protected:
    virtual void SetUp()
    {
        to_ = Surface();
    }

    Surface to_;    ///< test object
};

TEST_F(SurfaceTest, pixelCount)
{
    for(int i=1; i<10; i++) {

        VecI indices;
        for(int j=i; j>0; j--) {

            indices.push_back(j);
        }

        Surface to;
        to.pixelIndices(indices);
        EXPECT_EQ(i, to.pixelCount()) << "Pixel count mismatch.";
    }
}

TEST_F(SurfaceTest, pixelCount_overwrite)
{
    for(int i=1; i<10; i++) {

        VecI indices;
        for(int j=i; j>0; j--) {

            indices.push_back(j);
        }

        Surface to;

        to.overwritePixelCount((i+1)*10);
        EXPECT_EQ((i+1)*10, to.pixelCount()) << "Pixel count mismatch.";

        to.pixelIndices(indices);
        EXPECT_EQ(i, to.pixelCount()) << "Pixel count mismatch.";

        to.overwritePixelCount((i+1)*10);
        EXPECT_EQ((i+1)*10, to.pixelCount()) << "Pixel count mismatch.";
    }
}

TEST_F(SurfaceTest, lastSeenCount)
{
    EXPECT_EQ(0, to_.lastSeenCount()) << "expecting initial count of zero.";

    const int N=5;
    for(int i=0; i<N; i++) {

        EXPECT_EQ(i, to_.lastSeenCount()) << "count not increasing";
        to_.lastSeenCount(false);
    }

    for(int i=0; i<N; i++) {

        to_.lastSeenCount(true);
        EXPECT_EQ(0, to_.lastSeenCount()) << "count did not reset.";
    }
}

TEST_F(SurfaceTest, Id)
{
    for(int i=-11; i<=11; i++) {

        to_.id(i);
        EXPECT_EQ(i, to_.id());
    }
}

TEST_F(SurfaceTest, ColorHistogram)
{
    for(int i=2; i<=11; i++) {

        to_.colorHistogram(cv::Mat1f(1, i, 1.f));
        EXPECT_MAT_DIMS_EQ(to_.colorHistogram(), cv::Mat1f(1, i, 1.f));
    }
}

} // annonymous namespace for unit tests
