#include "ni/core/surface.h"

#include "gtest/gtest.h"

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

} // annonymous namespace for unit tests