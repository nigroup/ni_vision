#include "ni/core/colorhistogram.h"

#include <opencv2/core/core.hpp>

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
    Mat in = Mat::zeros(3, 4, CV_8UC3);
    Mat1f hist;

    VecI indices = Mat_ToVec_<int>(ARange_<int>(0, static_cast<int>(in.total()), 1));

    computeColorHist(in, indices, 8, hist);

    Mat1f expected_hist = Mat1f::zeros(hist.size());
    expected_hist(0) = 1;
    EXPECT_MAT_EQ(hist, expected_hist);
}

} // annonymous namespace for tests
