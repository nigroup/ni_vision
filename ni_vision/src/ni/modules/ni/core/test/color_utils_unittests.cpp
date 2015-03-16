#include "ni/core/color_utils.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "elm/ts/mat_assertions.h"

using namespace cv;
using namespace elm;
using namespace ni;

namespace {

class ColorNormalizationTest : public ::testing::Test
{
protected:
    virtual void SetUp()
    {

    }
};

TEST_F(ColorNormalizationTest, ColorNormalization) {

    Mat img = cv::imread("/media/win/Users/woodstock/dev/data/lena.png");

//    std::vector<cv::Mat> v;

//    Mat1f sum = v[0]+v[1]+v[2];

//    std::vector<cv::Mat> v2;

//    cv::Mat1f tmp;
//    tmp = v[0]/sum;
//    v2.push_back(tmp);
//    tmp = v[1]/sum;
//    v2.push_back(tmp);
//    tmp = v[2]/sum;
//    v2.push_back(tmp);
//    Mat img2
//    cv::merge(v2, img2);
//     = img/sum;

//    cv::imshow("i",img);
//    cv::imshow("i2",elm::ConvertTo8U(img2));
//    cv::waitKey();
}

} // annonymous namespace for tests

