#include "ni/core/color_utils.h"

#include <vector>

#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;
using namespace ni;

void ni::normalizeColors(const Mat &src, Mat &dst)
{
    vector<Mat> ch;
    split(src, ch);

    const int NB_CHANNELS = static_cast<int>(ch.size());

    if(NB_CHANNELS > 1) {

        Mat1f sum = ch[0].clone();
        for (int i=1; i<NB_CHANNELS; i++) {

            sum += ch[i];
        }

        vector<Mat1f> ch_normalized(NB_CHANNELS);

        ch_normalized[NB_CHANNELS-1].setTo(1.f);

        for (int i=0; i<NB_CHANNELS-1; i++) {

            Mat1f tmp = static_cast<Mat1f>(ch[i]);
            cv::divide(tmp, sum, ch_normalized[i]); // returns zero wherever denominator is zero

            ch_normalized[src.channels()-1] -= ch_normalized[i];
        }

        cv::merge(ch_normalized, dst);
    }
    else if(NB_CHANNELS == 1) {

        dst = Mat1f::ones(src.size());
    }
}
