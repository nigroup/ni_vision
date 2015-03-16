#include "ni/core/colorhistogram.h"

#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;
using namespace elm;
using namespace ni;

void ni::computeColorHist(const Mat &src, const VecI &indices, int nb_bins, Mat1f &dst)
{
    dst = Mat1f::zeros(1, 3*nb_bins);

    int nb_bins_sq = nb_bins*nb_bins;

    // numerator has to be slightly greater than 1, otherwise there is a problem if a channel is equal to 255
    float bin_width = 1.0001f/static_cast<float>(nb_bins);

    uchar* data_ptr = src.data;

    for(size_t i=0; i<indices.size(); i++) {

        int pixel = i*3;
        float _b = static_cast<float>(data_ptr[pixel]);
        float _g = static_cast<float>(data_ptr[pixel+1]);
        float _r = static_cast<float>(data_ptr[pixel+2]);

        float sum = _b + _g + _r;
        if(sum != 0.f) {

            _b /= sum;
            _g /= sum;
            _r /= sum;
        }
        else {
            _b = _g = _r = 1.f/3.f; // why so for black pixels?
        }

        // perform binning
        int bin_b = static_cast<int>(_b/bin_width);
        int bin_g = static_cast<int>(_g/bin_width);
        int bin_r = static_cast<int>(_r/bin_width);

        // changing color order from BGR to RGB
        dst(bin_r*nb_bins_sq + bin_g*nb_bins + bin_b)++;
    }

    dst /= static_cast<float>(indices.size()); // normalizing
}
