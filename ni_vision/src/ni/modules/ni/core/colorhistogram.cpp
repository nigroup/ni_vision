#include "ni/core/colorhistogram.h"

#include <opencv2/core/core.hpp>

#include "elm/core/debug_utils.h"

using namespace std;
using namespace cv;
using namespace elm;
using namespace ni;

void ni::computeColorHist(const Mat &src, const VecI &indices, int nb_bins, Mat1f &dst)
{
    int nb_bins_sq = nb_bins*nb_bins;
    dst = Mat1f::zeros(1, nb_bins_sq*nb_bins);

    // numerator has to be slightly greater than 1, otherwise there is a problem if a channel is equal to 255
    float bin_width = 255.0001f/static_cast<float>(nb_bins);

    uchar* data_ptr = src.data;

    for(size_t i=0; i<indices.size(); i++) {

        int pixel_idx = indices[i]*3;
        float _b = static_cast<float>(data_ptr[pixel_idx]);
        float _g = static_cast<float>(data_ptr[pixel_idx+1]);
        float _r = static_cast<float>(data_ptr[pixel_idx+2]);

        // all gray intensities into a single bin
        if(_b == _g && _g == _r) {

            _b = _g = _r = 255.f/3.f;
        }

        // perform binning
        int bin_b = static_cast<int>(_b/bin_width);
        int bin_g = static_cast<int>(_g/bin_width);
        int bin_r = static_cast<int>(_r/bin_width);

        // changing color order from BGR to RGB
        dst(bin_r*nb_bins_sq + bin_g*nb_bins + bin_b)++;
    }

    if (!indices.empty()) {

        dst /= static_cast<float>(indices.size()); // normalize
    }
}
