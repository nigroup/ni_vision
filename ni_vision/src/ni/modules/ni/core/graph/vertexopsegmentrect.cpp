#include "ni/core/graph/vertexopsegmentrect.h"

#include <opencv2/core/core.hpp>

#include "elm/core/cv/mat_utils.h"

using namespace cv;
using namespace ni;

VertexOpSegmentRect::VertexOpSegmentRect()
    : base_GraphVertexOp()
{
}

Mat1f VertexOpSegmentRect::mutableOp(const Mat1i &img, const Mat &mask)
{
    return VertexOpSegmentRect::calcRect(img, mask);
}

Mat1f VertexOpSegmentRect::calcRect(const Mat1i &img, const Mat &mask)
{
    if(mask.empty()) {

        return Mat1f(); // do nothing
    }

    Mat non_zero_coords;
    cv::findNonZero(mask, non_zero_coords);
    Mat1i nz_coords = non_zero_coords; // split 2-channel mat into columns of a single-channel Mat of integers

    Point2i tl, br;
    tl.y = nz_coords(1);    // y coordinate of first found element
    br.y = nz_coords(static_cast<int>(nz_coords.total())-1); // y coordinate of last found element

    // tl and br's x coordinates are the max and min of all x coordinates found
    double min_val, max_val;
    minMaxIdx(nz_coords.col(0), &min_val, &max_val);

    tl.x = static_cast<int>(min_val);
    br.x = static_cast<int>(max_val);

    return elm::Rect2iToMat(Rect2i(tl, br));
}
