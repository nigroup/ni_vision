#include "ni/core/graph/vertexopsegmentrect.h"

#include <opencv2/core/core.hpp>

#include "elm/core/cv/mat_utils.h"
#include "elm/core/debug_utils.h"

using namespace cv;
using namespace ni;

VertexOpSegmentRect::VertexOpSegmentRect()
    : base_GraphVertexOp()
{
}

Mat1f VertexOpSegmentRect::mutableOp(const Mat1f &img, const Mat1b &mask)
{
    return VertexOpSegmentRect::calcRect(img, mask);
}

Mat1f VertexOpSegmentRect::calcRect(const Mat1f &img, const Mat1b &mask)
{
    Mat nonZeroCoordinates;
    cv::findNonZero(mask, nonZeroCoordinates);
//    for (size_t i = 0; i < nonZeroCoordinates.total(); i++) {
//        ELM_COUT_VAR("Zero#" << i << ": " << nonZeroCoordinates.at<Point>(i).x << ", " << nonZeroCoordinates.at<Point>(i).y);
//    }

    Mat1i nz_coords = nonZeroCoordinates;

    Point2i tl, br;
    tl.y = nz_coords(1);
    br.y = nz_coords(static_cast<int>(nz_coords.total())-1);

    Mat1i x = nz_coords.col(0);

    double min_val, max_val;
    minMaxIdx(x, &min_val, &max_val);

    tl.x = static_cast<int>(min_val);
    br.x = static_cast<int>(max_val);


    ELM_COUT_VAR(nonZeroCoordinates);

    return elm::Rect2iToMat(Rect2i(tl, br));
}
