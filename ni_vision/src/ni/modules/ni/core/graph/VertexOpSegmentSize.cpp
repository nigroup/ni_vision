#include "ni/core/graph/VertexOpSegmentSize.h"

using namespace cv;
using namespace elm;
using namespace ni;

VertexOpSegmentSize::VertexOpSegmentSize()
    : base_GraphVertexOp(),
      size_(0)
{
}

Mat1f VertexOpSegmentSize::mutableOp(const Mat1f &img, const Mat1b &mask)
{
    return VertexOpSegmentSize::calcSize(img, mask);
}

Mat1f VertexOpSegmentSize::calcSize(const Mat1f &img, const Mat1b &mask)
{
    return cv::Mat1f(1, 1, static_cast<float>(countNonZero(mask)));
}


