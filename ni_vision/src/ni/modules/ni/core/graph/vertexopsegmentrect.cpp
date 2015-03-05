#include "ni/core/graph/vertexopsegmentrect.h"

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

}
