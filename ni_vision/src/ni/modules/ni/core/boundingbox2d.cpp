#include "ni/core/boundingbox2d.h"

#include "elm/core/cv/mat_utils.h"

using namespace cv;
using namespace elm;
using namespace ni;

BoundingBox2D::BoundingBox2D()
    : base_BoundingBox(),
      Rect2i()
{
}

BoundingBox2D::BoundingBox2D(const Rect2i &rect)
    : base_BoundingBox(),
      Rect2i(rect)
{
}

float BoundingBox2D::diagonal() const
{
    double diag = norm(Point2Mat(tl())-Point2Mat(br()));

    return static_cast<float>(diag);
}

Mat1f BoundingBox2D::centralPoint() const
{
    return (Point2Mat(tl())+Point2Mat(br())) / 2.f;
}
