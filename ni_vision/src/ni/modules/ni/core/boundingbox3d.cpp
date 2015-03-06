#include "ni/core/boundingbox3d.h"

#include <opencv2/core/core_c.h>

#include "elm/core/exception.h"
#include "elm/core/cv/mat_utils.h"
#include "elm/core/pcl/cloud_.h"
#include "elm/core/pcl/point_traits.h"

using namespace cv;
using namespace pcl;
using namespace elm;
using namespace ni;

BoundingBox3D::BoundingBox3D()
    : base_BoundingBox()
{
}

BoundingBox3D::BoundingBox3D(CloudXYZPtr &cld)
{
    Mat1f point_coords = PointCloud2Mat_<PointXYZ>(cld);

    const int NB_FLOATS=PCLPointTraits_<PointXYZ>::NbFloats();

    int new_rows = point_coords.total()/NB_FLOATS;
    point_coords = point_coords.reshape(1, new_rows);

    if(point_coords.rows > 1) {

        cog_ = Mat1f(1, NB_FLOATS);
        reduce(point_coords, cog_, 0, CV_REDUCE_AVG);

        Mat1f _min, _max;
        reduce(point_coords, _min, 0, CV_REDUCE_MIN);
        reduce(point_coords, _max, 0, CV_REDUCE_MAX);

        box_xy_ = Mat1f(1, 4);
        box_xy_(0) = _min(0); // tl.x
        box_xy_(1) = _max(1); // tl.y
        box_xy_(2) = _max(0); // br.x
        box_xy_(3) = _min(1); // br.y

        box_yz_ = Mat1f(1, 4);
        box_yz_(0) = _min(2); // tl.x // closest top depth
        box_yz_(1) = _max(1); // tl.y
        box_yz_(2) = _max(2); // br.x
        box_yz_(3) = _min(1); // br.y
    }
    else if(point_coords.rows == 1) {

        cog_ = point_coords.row(0);
        box_xy_ = cog_.clone();
        box_yz_ = cog_.clone();
    }
    else {
        cog_ = Mat1f(0, NB_FLOATS);
        box_xy_ = Mat1f(0, 4);
        box_yz_ = Mat1f(0, 4);
    }
}

float BoundingBox3D::diagonal() const
{
    float dx = box_xy_(2) - box_xy_(0);
    float dy = box_xy_(1) - box_xy_(3);
    float dz = box_yz_(2) - box_yz_(0);

    return static_cast<float>(sqrt(dx*dx + dy*dy + dz*dz));
}

Mat1f BoundingBox3D::centralPoint() const
{
    return cog_;
}

float BoundingBox3D::volume() const
{
    float dx = box_xy_(2) - box_xy_(0);
    float dy = box_xy_(1) - box_xy_(3);
    float dz = box_yz_(2) - box_yz_(0);

    return dx*dy*dz;
}
