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
    const int NB_FIELDS=PCLPointTraits_<PointXYZ>::FieldCount();

    int new_rows = point_coords.total()/NB_FLOATS;

    if(!cld->empty()) {

        point_coords = point_coords.reshape(1, new_rows).colRange(0, NB_FIELDS);
    }

    if(point_coords.rows > 1) {

        cog_ = Mat1f(1, point_coords.cols);
        reduce(point_coords, cog_, 0, CV_REDUCE_AVG);

        Mat1f _min, _max;
        reduce(point_coords, _min, 0, CV_REDUCE_MIN);
        reduce(point_coords, _max, 0, CV_REDUCE_MAX);

        box_xy_ = Mat1f(1, 4);
        box_xy_(0) = _min(0); // tl.x
        box_xy_(1) = _max(1); // tl.y
        box_xy_(2) = _max(0); // br.x
        box_xy_(3) = _min(1); // br.y

        box_zy_ = Mat1f(1, 4);
        box_zy_(0) = _min(2); // tl.x // closest top depth
        box_zy_(1) = _max(1); // tl.y
        box_zy_(2) = _max(2); // br.x
        box_zy_(3) = _min(1); // br.y
    }
    else if(point_coords.rows == 1) {

        cog_ = point_coords.row(0);
        box_xy_ = cog_.clone();
        box_zy_ = cog_.clone();
    }
    else {
        cog_ = Mat1f(0, NB_FIELDS);
        box_xy_ = Mat1f(0, 4);
        box_zy_ = Mat1f(0, 4);
    }
}

float BoundingBox3D::diagonal() const
{
    float dx = box_xy_(2) - box_xy_(0);
    float dy = box_xy_(1) - box_xy_(3);
    float dz = box_zy_(2) - box_zy_(0);

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
    float dz = box_zy_(2) - box_zy_(0);

    return dx*dy*dz;
}

Matx23f BoundingBox3D::cubeVertices() const
{
    Matx23f cube;
    cube(0) = box_xy_(0); // cx_min;
    cube(1) = box_xy_(3); // cy_min;
    cube(2) = box_zy_(0); // cz_min;
    cube(3) = box_xy_(2); // cx_max;
    cube(4) = box_xy_(1); // cy_max;
    cube(5) = box_zy_(2); // cz_max;
    return cube;
}

void BoundingBox3D::cubeVertices(const Matx23f &cube)
{
    box_xy_(0) = cube(0); // cx_min;
    box_xy_(3) = box_zy_(1) = cube(1); // cy_min;
    box_zy_(0) = cube(2); // cz_min;
    box_xy_(2) = cube(3); // cx_max;
    box_xy_(1) = box_zy_(3) = cube(4); // cy_max;
    box_zy_(2) = cube(5); // cz_max;

    reduce(cube, cog_, 0, CV_REDUCE_AVG);
}
