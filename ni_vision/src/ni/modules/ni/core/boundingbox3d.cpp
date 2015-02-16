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
    int new_rows = point_coords.total()/PCLPointTraits_<PointXYZ>::NbFloats();
    point_coords = point_coords.reshape(1, new_rows);

    cog_ = Mat1f(1, PCLPointTraits_<PointXYZ>::NbFloats());
    reduce(point_coords, cog_, 0, CV_REDUCE_AVG);
}

float BoundingBox3D::diagonal() const
{
    ELM_THROW_NOT_IMPLEMENTED;
}

Mat1f BoundingBox3D::centralPoint() const
{
    return cog_;
}
