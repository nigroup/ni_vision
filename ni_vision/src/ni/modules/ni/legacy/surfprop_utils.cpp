#include "ni/legacy/surfprop_utils.h"

#include <opencv2/core/core.hpp>

#include "elm/core/typedefs_sfwd.h"
#include "elm/core/cv/mat_vector_utils_inl.h"

#include "ni/core/boundingbox2d.h"
#include "ni/legacy/surfprop.h"

using namespace cv;
using namespace elm;
using namespace ni;

void ni::SurfPropToVecSurfaces(const SurfProp &surf_prop, std::vector<Surface> &surfaces)
{
    for(size_t i=1; i<surf_prop.vnIdx.size(); i++) {

        Surface s;

        s.id(surf_prop.vnIdx[i]);
        s.pixelIndices(surf_prop.mnPtsIdx[i]); // sets pixel count also.

        // x, y, w=xmax-x, h=ymax-y
        Rect2i r(surf_prop.mnRect[i][0],
                surf_prop.mnRect[i][1],
                surf_prop.mnRect[i][2]-r.x,
                surf_prop.mnRect[i][3]-r.y);
        s.rect(r);

        VecF tmp = surf_prop.mnCubic[i];
        s.cubeVertices(elm::Vec_ToRowMat_<float>(tmp).reshape(1, 2).clone()); // also sets cube center

        s.colorHistogram(Mat1f(surf_prop.mnColorHist[i], true));
        s.diagonal(surf_prop.vnLength[i]);

        surfaces[i-1] = s;
    }
}

void ni::VecSurfacesToSurfProp(const std::vector<Surface> &surfaces, SurfProp &surf_prop)
{
    for(size_t i=0; i<surfaces.size(); i++) {

        Surface s = surfaces[i];

        surf_prop.vnIdx[i] = s.id();
        surf_prop.vnPtsCnt[i] = s.pixelCount();
        surf_prop.mnPtsIdx[i] = s.pixelIndices(); // do we need this?

        Rect2i r = s.rect();
        surf_prop.mnRect[i][0] = r.x;
        surf_prop.mnRect[i][1] = r.y;
        surf_prop.mnRect[i][2] = r.x + r.width;
        surf_prop.mnRect[i][3] = r.y + r.height;

        BoundingBox2D bbox(r);
        Mat1f pt = bbox.centralPoint();
        surf_prop.mnRCenter[i][0] = static_cast<int>(pt(0));
        surf_prop.mnRCenter[i][1] = static_cast<int>(pt(1));

        surf_prop.mnCubic[i]   = elm::Mat_ToVec_(Mat1f(s.cubeVertices()));
        surf_prop.mnCCenter[i] = elm::Mat_ToVec_(Mat1f(s.cubeCenter()));
        surf_prop.mnColorHist[i] = elm::Mat_ToVec_(s.colorHistogram());
        surf_prop.vnLength[i] = s.diagonal();
    }
}
