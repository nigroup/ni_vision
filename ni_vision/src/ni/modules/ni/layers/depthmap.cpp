#include "ni/layers/depthmap.h"

#include "elm/core/exception.h"
#include "elm/core/layerconfig.h"
#include "elm/core/signal.h"
#include "elm/ts/layerattr_.h"\

#include "ni/core/img_utils.h"

using namespace std;
using namespace cv;
using namespace elm;
using namespace ni;

/** Define parameters and I/O keys
  */
const string DepthMap::PARAM_DEPTH_MAX = "depth_min";
const float DepthMap::DEFAULT_DEPTH_MAX = 3.f;

/** @todo why does define guard lead to undefined reference error?
 */
//#ifdef __WITH_GTEST
#include <boost/assign/list_of.hpp>
template <>
elm::MapIONames LayerAttr_<DepthMap>::io_pairs = boost::assign::map_list_of
        ELM_ADD_INPUT_PAIR(detail::BASE_SINGLE_INPUT_FEATURE_LAYER__KEY_INPUT_STIMULUS)
        ELM_ADD_OUTPUT_PAIR(detail::BASE_MATOUTPUT_LAYER__KEY_OUTPUT_RESPONSE)
        ;
//#endif

///* Create depth map from RGB point cloud
// *
// * Input:
// * cloud - RGB point cloud from kinect (vector for RGB and vector for point cloud (z coordinate is depth))
// * nDsSize - size of point cloud
// * nDMax, nDMin - max and min values of depth data
// * nDIdxCntTmp - count of valid point
// * vnCloudIdx_d - indices of valid points
// *
// * Output:
// * vnX, vnY, vnZ - output vectors for point cloud coordinates
// */
//void MakeDepthMap (const CloudXYZPtr cld, int nDsSize, int nDsWidth, float &nDMax, float &nDMin,
//                   int &nDIdxCntTmp, std::vector<int> &vnCloudIdx_d, std::vector<float> &vnX, std::vector<float> &vnY, std::vector<float> &vnZ)
//{
//    float nDLimit = 0;
//    if(nDsWidth <= 320) {
//        for(int i = 1; i < nDsSize; i++) {// Ignore first pixel. It has wrong depth info
//            if (!pcl_isfinite (cld->points[i].z)) continue;
//            vnX[i] = cld->points[i].x;
//            vnY[i] = cld->points[i].y;
//            vnZ[i] = fabs(cld->points[i].z);
//            if (vnZ[i] > nDLimit) vnZ[i] = nDLimit;
//            if (vnZ[i] > nDMax) nDMax = vnZ[i];
//            if (vnZ[i] < nDMin) nDMin = vnZ[i];
//            vnCloudIdx_d[nDIdxCntTmp++] = i;
//        }
//    }
//    else {
//        for(int i = 1; i < nDsSize; i++) {// the first pixel has wrong depth info
//            int x, y;
//            indexToRowCol(i, nDsWidth, x, y);
//            if(!pcl_isfinite (cld->points[i].z)) continue;
//            vnCloudIdx_d[nDIdxCntTmp++] = i;

//            if(x%2 || y%2) continue;
//            vnX[i] = cld->points[i].x; vnX[i+1] = vnX[i]; vnX[i+nDsWidth] = vnX[i]; vnX[i+nDsWidth+1] = vnX[i];
//            vnY[i] = cld->points[i].y; vnY[i+1] = vnY[i]; vnY[i+nDsWidth] = vnY[i]; vnY[i+nDsWidth+1] = vnY[i];
//            vnZ[i] = fabs(cld->points[i].z); vnZ[i+1] = vnZ[i]; vnZ[i+nDsWidth] = vnZ[i]; vnZ[i+nDsWidth+1] = vnZ[i];
//            if(vnZ[i] > nDLimit) vnZ[i] = nDLimit;
//            if(vnZ[i] > nDMax) nDMax = vnZ[i];
//            if(vnZ[i] < nDMin) nDMin = vnZ[i];

//        }
//    }

//    vnCloudIdx_d.resize(nDIdxCntTmp);
//}

void DepthMap::Clear()
{
    m_ = Mat1f();
}

void DepthMap::Reset(const LayerConfig &config)
{
    Reconfigure(config);
}

void DepthMap::Reconfigure(const LayerConfig &config)
{
    depth_max_ = config.Params().get<float>(PARAM_DEPTH_MAX, 3.f);
}

void DepthMap::Activate(const Signal &signal)
{
    CloudXYZPtr cld = signal.MostRecent(name_input_).get<CloudXYZPtr>();

//    int nDsSize = cld->width*cld->height;
//    std::vector<float> vnX(nDsSize, 0), vnY(nDsSize, 0), vnZ(nDsSize, 0);   // vnZ: Depth Map
//    std::vector<int> vnCloudIdx_d(nDsSize, 0);                              // valid Point Cloud indices

//    float min_depth, max_depth;
//    int nDIdxCndTmp;
//    MakeDepthMap(cld, nDsSize, cld->width, min_depth, max_depth, nDIdxCndTmp, vnCloudIdx_d, vnX, vnY, vnZ);

    // - convert point cloud data into a Mat
    // - reshape to data of 1 point per row
    // - extract z column
    m_ = PointCloud2Mat_<pcl::PointXYZ>(cld);

    m_ = m_.reshape(1, static_cast<int>(cld->size()));
    m_ = m_.col(2).t();

    // - reshape in case of organized point cloud
    if(cld->isOrganized()) {

        m_ = m_.clone().reshape(1, cld->height);
    }

    // - mask depth values that are too large
    m_.setTo(depth_max_, m_ > depth_max_);
}

DepthMap::DepthMap()
    : base_FeatureTransformationLayer()
{
    Clear();
}

DepthMap::DepthMap(const LayerConfig& config)
    : base_FeatureTransformationLayer(config)
{
    Clear();
    Reconfigure(config);
    IONames(config);
}
