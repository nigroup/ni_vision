/*
 * Functions for preprocessing of segmentation; creating depth map/depth-gradient map
 */

#include "elm/core/pcl/typedefs_fwd.h"

namespace pcl {

class PointXYZRGB;

}

/* Create depth map from RGB point cloud
 *
 * Input:
 * cloud - RGB point cloud from kinect (vector for RGB and vector for point cloud (z coordinate is depth))
 * nDsSize - size of point cloud
 * nDMax, nDMin - max and min values of depth data
 * nDIdxCntTmp - count of valid point
 * vnCloudIdx_d - indices of valid points
 *
 * Output:
 * vnX, vnY, vnZ - output vectors for point cloud coordinates
 */
void MakeDepthMap (pcl::PointCloud<pcl::PointXYZRGB> cloud, int nDsSize, int nDsWidth, float &nDMax, float &nDMin,
                   int &nDIdxCntTmp, std::vector<int> &vnCloudIdx_d, std::vector<float> &vnX, std::vector<float> &vnY, std::vector<float> &vnZ);

/* Create depth map from monochrome point cloud
 *
 * Input:
 * cloud - monochrome point cloud from kinect (vector for point cloud (z coordinate is depth))
 * nDsSize - size of point cloud
 * nDMax, nDMin - max and min values of depth data
 * nDIdxCntTmp - count of valid point
 * vnCloudIdx_d - indices of valid points
 *
 * Output:
 * vnX, vnY, vnZ - output vectors for point cloud coordinates
 */
void MakeDepthMap (pcl::PointCloud<pcl::PointXYZ> cloud, int nDsSize, float &nDMax, float &nDMin,
                   int &nDIdxCntTmp, std::vector<int> &vnCloudIdx_d, std::vector<float> &vnX, std::vector<float> &vnY, std::vector<float> &vnZ);



/* Create depth-gradient map from depth map
 *
 * Input:
 * vDepth - depth map
 * vnCloudIdx_d - indices of valid points
 * nDIdxCntTmp - count of valid points
 * nDGradConst - constant of the weighted depth
 * nSegmDThres - threshold for very steep depth-gradient
 * nDGradNan - constant for the very steep depth-gradient
 * nDsWidth - width of depth map
 * nDGradXMin, nDGradXMax, nDGradYMin, nDGradYMax - max and min values of depth-gradient data
 *
 * Output:
 * vDGradX,vDGradY - depth-gradient maps
 */
void MakeDGradMap (std::vector<float> vDepth, std::vector<int> vCloudIdx_d, int nDIdxCntTmp, float nDGradConst, float nSegmDThres, float nDGradNan, int nDsWidth,
                      float &nDGradXMin, float &nDGradXMax, float &nDGradYMin, float &nDGradYMax, std::vector<float> &vDGradX, std::vector<float> &vDGradY);
