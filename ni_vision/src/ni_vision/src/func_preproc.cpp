/*
 * Functions for preprocessing of segmentation; creating depth map/depth-gradient map
 */


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
void MakeDepthMap (pcl::PointCloud<pcl::PointXYZRGB> cloud, int nDsSize, float &nDMax, float &nDMin,
                   int &nDIdxCntTmp, std::vector<int> &vnCloudIdx_d, std::vector<float> &vnX, std::vector<float> &vnY, std::vector<float> &vnZ)
{
    for (int i = 1; i < nDsSize; i++) {// the first pixel has wrong depth info
        if (!pcl_isfinite (cloud.points[i].z)) continue;
        vnX[i] = cloud.points[i].x;
        vnY[i] = cloud.points[i].y;
        vnZ[i] = fabs(cloud.points[i].z);
        if (vnZ[i] > nDLimit) vnZ[i] = nDLimit;
        if (vnZ[i] > nDMax) nDMax = vnZ[i];
        if (vnZ[i] < nDMin) nDMin = vnZ[i];
        vnCloudIdx_d[nDIdxCntTmp++] = i;
    }
    vnCloudIdx_d.resize(nDIdxCntTmp);
}



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
                   int &nDIdxCntTmp, std::vector<int> &vnCloudIdx_d, std::vector<float> &vnX, std::vector<float> &vnY, std::vector<float> &vnZ)
{
    for (int i = 1; i < nDsSize; i++) {// the first pixel has wrong depth info
        if (!pcl_isfinite (cloud.points[i].z)) continue;
        vnX[i] = cloud.points[i].x;
        vnY[i] = cloud.points[i].y;
        vnZ[i] = fabs(cloud.points[i].z);
        if (vnZ[i] > nDLimit) vnZ[i] = nDLimit;
        if (vnZ[i] > nDMax) nDMax = vnZ[i];
        if (vnZ[i] < nDMin) nDMin = vnZ[i];
        vnCloudIdx_d[nDIdxCntTmp++] = i;
    }
    vnCloudIdx_d.resize(nDIdxCntTmp);
}



/* Create depth-gradient map from depth map
 *
 * Input:
 * vDepth - depth map
 * vnCloudIdx_d - indices of valid points
 * nDIdxCntTmp - count of valid points
 * nDGradConst - constant of the weighted depth
 * nDSegmDThres - threshold for very steep depth-gradient
 * nDGradNan - constant for the very steep depth-gradient
 * nDsWidth - width of depth map
 * nDGradXMin, nDGradXMax, nDGradYMin, nDGradYMax - max and min values of depth-gradient data
 *
 * Output:
 * vDGradX,vDGradY - depth-gradient maps
 */
void MakeDGradMap (std::vector<float> vDepth, std::vector<int> vCloudIdx_d, int nDIdxCntTmp, float nDGradConst, float nDSegmDThres, float nDGradNan, int nDsWidth,
                      float &nDGradXMin, float &nDGradXMax, float &nDGradYMin, float &nDGradYMax, std::vector<float> &vDGradX, std::vector<float> &vDGradY)
{
    /////// Processing depth map: make depth gradient map ////////////////////////////////////////////////
    for(int i = 0; i < nDIdxCntTmp; i++) {
        int x, y, idx;
        idx = vCloudIdx_d[i];
        GetPixelPos(idx, nDsWidth, x, y);
        if (!vDepth[idx]) continue;

        if (y > 0) {
            if (vDepth[idx-nDsWidth]) {
                //vDGradY[idx] = vDepth[idx] - vDepth[idx-nDsWidth];
                vDGradY[idx] = (vDepth[idx] - vDepth[idx-nDsWidth]) / (vDepth[idx] + nDGradConst);
                if (fabs(vDGradY[idx]) > nDSegmDThres) vDGradY[idx] = nDGradNan;
                else {
                    if (vDGradY[idx] > nDGradYMax) nDGradYMax = vDGradY[idx];
                    if (vDGradY[idx] < nDGradYMin) nDGradYMin = vDGradY[idx];
                }
            }
        }
        if (x > 0) {
            if (vDepth[idx-1]) {
                vDGradX[idx] = vDepth[idx] - vDepth[idx-1];
                if (fabs(vDGradX[idx]) > nDSegmDThres) vDGradX[idx] = nDGradNan;
                else {
                    if (vDGradX[idx] > nDGradXMax) nDGradXMax = vDGradX[idx];
                    if (vDGradX[idx] < nDGradXMin) nDGradXMin = vDGradX[idx];
                }
            }
        }
    }
}
