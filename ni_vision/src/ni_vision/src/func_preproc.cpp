/*
 * Functions for preprocessing of segmentation
 */



void MakeDepthMap (pcl::PointCloud<pcl::PointXYZRGB> cloud, bool bDepthCalib,
                   int nCalibX, int nCalibY, int nCalibW, int nCalibH, int nCvSize, int nCvWidth, int nCvHeight, float &nDMax, float &nDMin,
                   int &nDIdxCntTmp, std::vector<int> &vnCloudIdx_d, std::vector<float> &vnX, std::vector<float> &vnY, std::vector<float> &vnZ)
{
    if (!bDepthCalib) {
        for (int i = 1; i < nCvSize; i++) {// the first pixel has wrong depth info
            if (!pcl_isfinite (cloud.points[i].z)) continue;
            vnX[i] = cloud.points[i].x;
            vnY[i] = cloud.points[i].y;
            vnZ[i] = fabs(cloud.points[i].z);
            if (vnZ[i] > nDLimit) vnZ[i] = nDLimit;
            if (vnZ[i] > nDMax) nDMax = vnZ[i];
            if (vnZ[i] < nDMin) nDMin = vnZ[i];
            vnCloudIdx_d[nDIdxCntTmp++] = i;
        }
    }
    else {
        for (int j = nCalibY; j < nCalibY+nCalibH; j++) {
            float y = (j-nCalibY)*nCvHeight/float(nCalibH);
            int y_tmp = y;
            int yy;
            if (y-y_tmp < 0.5) yy = y;
            else yy = y+1;
            for (int i = nCalibX; i < nCalibX+nCalibW; i++) {
                float x = (i-nCalibX)*nCvWidth/nCalibW;
                int x_tmp = x/1;
                int xx;
                if (x-x_tmp < 0.5) xx = x;
                else xx = x+1;

                int idx_org = yy*nCvWidth + xx;
                if (!pcl_isfinite (cloud.points[idx_org].z)) continue;

                int idx = j*nCvWidth + i;

                vnX[idx] = cloud.points[idx_org].x;
                vnY[idx] = cloud.points[idx_org].y;
                vnZ[idx] = fabs(cloud.points[idx_org].z);
                if (vnZ[idx] > nDLimit) vnZ[idx] = nDLimit;
                if (vnZ[idx] > nDMax) nDMax = vnZ[idx];
                if (vnZ[idx] < nDMin) nDMin = vnZ[idx];
                vnCloudIdx_d[nDIdxCntTmp++] = idx;
            }
        }
    }
    vnCloudIdx_d.resize(nDIdxCntTmp);
}

void MakeDepthMap (pcl::PointCloud<pcl::PointXYZ> cloud, bool bDepthCalib,
                   int nCalibX, int nCalibY, int nCalibW, int nCalibH, int nCvSize, int nCvWidth, int nCvHeight, float &nDMax, float &nDMin,
                   int &nDIdxCntTmp, std::vector<int> &vnCloudIdx_d, std::vector<float> &vnX, std::vector<float> &vnY, std::vector<float> &vnZ)
{
    if (!bDepthCalib) {
        for (int i = 1; i < nCvSize; i++) {// the first pixel has wrong depth info
            if (!pcl_isfinite (cloud.points[i].z)) continue;
            vnX[i] = cloud.points[i].x;
            vnY[i] = cloud.points[i].y;
            vnZ[i] = fabs(cloud.points[i].z);
            if (vnZ[i] > nDLimit) vnZ[i] = nDLimit;
            if (vnZ[i] > nDMax) nDMax = vnZ[i];
            if (vnZ[i] < nDMin) nDMin = vnZ[i];
            vnCloudIdx_d[nDIdxCntTmp++] = i;
        }
    }
    else {
        for (int j = nCalibY; j < nCalibY+nCalibH; j++) {
            float y = (j-nCalibY)*nCvHeight/float(nCalibH);
            int y_tmp = y;
            int yy;
            if (y-y_tmp < 0.5) yy = y;
            else yy = y+1;
            for (int i = nCalibX; i < nCalibX+nCalibW; i++) {
                float x = (i-nCalibX)*nCvWidth/nCalibW;
                int x_tmp = x/1;
                int xx;
                if (x-x_tmp < 0.5) xx = x;
                else xx = x+1;

                int idx_org = yy*nCvWidth + xx;
                if (!pcl_isfinite (cloud.points[idx_org].z)) continue;

                int idx = j*nCvWidth + i;

                vnX[idx] = cloud.points[idx_org].x;
                vnY[idx] = cloud.points[idx_org].y;
                vnZ[idx] = fabs(cloud.points[idx_org].z);
                if (vnZ[idx] > nDLimit) vnZ[idx] = nDLimit;
                if (vnZ[idx] > nDMax) nDMax = vnZ[idx];
                if (vnZ[idx] < nDMin) nDMin = vnZ[idx];
                vnCloudIdx_d[nDIdxCntTmp++] = idx;
            }
        }
    }
    vnCloudIdx_d.resize(nDIdxCntTmp);
}



void MakeDGradMap (std::vector<float> vDepth, std::vector<int> vCloudIdx_d, int nDIdxCntTmp, float nDGradConst, float nDSegmDThres, float nDGradNan, int nCvWidth,
                      float &nDGradXMin, float &nDGradXMax, float &nDGradYMin, float &nDGradYMax, std::vector<float> &vDGradX, std::vector<float> &vDGradY)
{
    /////// Processing depth map: make depth gradient map ////////////////////////////////////////////////
    for(int i = 0; i < nDIdxCntTmp; i++) {
        int x, y, idx;
        idx = vCloudIdx_d[i];
        GetPixelPos(idx, nCvWidth, x, y);
        if (!vDepth[idx]) continue;

        if (y > 0) {
            if (vDepth[idx-nCvWidth]) {
                //vDGradY[idx] = vDepth[idx] - vDepth[idx-nCvWidth];
                vDGradY[idx] = (vDepth[idx] - vDepth[idx-nCvWidth]) / (vDepth[idx] + nDGradConst);
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
