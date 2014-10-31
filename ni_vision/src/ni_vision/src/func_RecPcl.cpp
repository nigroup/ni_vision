#include <pcl/io/pcd_io.h>


void RecPcl(const sensor_msgs::PointCloud2ConstPtr cloud_, boost::mutex & m)
{
    pcl::PointCloud<pcl::PointXYZ> cloud_xyz;
    pcl::PointCloud<pcl::PointXYZRGB> cloud_xyz_rgb;

    static const std::string sPclDir = "Pcl_data";
    mkdir (sPclDir.data(), 0777);
    std::string sPcl_fn;

    char cExt[] = "0";

    for (int nPclIdx = 1; nPclIdx <= 5; nPclIdx++)
    {
        cExt[0]++;
        sPcl_fn = sPclDir + "/" + "PointCloud_" + std::string(cExt) + ".pcd";
        m.lock ();
        if (pcl::getFieldIndex (*cloud_, "rgb") != -1)
        {
            pcl::fromROSMsg (*cloud_, cloud_xyz_rgb);
            pcl::io::savePCDFileASCII(sPcl_fn, cloud_xyz_rgb);
        }
        else
        {
            pcl::fromROSMsg (*cloud_, cloud_xyz);
            pcl::io::savePCDFileASCII(sPcl_fn, cloud_xyz);
        }
        m.unlock ();
    }
}


/*
struct CloudPtr
{
    pcl::PointCloud<pcl::PointXYZ> * XYZPtr;
    pcl::PointCloud<pcl::PointXYZRGB> * XYZRGBPtr;

    CloudPtr(pcl::PointCloud<pcl::PointXYZ> & inp){XYZPtr = &inp; XYZRGBPtr = NULL;}
    CloudPtr(pcl::PointCloud<pcl::PointXYZRGB> & inp){XYZRGBPtr = &inp; XYZPtr = NULL;}
};


void RecPcl2(CloudPtr cloudP)
{
    static struct timespec curTime;
    static struct timespec lastTime = -1;

    clock_gettime(CLOCK_MONOTONIC_RAW, curTime);


    bool isRgb;
    if (cloudP.XYZRGBPtr != NULL)
    {
        isRgb = true;
        pcl::PointCloud<pcl::PointXYZRGB> * cloud = cloudP.XYZRGBPtr;
    }
    else
    {
        iRgb = false;
        pcl::PointCloud<pcl::PointXYZ> * cloud = cloudP.XYZPtr;
    }

    static const std::string sPclDir = "Pcl_data";
    mkdir (sPclDir.data(), 0777);
    std::string sPcl_fn;

    char cExt[] = "0";

    for (int nPclIdx = 1; nPclIdx <= 5; nPclIdx++)
    {
        cExt[0]++;
        sPcl_fn = sPclDir + "/" + "PointCloud_" + std::string(cExt) + ".pcd";
        m.lock ();
        if (pcl::getFieldIndex (*cloud_, "rgb") != -1)
        {
            pcl::fromROSMsg (*cloud_, cloud_xyz_rgb);
            pcl::io::savePCDFileASCII(sPcl_fn, cloud_xyz_rgb);
        }
        else
        {
            pcl::fromROSMsg (*cloud_, cloud_xyz);
            pcl::io::savePCDFileASCII(sPcl_fn, cloud_xyz);
        }
        m.unlock ();
    }
}
*/
