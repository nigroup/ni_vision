#include <pcl/io/pcd_io.h>
#include <ctime>


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


/* store point cloud data to pcd file
 *returns false while the process is not yet finished
 *returns true if all point clouds have been stored
 */
template <typename PointT>
bool RecPcl2(pcl::PointCloud<PointT> & cloud)
{
    // directory and file name
    static const std::string sPclDir = "Pcl_data";
    mkdir (sPclDir.data(), 0777);
    std::string sPcl_fn;
    string ext;

    //time control
    static clock_t curTime;
    static clock_t lastTime;
    static int count = 0;
    int maxnum = 10; //number of point clouds to be recorded

    clock_t delay = 5 * CLOCKS_PER_SEC; //delay time in clock units
    curTime = clock();

    if (curTime - lastTime > delay)
    {
        lastTime = curTime;
        ext = static_cast<ostringstream*>( &(ostringstream() << (count + 1)))->str();
        sPcl_fn = sPclDir + "/" + "PointCloud_" + ext + ".pcd";
        pcl::io::savePCDFileASCII(sPcl_fn, cloud);
        count++;
        if (count == maxnum)
        {
            count = 0;
            return true;
        }
        else
            return false;
    }
    return false;
}


