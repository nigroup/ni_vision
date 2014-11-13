#include <pcl/io/pcd_io.h>
#include <ctime>
#include "gnuplot-iostream.h"
#include <pcl_ros/filters/filter.h>


#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/image_viewer.h>


void RecPcl2(const sensor_msgs::PointCloud2ConstPtr cloud_, boost::mutex & m)
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
//template <typename PointT>
bool RecPcl(pcl::PointCloud<pcl::PointXYZRGB> & cloud)
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

        //edit point cloud
        //apply threshold, remove nan and resize
        pcl::PointCloud<pcl::PointXYZRGB> cloud_mod;
        std::vector<int> index;
        pcl::removeNaNFromPointCloud(cloud, cloud_mod, index);



        //write point cloud data to .pcd file
        pcl::io::savePCDFileASCII(sPcl_fn, cloud);


        /*
        //show PointCloud;  seems to be a problem with threading
        pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr ptrCloud(&cloud);
        pcl::visualization::CloudViewer viewer("Simple Cloud Viewer");
        viewer.showCloud(ptrCloud);
        while (!viewer.wasStopped())
        {
        }
        */


        /*
        //try pcl_visualizer class; does this work in a thread?
        boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
        pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr ptrCloud(&cloud);
        viewer->addPointCloud<pcl::PointXYZRGB> (ptrCloud, "sample cloud");
        viewer->initCameraParameters ();
        while (!viewer->wasStopped ())
        {
          viewer->spinOnce (100);
          boost::this_thread::sleep (boost::posix_time::microseconds (100000));
        }  */


        /*  //try to convert point cloud to 2D image;  does not work: illegal instruction (core dumped)
        pcl::visualization::ImageViewer viewer("View Clouds");
        viewer.showRGBImage(cloud);
        */

        //try sending data to gnuplot
        //Gnuplot gp;
        //gp << "splot \"" << sPcl_fn << "\" using 1:2:3:4 with dots palette\n";


        //count how many point clouds have been stored in this round
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






