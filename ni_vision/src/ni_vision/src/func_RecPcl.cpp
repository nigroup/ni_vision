#include <pcl/io/pcd_io.h>
#include <ctime>
#include "gnuplot-iostream.h"
#include <pcl_ros/filters/filter.h>
#include <limits>


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
 *maxnum: number of snapshots to be recorded
 *delayS: delay time in seconds
 *width, height, depth: size of the bounding box (in meters)
 *cThreshRel: in the range [0,1]; relative (to max) threshold for rgb values
 */
//template <typename PointT>
bool RecPcl(const pcl::PointCloud<pcl::PointXYZRGB> & cloud,
            int maxnum, int delayS, float width, float height, float depth, float cThreshRel)
{
    // directory where the data is stored and file name
    static const std::string sPclDir = "Pcl_data";
    mkdir (sPclDir.data(), 0777);
    std::string sPcl_fn;
    string ext;

    //time control
    static clock_t curTime;
    static clock_t lastTime;
    static int count = 0;

    clock_t delay = delayS * CLOCKS_PER_SEC; //delay time in clock units
    curTime = clock();

    if (curTime - lastTime > delay)
    {
        lastTime = curTime;
        ext = static_cast<ostringstream*>( &(ostringstream() << (count + 1)))->str();
        sPcl_fn = sPclDir + "/" + "PointCloud_" + ext + ".pcd";

        //edit point cloud
        //apply threshold
        pcl::PointCloud<pcl::PointXYZRGB> cloud_thresh = cloud;
        pcl::PointCloud<pcl::PointXYZRGB>::iterator it;

        //find range of the rgb values
        double min, max;
        it = cloud_thresh.points.begin();
        min = max = it->rgb;
        for (it = cloud_thresh.points.begin(); it < cloud_thresh.points.end(); it++)
        {
            if(it->rgb > max)
                max = it->rgb;
            else if(it->rgb < min)
                min = it->rgb;
        }

        //define bounding box
        vector<pcl::PointXYZ> box(2);
        box[0] = pcl::PointXYZ(- width / 2, - height / 2, 0);
        box[1] = pcl::PointXYZ(width / 2, height / 2, depth);

        //remove unwanted values (set them to nan)
        double threshold = cThreshRel * max;
        const float nan = std::numeric_limits<float>::quiet_NaN();
        for (it = cloud_thresh.points.begin(); it < cloud_thresh.points.end(); it++)
        {
            if(it->rgb < threshold || it->x < box[0].x || it->x > box[1].x || it->y < box[0].y || it->y > box[1].y || it->z < box[0].z || it->z > box[1].z)
                it->x = nan;
        }


        //remove nan
        pcl::PointCloud<pcl::PointXYZRGB> cloud_mod;
        std::vector<int> index;
        pcl::removeNaNFromPointCloud(cloud_thresh, cloud_mod, index);


        //write point cloud data to .pcd file
        pcl::io::savePCDFileASCII(sPcl_fn, cloud_mod);


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
        static boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
        pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr ptrCloud(&cloud);
        viewer->addPointCloud<pcl::PointXYZRGB> (ptrCloud, "sample cloud");
        viewer->initCameraParameters ();
        if (!viewer->wasStopped ())
        {
          viewer->spinOnce (100);
          boost::this_thread::sleep (boost::posix_time::microseconds (100000));
        }
        viewer->removePointCloud();   */

        /*
        //try to convert point cloud to 2D image; works in thread; viewer can also be declared static
        static pcl::visualization::ImageViewer viewer("View Clouds");
        viewer.showRGBImage(cloud);   */


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
        {
            return false;
        }

    }
    return false;
}






