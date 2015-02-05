// load point clouds and align them to generate full 3D model

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
//#include <pcl/common/io.h>
//#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/recognition/auxiliary.h>
#include <sstream>



int main()
{
     // load reference data
     pcl::PointCloud<pcl::PointXYZRGB> cloud1;
     pcl::io::loadPCDFile ("Dose_views/PointCloud_1.pcd", cloud1);

     // compute the mean of the reference point cloud
     Eigen::Vector4f centroid; //ignore 4th entry
     pcl::compute3DCentroid(cloud1, centroid);
     // position of the axis of rotation, relative to computed centroid
     Eigen::Vector4f offset(0, 0, 0.05, 0);
     Eigen::Vector4f fix = centroid + offset;
     Eigen::Vector4f fix_neg = -fix;



     // specify axis of rotation (magnitude = angle in radians) and construct transformation matrix
     float axis_angle[3] = {0, float(9 * 2 * M_PI / 360) , 0};
     float rotation_matrix[9];
     pcl::recognition::aux::axisAngleToRotationMatrix(axis_angle, rotation_matrix);
     // very ugly code...
     Eigen::Matrix< float, 4, 4 > transform;
     transform(0,0) = rotation_matrix[0];
     transform(0,1) = rotation_matrix[1];
     transform(0,2) = rotation_matrix[2];
     transform(1,0) = rotation_matrix[3];
     transform(1,1) = rotation_matrix[4];
     transform(1,2) = rotation_matrix[5];
     transform(2,0) = rotation_matrix[6];
     transform(2,1) = rotation_matrix[7];
     transform(2,2) = rotation_matrix[8];
     transform(0,3) = 0;
     transform(1,3) = 0;
     transform(2,3) = 0;
     transform(3,0) = 0;
     transform(3,1) = 0;
     transform(3,2) = 0;
     transform(3,3) = 1;


     // load and rotate the other point clouds
     for (int i = 2; i <= 41; i++)
     {
        std::string num = static_cast<std::ostringstream*>( &(std::ostringstream() << i))->str();
        std::string fn_in = "Dose_views/PointCloud_" + num + ".pcd";
        std::string fn_out = "Dose_rot/cloud" + num + "_rot.pcd";

        pcl::PointCloud<pcl::PointXYZRGB> cloud2;
        pcl::PointCloud<pcl::PointXYZRGB> cloud2_rot;
        pcl::io::loadPCDFile (fn_in, cloud2);

        pcl::PointCloud<pcl::PointXYZRGB> cloud_temp1;
        pcl::PointCloud<pcl::PointXYZRGB> cloud_temp2;
        // perform transformation
        pcl::demeanPointCloud(cloud2, fix, cloud_temp1);
        for (int j = 1; j < i; j++)
        {
            pcl::copyPointCloud<pcl::PointXYZRGB>(cloud_temp1, cloud_temp2);
            pcl::transformPointCloud(cloud_temp2, cloud_temp1, transform);
        }
        pcl::demeanPointCloud(cloud_temp1, fix_neg, cloud2_rot);


        // save the result
        pcl::io::savePCDFileASCII(fn_out, cloud2_rot);

        // merge point clouds
        cloud1 += cloud2_rot;
     }

     pcl::io::savePCDFileASCII("big_cloud.pcd", cloud1);

}
