// load point clouds and align them to generate full 3D model

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
//#include <pcl/common/io.h>
//#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/recognition/auxiliary.h>
#include <sstream>
#include <Eigen/Dense>
#include <iostream>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/kdtree.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>



bool optimize = 0;


static const int n = 2;
static const int numClouds = 41;
float cost(float params[n], pcl::PointCloud<pcl::PointXYZRGB> * clouds, Eigen::Vector4f centroid);
float * simuatedAnnealing(float params[n], pcl::PointCloud<pcl::PointXYZRGB> * clouds, Eigen::Vector4f centroid);

int main()
{
     // load data
    pcl::PointCloud<pcl::PointXYZRGB> clouds[numClouds];
    for (int i = 0; i < numClouds; i++)
    {
        std::string num = static_cast<std::ostringstream*>( &(std::ostringstream() << (i+1)))->str();
        std::string fn_in = "Dose_views/PointCloud_" + num + ".pcd";
        pcl::io::loadPCDFile (fn_in, clouds[i]);
    }


     // compute the mean of the reference point cloud
     Eigen::Vector4f centroid; //ignore 4th entry
     pcl::compute3DCentroid(clouds[1], centroid);

     // variable to store final parameters
     float * params;
     float params_a[n];

     //----------------------------------------------------------------------------------------
     if (!optimize)
     {
         // show results for various combinations af parameters
         float best = INFINITY;
         float bestx, bestz;
         float test[2];

         int lenx = 21;
         int lenz = 21;
         float midx = 0;
         float midz = 0.04;
         float step = 0.001;

         cv::Mat results = cv::Mat::zeros(lenx,lenz,CV_32FC1);
         for (int i = 0; i < lenx; i++)
         {
             std::cout << i << std::endl;
             for (int j = 0; j < lenz; j++)
             {
                 test[0] = midx+(-(lenx/2)+i)*step;
                 test[1] = midz+(-(lenz/2)+j)*step;
                 float newc = cost(test, clouds, centroid);
                 results.at<float>(i,j) = newc;
                 if (newc < best)
                 {
                     best = newc;
                     bestx = midx+(-(lenx/2)+i)*step;
                     bestz = midz+(-(lenz/2)+j)*step;
                 }
                 //std::cout << test[0] << ", " << test[1] << ", " << newc << std::endl;
             }
         }
         double min, max;
         cv::minMaxLoc(results, &min, &max);

         std::cout << "best result: " << bestx << ", " << bestz << ", " << best << std::endl;

         cv::Mat mapped;
         results.convertTo(mapped,CV_8U,255.0/(max-min),-255.0*min/(max-min));
         cv::namedWindow("results", cv::WINDOW_NORMAL);
         cv::imshow("results", mapped);
         cv::waitKey();

         params_a[0]=bestx;
         params_a[1]=bestz;
         params = &params_a[0];
     }
     //------------------------------------------------------------------------------------------------


     if (optimize)
     {
         // initialize parameters
         float params_init[n] = {0, 0};

         // optimize parameters
         params = simuatedAnnealing(params_init, clouds, centroid);
         std::cout << "final result:" << std::endl;
         for (int i = 0; i < n; i++)
             std::cout << params[i] << std::endl;
     }

     //---------------------------------------------------------------------------------------------------

     // rotate point clouds using the optimized parameters and store the result
     std::cout << "rotate point clouds using bestx = " << params[0] << ", bestz = " << params[1] << "..." << std::endl;
     Eigen::Vector4f offset(params[0], 0, params[1], 0);
     Eigen::Vector4f fix = centroid + offset;
     Eigen::Vector4f fix_neg = -fix;
     float axis_angle[3] = {0, double(9 * 2 * M_PI / 360) , 0};   // 9° steps
     float rotation_matrix[9];
     pcl::recognition::aux::axisAngleToRotationMatrix(axis_angle, rotation_matrix);
     Eigen::Matrix< float, 4, 4 > transform;
     transform.row(0) << rotation_matrix[0] , rotation_matrix[1] ,rotation_matrix[2] , 0;
     transform.row(1) << rotation_matrix[3] , rotation_matrix[4] ,rotation_matrix[5] , 0;
     transform.row(2) << rotation_matrix[6] , rotation_matrix[7] ,rotation_matrix[8] , 0;
     transform.row(3) << 0, 0, 0, 1;
     pcl::PointCloud<pcl::PointXYZRGB> cloud_temp1;
     pcl::PointCloud<pcl::PointXYZRGB> cloud_temp2;
     pcl::PointCloud<pcl::PointXYZRGB> cloud_rot;
     pcl::PointCloud<pcl::PointXYZRGB> result;
     pcl::copyPointCloud<pcl::PointXYZRGB>(clouds[0], result);
     for (int i = 1; i < numClouds; i++)
     {
         pcl::demeanPointCloud(clouds[i], fix, cloud_temp1);
         for (int j = 1; j <= i; j++)
         {
             pcl::copyPointCloud<pcl::PointXYZRGB>(cloud_temp1, cloud_temp2);
             pcl::transformPointCloud(cloud_temp2, cloud_temp1, transform);
         }
         pcl::demeanPointCloud(cloud_temp1, fix_neg, cloud_rot);
         result += cloud_rot;
     }

     pcl::io::savePCDFileASCII("SA_result.pcd", result);



     return 0;
}



// simulated annealing routine
float * simuatedAnnealing(float params[n], pcl::PointCloud<pcl::PointXYZRGB> *clouds, Eigen::Vector4f centroid)
{

    // initialize step sizes
    float ss[n];
    ss[0] = 0.001;   //0.1cm
    ss[1] = 0.001;   //0.1cm

    // annealing schedule
    float T0 = 0.01;
    float T;

    // some variables
    int idx;
    int dir;
    float neighbor[n];
    for (int i = 0; i < n; i++)
        neighbor[i] = params[i];
    float f_old, f_new;
    f_new = cost(params, clouds, centroid);
    int it = 1;

    do
    {
        std::cout << "aktuelle Kosten: " << f_old << std::endl;
        T = T0/log(it); // annealing schedule
        idx = rand() % n;        // pick vector component to alter
        dir = (rand() % 2) - 1;  // increase or decrease
        neighbor[idx] += dir * ss[idx];
        f_new = cost(neighbor, clouds, centroid);
        if (exp( (f_old - f_new) / T ) > (double(rand()) / RAND_MAX) )
        {
            params[idx] = neighbor[idx];
            f_old = f_new;
        }
        else
        {
            neighbor[idx] = params[idx];
        }
        it ++;
    } while( it < 1000);

    return params;
}



// compute costs of a given set of parameters
float cost(float params[n], pcl::PointCloud<pcl::PointXYZRGB> *clouds, Eigen::Vector4f centroid)
{
    // position of the axis of rotation, relative to computed centroid
    Eigen::Vector4f offset(params[0], 0, params[1], 0);
    Eigen::Vector4f fix = centroid + offset;
    Eigen::Vector4f fix_neg = -fix;

    // specify axis of rotation (magnitude = angle in radians) and construct transformation matrix
    float axis_angle[3] = {0, double(9 * 2 * M_PI / 360) , 0};   // 9° steps
    float rotation_matrix[9];
    pcl::recognition::aux::axisAngleToRotationMatrix(axis_angle, rotation_matrix);

    Eigen::Matrix< float, 4, 4 > transform;
    transform.row(0) << rotation_matrix[0] , rotation_matrix[1] ,rotation_matrix[2] , 0;
    transform.row(1) << rotation_matrix[3] , rotation_matrix[4] ,rotation_matrix[5] , 0;
    transform.row(2) << rotation_matrix[6] , rotation_matrix[7] ,rotation_matrix[8] , 0;
    transform.row(3) << 0, 0, 0, 1;


    // transform clouds (not the first one = reference)
    pcl::PointCloud<pcl::PointXYZRGB> cloud_temp1;
    pcl::PointCloud<pcl::PointXYZRGB> cloud_temp2;
    pcl::PointCloud<pcl::PointXYZRGB> clouds_rot[numClouds];
    for (int i = 1; i <= 1; i++)  //erstmal nur die ersten beiden...
    {
        pcl::demeanPointCloud(clouds[i], fix, cloud_temp1);
        for (int j = 1; j <= i; j++){
            pcl::copyPointCloud<pcl::PointXYZRGB>(cloud_temp1, cloud_temp2);
            pcl::transformPointCloud(cloud_temp2, cloud_temp1, transform);
        }
        pcl::demeanPointCloud(cloud_temp1, fix_neg, clouds_rot[i]);
    }


    // compute cost function
    float cost = 0;
    int K = 1;
    double p1 = 0.01;  // some parameter that has to be adjusted


    // use XYZ to find the nearest neighbor
    pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr ref(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::copyPointCloud<pcl::PointXYZRGB>(clouds[0], *ref);
    kdtree.setInputCloud (ref);

    // iterate over second point cloud
    pcl::PointXYZRGB neighbor;
    pcl::PointCloud<pcl::PointXYZRGB>::iterator it;
    for (it = clouds_rot[1].begin(); it < clouds_rot[1].end(); it++)
    {
        pcl::PointXYZRGB searchPoint = *it;
        std::vector<int> pointIdxNKNSearch(K);
        std::vector<float> pointNKNSquaredDistance(K);
        kdtree.nearestKSearch (searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance);
        neighbor = ref->points[pointIdxNKNSearch[0]];
        double dist = sqrt(pointNKNSquaredDistance[0]);

        // compute difference between RGB values
        uint32_t RGB1 = searchPoint.rgb;
        uint32_t RGB2 = neighbor.rgb;
        uint8_t r1 = (int(RGB1) >> 16) & 0x0000ff;
        uint8_t g1 = (RGB1 >> 8) & 0x0000ff;
        uint8_t b1 = (RGB1) & 0x0000ff;
        uint8_t r2 = (RGB2 >> 16) & 0x0000ff;
        uint8_t g2 = (RGB2 >> 8) & 0x0000ff;
        uint8_t b2 = (RGB2) & 0x0000ff;
        //std::cout << r1 << ", " << g1 << ", " << b1 << std::endl;
        //std::cout << r2 << ", " << g2 << ", " << b2 << std::endl;
        std::cout << int(r1) << std::endl;


        if (dist < p1)
            cost = cost + dist - p1;
    }

    return cost;
}









