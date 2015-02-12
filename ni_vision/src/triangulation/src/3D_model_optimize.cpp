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

#include <bitset> // check binary representation of pcl rgb values

#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>


typedef boost::variate_generator<boost::mt19937, boost::normal_distribution<float> > generator;

bool optimize = 1;

std::vector<std::pair<int,int> > pairs;  // pairs of point clouds which are considered in the cost function

static const int n = 5;  // degrees of freedom
static const int numClouds = 41;  //number of point clouds
static const double rotAng = 9.0 * 2.0 * M_PI / 360.0;  //angle of rotation (turn table) in radians

float cost(const Eigen::VectorXf params, pcl::PointCloud<pcl::PointXYZRGB> * clouds, const Eigen::Vector4f centroid);
float simuatedAnnealing(Eigen::VectorXf & params, pcl::PointCloud<pcl::PointXYZRGB> * clouds, Eigen::Vector4f centroid);
void sample_ds(const Eigen::VectorXf & params, Eigen::VectorXf & neighbor);
void sample_cs(const Eigen::VectorXf & params, Eigen::VectorXf & neighbor);
void sample_gaussian(const Eigen::VectorXf & params, Eigen::VectorXf & neighbor, int it, generator * sampler);



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

    // define which pairs of clouds are considered to compute the cost function
    pairs.push_back(std::pair<int,int>(40,0));


     // compute the mean of the reference point cloud
     Eigen::Vector4f centroid; //ignore 4th entry
     pcl::compute3DCentroid(clouds[0], centroid);

     // variable to store final parameters
     Eigen::VectorXf params(n);

     //----------------------------------------------------------------------------------------
     if (!optimize)
     {
         // show results for various combinations af parameters
         float best = INFINITY;
         float bestx, bestz;
         Eigen::Vector2f test;

         int lenx = 21;
         int lenz = 21;
         float midx = 0.0;
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

         params << bestx, bestz;
     }
     //------------------------------------------------------------------------------------------------


     if (optimize)
     {
         // initialize parameters
         params.setZero();
         params[4] = rotAng;

         // optimize parameters
         float cost = simuatedAnnealing(params, clouds, centroid);
         std::cout << "final result:" << std::endl;
         std::cout << params << std::endl;
         std::cout << "value of cost function: " << cost << std::endl;
         std::cout << "angle of rotation: " << sqrt(params[2]*params[2] + params[4]*params[4] + params[4]*params[4]) * 360 / (2*M_PI) << std::endl;
     }

     //---------------------------------------------------------------------------------------------------

     // rotate point clouds using the optimized parameters and store the result
     std::cout << "rotate point clouds using the following parameters:" << std::endl;
     std::cout << params << std::endl;
     Eigen::Vector4f offset(params[0], 0, params[1], 0);
     Eigen::Vector4f fix = centroid + offset;
     Eigen::Vector4f fix_neg = -fix;
     float axis_angle[3] = {params[2], params[4], params[3]};
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

     pcl::io::savePCDFileASCII("optimization_result.pcd", result);


     return 0;
}





// compute costs of a given set of parameters
float cost(const Eigen::VectorXf params, pcl::PointCloud<pcl::PointXYZRGB> *clouds, const Eigen::Vector4f centroid)
{
    // position of the axis of rotation, relative to computed centroid
    Eigen::Vector4f offset(params[0], 0, params[1], 0);
    Eigen::Vector4f fix = centroid + offset;
    Eigen::Vector4f fix_neg = -fix;

    // specify axis of rotation (magnitude = angle in radians) and construct transformation matrix
    float axis_angle[3] = {params[2], params[4], params[3]};
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
    pcl::copyPointCloud<pcl::PointXYZRGB>(clouds[0], clouds_rot[0]);

    for (int p = 0; p < pairs.size(); p++)
    {
        pcl::demeanPointCloud(clouds[pairs[p].first], fix, cloud_temp1);
        for (int j = 1; j <= pairs[p].first; j++){
            pcl::copyPointCloud<pcl::PointXYZRGB>(cloud_temp1, cloud_temp2);
            pcl::transformPointCloud(cloud_temp2, cloud_temp1, transform);
        }
        pcl::demeanPointCloud(cloud_temp1, fix_neg, clouds_rot[pairs[p].first]);

        pcl::demeanPointCloud(clouds[pairs[p].second], fix, cloud_temp1);
        for (int j = 1; j <= pairs[p].second; j++){
            pcl::copyPointCloud<pcl::PointXYZRGB>(cloud_temp1, cloud_temp2);
            pcl::transformPointCloud(cloud_temp2, cloud_temp1, transform);
        }
        pcl::demeanPointCloud(cloud_temp1, fix_neg, clouds_rot[pairs[p].second]);
    }


    // compute cost function
    float cost = 0;
    int K = 1;         // for k-nearest neighbors
    double p1 = 0.01;  // some parameter that has to be adjusted


    for (int p = 0; p < pairs.size(); p++)
    {
        // use XYZ to find the nearest neighbor
        pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr ref(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::copyPointCloud<pcl::PointXYZRGB>(clouds_rot[pairs[p].first], *ref);
        kdtree.setInputCloud (ref);

        // iterate over second point cloud
        pcl::PointXYZRGB neighbor;
        pcl::PointCloud<pcl::PointXYZRGB>::iterator it;

        for (it = clouds_rot[pairs[p].second].begin(); it < clouds_rot[pairs[p].second].end(); it++)
        {
            pcl::PointXYZRGB searchPoint = *it;
            std::vector<int> pointIdxNKNSearch(K);
            std::vector<float> pointNKNSquaredDistance(K);
            kdtree.nearestKSearch (searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance);
            neighbor = ref->points[pointIdxNKNSearch[0]];
            double dist = sqrt(pointNKNSquaredDistance[0]);

            // compute difference between RGB values
            float RGB1_f = searchPoint.rgb;
            float RGB2_f = neighbor.rgb;
            uint32_t RGB1_i, RGB2_i;
            memcpy(&RGB1_i, &RGB1_f, sizeof(RGB1_f));
            memcpy(&RGB2_i, &RGB2_f, sizeof(RGB2_f));
            //std::bitset<32> bb(RGB1_i);
            //std::cout << bb << std::endl;
            cv::Vec3b rgb1((RGB1_i >> 16) & 0x0000ff, (RGB1_i >> 8) & 0x0000ff, (RGB1_i) & 0x0000ff);
            cv::Vec3b rgb2((RGB2_i >> 16) & 0x0000ff, (RGB2_i >> 8) & 0x0000ff, (RGB2_i) & 0x0000ff);
            float normConst = sqrt(3) * 255.0 / p1;  //normalization constant, RGB distances should be comparable to XYZ distances
            double cdist = norm(rgb1-rgb2) / normConst;    //is the euclidean metric a sensible choice?

            // update cost
            if (dist < p1)
                cost = cost + dist - p1 + cdist;
        }
    }

    return cost;
}



// simulated annealing routine
float simuatedAnnealing(Eigen::VectorXf & params, pcl::PointCloud<pcl::PointXYZRGB> *clouds, Eigen::Vector4f centroid)
{
    // random seed
    srand((unsigned int) time(0));
    generator sampler(boost::mt19937(time(0)),boost::normal_distribution<float>(0,1));

    // annealing schedule
    float T0 = 1;
    float T;

    // some variables
    Eigen::VectorXf neighbor(n);
    float f_old, f_new;
    f_new = cost(params, clouds, centroid);
    int it = 1;
    float dE;

    do
    {
        std::cout << "aktuelle Kosten: " << f_old << std::endl;
        if(it > 100)
            T = T0/it; // annealing schedule
        sample_gaussian(params, neighbor, it, &sampler);
        f_new = cost(neighbor, clouds, centroid);
        dE = f_new - f_old;
        if ( dE < 0 || exp(-dE/T) > (double(rand()) / RAND_MAX) )
        {
            params = neighbor;
            f_old = f_new;
        }
        it ++;
    } while( it < 1000);

    return f_old;
}


// generate new sample for simulated annealing
// discrete version
void sample_ds(const Eigen::VectorXf & params, Eigen::VectorXf & neighbor)
{
    // initialize step sizes
    float ss[n];
    ss[0] = 0.001;   //0.1cm
    ss[1] = 0.001;   //0.1cm

    neighbor = params;
    int idx = rand() % n;    // pick vector component to alter
    int dir = (rand() % 2);  // increase or decrease
    neighbor[idx] += (dir*1 + (1-dir)*(-1)) * ss[idx];
}


// generate new sample for simulated annealing
// add a random vector
void sample_cs(const Eigen::VectorXf & params, Eigen::VectorXf & neighbor)
{
    float rangeMax = 0.05;  //maximum step size
    neighbor = rangeMax * Eigen::VectorXf::Random(n);
    std::cout << neighbor << std::endl;
}


// generate new sample for simulated annealing
// sample from Gaussian distribution centered around the current state
void sample_gaussian(const Eigen::VectorXf & params, Eigen::VectorXf & neighbor, int it, generator * sampler)
{
    Eigen::VectorXf stds(n);
    stds << 0.01, 0.01, 0.0001, 0.001, 0.001;
    if (it > 100)
        stds /= log(it);
    neighbor = params;
    for (int i = 0; i < n; i++)
        neighbor[i] += stds[i] * (*sampler)();
    //std::cout << neighbor << std::endl;
}

















