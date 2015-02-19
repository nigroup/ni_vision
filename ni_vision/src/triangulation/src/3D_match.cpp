// match point cloud to 3D object model

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/recognition/auxiliary.h>
#include <sstream>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <iostream>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/kdtree.h>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>


// visualization
#include <pcl/visualization/pcl_visualizer.h>
boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));


typedef boost::variate_generator<boost::mt19937, boost::normal_distribution<float> > generator;

// parametrization
static const int n = 6;  // degrees of freedom
static const bool vary[n] = {1, 1, 1, 1, 1, 1};  // specify which parameters should be optimized (true / 1) and which should be considered fixed (false / 0)

// initial values and step sizes
static const float pRange[n][2] =  { {0.0, 0.01},
                                     {0.0, 0.01},
                                     {0.0, 0.01},
                                     {0.0, 1.0},
                                     {0.0, 1.0},
                                     {0.0, 1.0} };


// parameters for cost function and simulated annealing
double p1 = 0.1;          // maximum distance of neighbors which are considered a match in cost function
double cWeight = 1;        // weight of RGB distance relative to XYZ distance in cost function
float T0 = 1;              // initial temperature
double beta1 = 0.995;      // temperature decay in each iteration
double beta2 = 0.998;      // variance decay (gaussian sampling of the next candidate state in simulated annealing)
int max_it_noAccept = 100;  // stopping rule for simulated annealing: maximum number of iterations without acceptance of new state


// function declarations
float cost(const Eigen::VectorXf params, pcl::PointCloud<pcl::PointXYZRGB> * model,
           pcl::PointCloud<pcl::PointXYZRGB> * query, pcl::KdTreeFLANN<pcl::PointXYZRGB> * kdtree);
float simuatedAnnealing(Eigen::VectorXf & params, pcl::PointCloud<pcl::PointXYZRGB> *model,
                        pcl::PointCloud<pcl::PointXYZRGB> *query, pcl::KdTreeFLANN<pcl::PointXYZRGB> * kdtree);
void sample_gaussian(const Eigen::VectorXf & params, Eigen::VectorXf & neighbor, int it, generator * sampler);
void homogeneousTransform(const Eigen::VectorXf & params, pcl::PointCloud<pcl::PointXYZRGB> & query, pcl::PointCloud<pcl::PointXYZRGB> & aligned);



int main()
{
    // load 3D object model
    std::string fn_model = "Dose_3DMod.pcd";
    pcl::PointCloud<pcl::PointXYZRGB> model;
    pcl::io::loadPCDFile (fn_model, model);

    // load query point cloud
    std::string fn_query = "Dose_views/PointCloud_1.pcd";
    pcl::PointCloud<pcl::PointXYZRGB> query;
    pcl::io::loadPCDFile (fn_query, query);

    // rough alignment: centering
    Eigen::Vector4f centroid_model; //ignore 4th entry
    Eigen::Vector4f centroid_query;
    pcl::compute3DCentroid(model, centroid_model);
    pcl::compute3DCentroid(query, centroid_query);
    pcl::demeanPointCloud(model, centroid_model, model);
    pcl::demeanPointCloud(query, centroid_query, query);

    // variable to store parameters
    Eigen::VectorXf params(n);


    // initialize parameters
    for (int i = 0; i < n; i++)
        params[i] = pRange[i][0];


    // construct kd-tree for model (for efficient nearest neighbor search)
    pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr ref(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::copyPointCloud<pcl::PointXYZRGB>(model, *ref);
    kdtree.setInputCloud (ref);

    // visualization
    viewer->addPointCloud<pcl::PointXYZRGB> (ref, "model");

    // optimize parameters
    float cost = simuatedAnnealing(params, &model, &query, &kdtree);
    std::cout << "final result:" << std::endl;
    std::cout << params << std::endl;
    std::cout << "value of cost function: " << cost << std::endl;


    // transform query using the final parameters and save result to pcd file
    pcl::PointCloud<pcl::PointXYZRGB> aligned;
    homogeneousTransform(params, query, aligned);
    pcl::PointCloud<pcl::PointXYZRGB> result;
    pcl::copyPointCloud<pcl::PointXYZRGB>(aligned, result);
    result += model;
    pcl::io::savePCDFileASCII("3D_match.pcd", result);



    return 0;
}





// compute costs of a given set of parameters
float cost(const Eigen::VectorXf params, pcl::PointCloud<pcl::PointXYZRGB> * model,
           pcl::PointCloud<pcl::PointXYZRGB> * query, pcl::KdTreeFLANN<pcl::PointXYZRGB> * kdtree)
{

    // transform query
    pcl::PointCloud<pcl::PointXYZRGB> aligned;
    homogeneousTransform(params, *query, aligned);

    // compute cost function
    float cost = 0;
    int K = 1;         // for k-nearest neighbors
    float normConst = sqrt(3) * 255.0 / p1;  //normalization constant, RGB distances should be comparable to XYZ distances


    pcl::PointXYZRGB neighbor;
    pcl::PointCloud<pcl::PointXYZRGB>::iterator it;

    // iterate over transformed cloud
    for (it = aligned.begin(); it < aligned.end(); it++)
    {
        pcl::PointXYZRGB searchPoint = *it;
        std::vector<int> pointIdxNKNSearch(K);
        std::vector<float> pointNKNSquaredDistance(K);
        kdtree->nearestKSearch (searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance);
        neighbor = model->points[pointIdxNKNSearch[0]];
        double dist = sqrt(pointNKNSquaredDistance[0]);

        // compute difference between RGB values
        float RGB1_f = searchPoint.rgb;
        float RGB2_f = neighbor.rgb;
        uint32_t RGB1_i, RGB2_i;
        memcpy(&RGB1_i, &RGB1_f, sizeof(RGB1_f));
        memcpy(&RGB2_i, &RGB2_f, sizeof(RGB2_f));
        //std::bitset<32> bb(RGB1_i);
        //std::cout << bb << std::endl;
        Eigen::Vector3f rgb1((RGB1_i >> 16) & 0x0000ff, (RGB1_i >> 8) & 0x0000ff, (RGB1_i) & 0x0000ff);
        Eigen::Vector3f rgb2((RGB2_i >> 16) & 0x0000ff, (RGB2_i >> 8) & 0x0000ff, (RGB2_i) & 0x0000ff);
        Eigen::Vector3f RGBdiff = rgb1 - rgb2;
        double cdist = RGBdiff.norm() / normConst;    //is the euclidean metric a sensible choice?


        // update cost
        if (dist < p1)
            cost = cost + dist - p1 + cWeight * cdist;
    }


    return cost;
}


// simulated annealing routine
float simuatedAnnealing(Eigen::VectorXf & params, pcl::PointCloud<pcl::PointXYZRGB> *model,
                        pcl::PointCloud<pcl::PointXYZRGB> *query, pcl::KdTreeFLANN<pcl::PointXYZRGB> * kdtree)
{
    // random seed
    srand((unsigned int) time(0));
    generator sampler(boost::mt19937(time(0)),boost::normal_distribution<float>(0,1));

    // temperature
    float T = T0;

    // some variables
    Eigen::VectorXf params_best(n);  // best parameter values found
    Eigen::VectorXf neighbor(n); // new candidate state
    float f_old, f_new, f_best;  // old, current and best value of the cost function
    f_best = f_old = cost(params, model, query, kdtree);
    std::cout << "initial costs: " << f_old << std::endl;
    int it = 1;  //number of iterations
    int it_noAccept = 0;  //number of iterations without acceptance of new state
    float dE;

    do
    {
        std::cout << "aktuelle Kosten: " << f_old << std::endl;

        // draw new candidate state and compute corresponding value of the cost function
        sample_gaussian(params, neighbor, it, &sampler);
        std::cout << neighbor << std::endl;
        pcl::PointCloud<pcl::PointXYZRGB> aligned;
        homogeneousTransform(params, *query, aligned);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr ref(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::copyPointCloud<pcl::PointXYZRGB>(aligned, *ref);
        viewer->addPointCloud<pcl::PointXYZRGB> (ref, "aligned");
        std::cout << "press q to continue" << std::endl;
        viewer->spin();
        viewer->removePointCloud("aligned");


        f_new = cost(neighbor, model, query, kdtree);
        if (f_new < f_best)
        {
            f_best = f_new;
            params_best = neighbor;
        }
        dE = f_new - f_old;

        // accept or refuse to accept new state
        if ( dE < 0 || exp(-dE/T) > (double(rand()) / RAND_MAX) )
        {
            params = neighbor;
            f_old = f_new;
            it_noAccept = 0;
        }
        else
            it_noAccept ++;

        // update temperature (annealing) and iteration counter
        T *= beta1;
        it ++;
    } while( it_noAccept < max_it_noAccept);

    std::cout << "number of iterations in simulated annealing: " << it-1 << std::endl;

    params = params_best;
    return f_best;
}


// generate new sample for simulated annealing
// sample from Gaussian distribution centered around the current state
void sample_gaussian(const Eigen::VectorXf & params, Eigen::VectorXf & neighbor, int it, generator * sampler)
{
    static double fact = 1;
    fact *=beta2;
    neighbor = params;
    for (int i = 0; i < n; i++)
        if (vary[i])
            neighbor[i] += fact * pRange[i][1] * (*sampler)();
    //std::cout << neighbor << std::endl;
}


void homogeneousTransform(const Eigen::VectorXf & params, pcl::PointCloud<pcl::PointXYZRGB> & query, pcl::PointCloud<pcl::PointXYZRGB> & aligned)
{
    Eigen::AngleAxisd rollAngle(params[3], Eigen::Vector3d::UnitZ());
    Eigen::AngleAxisd yawAngle(params[4], Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd pitchAngle(params[5], Eigen::Vector3d::UnitX());
    Eigen::Quaternion<double> q = rollAngle * yawAngle * pitchAngle;
    Eigen::Matrix3d rotation_matrix = q.matrix();
    Eigen::Matrix< float, 4, 4 > transform;
    transform.row(0) << rotation_matrix(0,0) , rotation_matrix(0,1) ,rotation_matrix(0,2) , params[0];
    transform.row(1) << rotation_matrix(1,0) , rotation_matrix(1,1) ,rotation_matrix(1,2) , params[1];
    transform.row(2) << rotation_matrix(2,0) , rotation_matrix(2,1) ,rotation_matrix(2,2) , params[2];
    transform.row(3) << 0, 0, 0, 1;
    pcl::transformPointCloud(query, aligned, transform);
}


