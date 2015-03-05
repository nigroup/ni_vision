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
#include <limits>
#include <cmath>
#include <numeric>


// visualization
bool visualize = 1;
#include <pcl/visualization/pcl_visualizer.h>
boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));


typedef boost::variate_generator<boost::mt19937, boost::normal_distribution<float> > generator;

// parametrization
static const int n = 6;  // degrees of freedom
static const int vary[6] = { 0, 1, 2, 3, 4, 5};  // specify which parameters should be optimized

// initial values and step sizes
static const float pRange[n][2] =  { {0.025, 0.1},
                                     {0.0, 0.1},
                                     {0.0, 0.1},
                                     {0.0, 1},
                                     {0.0, 1},
                                     {0.0, 1} };


// parameters for cost function and simulated annealing
double varWeight = 0;   // weight of the variance in cost function
double p1 = 0.01;          // maximum distance of neighbors which are considered a match in cost function
double cWeight = 1;        // weight of RGB distance relative to XYZ distance in cost function
float T0 =1;              // initial temperature
double beta1 = 0.995;      // temperature decay in each iteration
double beta2 = 0.998;      // variance decay (gaussian sampling of the next candidate state in simulated annealing)
int max_it_noAccept = 100;  // stopping rule for simulated annealing: maximum number of iterations without acceptance of new state
int it_interval = 100;      // stopping rule for simulated annealing: the algorithm stopps if the relative change of the function value
float min_change = 0.01;    // within it_interval iterations is smaller than min_change


// function declarations
float cost(const Eigen::VectorXf params, pcl::PointCloud<pcl::PointXYZRGB> * model,
           pcl::PointCloud<pcl::PointXYZRGB> * query, pcl::KdTreeFLANN<pcl::PointXYZRGB> * kdtree);
float cost2(const Eigen::VectorXf params, pcl::PointCloud<pcl::PointXYZRGB> * model,
           pcl::PointCloud<pcl::PointXYZRGB> * query, pcl::KdTreeFLANN<pcl::PointXYZRGB> * kdtree);
float simuatedAnnealing(Eigen::VectorXf & params, pcl::PointCloud<pcl::PointXYZRGB> *model,
                        pcl::PointCloud<pcl::PointXYZRGB> *query, pcl::KdTreeFLANN<pcl::PointXYZRGB> * kdtree);
void sample_gaussian(const Eigen::VectorXf & params, Eigen::VectorXf & neighbor, int it, generator * sampler);
void sample_gaussian_single(const Eigen::VectorXf & params, Eigen::VectorXf & neighbor, int it, generator * sampler, int i);
void sample_gaussian_randomComps(const Eigen::VectorXf & params, Eigen::VectorXf & neighbor, int it, generator * sampler);
void homogeneousTransform(const Eigen::VectorXf & params, pcl::PointCloud<pcl::PointXYZRGB> & query, pcl::PointCloud<pcl::PointXYZRGB> & aligned);
void homogeneousTransform2(const Eigen::VectorXf & params, pcl::PointCloud<pcl::PointXYZRGB> & query, pcl::PointCloud<pcl::PointXYZRGB> & aligned);
void homogeneousTransform3(const Eigen::VectorXf & params, pcl::PointCloud<pcl::PointXYZRGB> & query, pcl::PointCloud<pcl::PointXYZRGB> & aligned);


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



// compute costs of a given set of parameters
float cost2(const Eigen::VectorXf params, pcl::PointCloud<pcl::PointXYZRGB> * model,
           pcl::PointCloud<pcl::PointXYZRGB> * query, pcl::KdTreeFLANN<pcl::PointXYZRGB> * kdtree)
{

    // transform query
    pcl::PointCloud<pcl::PointXYZRGB> aligned;
    homogeneousTransform(params, *query, aligned);


    std::vector<double> distances(aligned.points.size());
    int K = 1;
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
        distances.push_back(sqrt(pointNKNSquaredDistance[0]));
    }

    double sum = std::accumulate(distances.begin(), distances.end(), 0.0);
    double mean = sum / distances.size();

    sum = 0.0;
    for (std::vector<double>::iterator d = distances.begin(); d < distances.end(); d++)
        sum += (*d - mean) * (*d - mean);
    double var = sum / (distances.size()-1);

    float cost = mean + varWeight * var;
    //std::cout << "mean: " << mean << ", variance: " << var << std::endl;

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
    float dE;
    float f_before[it_interval+1] ;

    do
    {
        std::cout << "aktuelle Kosten: " << f_old << std::endl;
        f_before[(it-1) % (it_interval+1)] = f_old;

        for (int i = 0; i < sizeof(vary)/sizeof(vary[0]); i++)
        {
            // draw new candidate state and compute corresponding value of the cost function
            sample_gaussian_single(params, neighbor, it, &sampler, vary[i]);

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

                if (visualize)
                {
                    pcl::PointCloud<pcl::PointXYZRGB> aligned;
                    homogeneousTransform(neighbor, *query, aligned);
                    pcl::PointCloud<pcl::PointXYZRGB>::Ptr ref(new pcl::PointCloud<pcl::PointXYZRGB>);
                    pcl::copyPointCloud<pcl::PointXYZRGB>(aligned, *ref);
                    viewer->addPointCloud<pcl::PointXYZRGB> (ref, "aligned");
                    std::cout << "press q to continue" << std::endl;
                    viewer->spin();
                    viewer->removePointCloud("aligned");
                }
            }
        }


        // update temperature (annealing) and iteration counter
        T *= beta1;
        it ++;
    } while( it <= it_interval+1 || fabs(f_before[(it-2) % it_interval]-f_old)/fabs(f_before[(it-2) % it_interval]) > min_change);
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


// generate new sample for simulated annealing
// one component/dimension at a time
void sample_gaussian_single(const Eigen::VectorXf & params, Eigen::VectorXf & neighbor, int it, generator * sampler, int i)
{
    static double fact = 1;
    fact *=beta2;
    neighbor = params;
    neighbor[i] += fact * pRange[i][1] * (*sampler)();
    //std::cout << neighbor << std::endl;
}


// generate new sample for simulated annealing
// random number of components to be disturbed
void sample_gaussian_randomComps(const Eigen::VectorXf & params, Eigen::VectorXf & neighbor, int it, generator * sampler)
{
    static double fact = 1;
    fact *=beta2;
    neighbor = params;
    for (int i = 0; i < sizeof(vary)/sizeof(vary[0]); i++)
    {
        int mut = (rand() % 2);  // decide wether to change component or not
        if (mut)
            neighbor[vary[i]] += fact * pRange[vary[i]][1] * (*sampler)();;
    }
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


void homogeneousTransform2(const Eigen::VectorXf & params, pcl::PointCloud<pcl::PointXYZRGB> & query, pcl::PointCloud<pcl::PointXYZRGB> & aligned)
{
    float a2 = params[5] / sqrt(1 + params[3]*params[3] + params[4]*params[4]);
    float a1 = a2 * params[3];
    float a3 = a2 * params[4];
    float axis_angle[3] = {a1, a2, a3};
    float rotation_matrix[9];
    pcl::recognition::aux::axisAngleToRotationMatrix(axis_angle, rotation_matrix);
    Eigen::Matrix< float, 4, 4 > transform;
    transform.row(0) << rotation_matrix[0] , rotation_matrix[1] ,rotation_matrix[2] , params[0];
    transform.row(1) << rotation_matrix[3] , rotation_matrix[4] ,rotation_matrix[5] , params[1];
    transform.row(2) << rotation_matrix[6] , rotation_matrix[7] ,rotation_matrix[8] , params[2];
    transform.row(3) << 0, 0, 0, 1;
    pcl::transformPointCloud(query, aligned, transform);
}



void homogeneousTransform3(const Eigen::VectorXf & params, pcl::PointCloud<pcl::PointXYZRGB> & query, pcl::PointCloud<pcl::PointXYZRGB> & aligned)
{
    float a1 = params[3];
    float a2 = params[4];
    float a3 = params[5];
    float axis_angle[3] = {a1, a2, a3};
    float rotation_matrix[9];
    pcl::recognition::aux::axisAngleToRotationMatrix(axis_angle, rotation_matrix);
    Eigen::Matrix< float, 4, 4 > transform;
    transform.row(0) << rotation_matrix[0] , rotation_matrix[1] ,rotation_matrix[2] , params[0];
    transform.row(1) << rotation_matrix[3] , rotation_matrix[4] ,rotation_matrix[5] , params[1];
    transform.row(2) << rotation_matrix[6] , rotation_matrix[7] ,rotation_matrix[8] , params[2];
    transform.row(3) << 0, 0, 0, 1;
    pcl::transformPointCloud(query, aligned, transform);
}




