// match point cloud to 3D object model
// reduced number of degrees of freedom

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
#include <pcl/features/normal_3d.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


// visualization
static bool visualize = 1;
#include <pcl/visualization/pcl_visualizer.h>
static boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));


typedef boost::variate_generator<boost::mt19937, boost::normal_distribution<float> > generator;

// parametrization
static const int n = 2;  // degrees of freedom
static const int vary[2] = { 0, 1 };  // specify which parameters should be optimized

// initial values and step sizes
static float rotInit = 0.0;
static float rotStd = 0.1;

// parameters for normal estimation
static int KSearch = 100;

// parameters for cost function and simulated annealing
double p1 = 0.1;
double cWeight = 1;
static double varWeight = 0;   // weight of the variance in cost function
static float T0 = 100;              // initial temperature
static double beta1 = 0.998;      // temperature decay in each iteration
static double beta2 = 0.995;      // variance decay (gaussian sampling of the next candidate state in simulated annealing)
static int K = 1500;
static double beta3 = 0.995;
static int it_interval = 1000;      // stopping rule for simulated annealing: the algorithm stopps if the relative change of the function value
static float min_change = 0.01;    // within it_interval iterations is smaller than min_change


// function declarations
float cost(const Eigen::VectorXf params, pcl::PointCloud<pcl::PointXYZRGB> * model,
           pcl::PointCloud<pcl::PointXYZRGB> * query, pcl::KdTreeFLANN<pcl::PointXYZRGB> * kdtree, pcl::PointCloud<pcl::PointXYZRGBNormal> & model_normals);
float cost2(const Eigen::VectorXf params, pcl::PointCloud<pcl::PointXYZRGB> * model,
           pcl::PointCloud<pcl::PointXYZRGB> * query, pcl::KdTreeFLANN<pcl::PointXYZRGB> * kdtree,  pcl::PointCloud<pcl::PointXYZRGBNormal> & model_normals);
float simuatedAnnealing(Eigen::VectorXf & params, pcl::PointCloud<pcl::PointXYZRGB> *model,
                        pcl::PointCloud<pcl::PointXYZRGB> *query, pcl::KdTreeFLANN<pcl::PointXYZRGB> * kdtree,  pcl::PointCloud<pcl::PointXYZRGBNormal> & model_normals);
void sample_gaussian(const Eigen::VectorXf & params, Eigen::VectorXf & neighbor, int it, generator * sampler);
void sample_gaussian_single(const Eigen::VectorXf & params, Eigen::VectorXf & neighbor, int it, generator * sampler, int i);
void sample_gaussian_randomComps(const Eigen::VectorXf & params, Eigen::VectorXf & neighbor, int it, generator * sampler);
void sample(const Eigen::VectorXf & params, Eigen::VectorXf & neighbor, int it, generator * sampler, int i, pcl::PointCloud<pcl::PointXYZRGB> *model, pcl::KdTreeFLANN<pcl::PointXYZRGB> * kdtree);
void homogeneousTransform(const Eigen::VectorXf & params, pcl::PointCloud<pcl::PointXYZRGBNormal> & model_normals, pcl::PointCloud<pcl::PointXYZRGB> & query, pcl::PointCloud<pcl::PointXYZRGB> & aligned);

int main()
{
    //random seed
    srand((unsigned int) time(0));

    // load 3D object model
    std::string fn_model = "/home/anna/catkin_ws/src/ni_vision/ni_vision/src/triangulation/Dose_3DMod.pcd";
    pcl::PointCloud<pcl::PointXYZRGB> model;
    pcl::io::loadPCDFile (fn_model, model);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr model_Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::copyPointCloud<pcl::PointXYZRGB>(model, *model_Ptr);

    // load query point cloud
    std::string fn_query = "/home/anna/catkin_ws/src/ni_vision/ni_vision/src/triangulation/Dose_views/PointCloud_1.pcd";
    pcl::PointCloud<pcl::PointXYZRGB> query;
    pcl::io::loadPCDFile (fn_query, query);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr query_Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::copyPointCloud<pcl::PointXYZRGB>(query, *query_Ptr);

    // centering of model
    Eigen::Vector4f centroid_model; //ignore 4th entry
    pcl::compute3DCentroid(model, centroid_model);
    pcl::demeanPointCloud(model, centroid_model, model);

    // normal estimation for model
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr searchTree (new pcl::search::KdTree<pcl::PointXYZRGB>);
    searchTree->setInputCloud (model_Ptr);
    pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> nEst;
    pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
    nEst.setInputCloud (model_Ptr);
    nEst.setSearchMethod (searchTree);
    nEst.setKSearch (KSearch);
    nEst.compute (*normals);
    pcl::PointCloud<pcl::PointXYZRGBNormal> model_normals;
    pcl::concatenateFields (model, *normals, model_normals);

    // construct kd-tree for model (for efficient nearest neighbor search)
    pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;
    kdtree.setInputCloud (model_Ptr);



    // find point in query which is closest to the centroid, use it as fix point for rotation
    Eigen::Vector4f centroid_query;
    pcl::compute3DCentroid(query, centroid_query);
    pcl::PointCloud<pcl::PointXYZRGB>::iterator it;
    double minDist = std::numeric_limits<double>::max();
    double dist;
    int idx = 0;
    int minIdx = 0;
    pcl::PointXYZRGB fixPoint;
    for (it = query.begin(); it < query.end(); it++)
    {
        dist = (it->x - centroid_query[0])*(it->x - centroid_query[0])
                + (it->y - centroid_query[1])*(it->y - centroid_query[1])
                + (it->z - centroid_query[2])*(it->z - centroid_query[2]);
        if (dist < minDist)
        {
            minIdx = idx;
            minDist = dist;
            fixPoint = *it;
        }
        idx++;
    }
    Eigen::Vector4f fp_eigen;
    fp_eigen << fixPoint.x, fixPoint.y, fixPoint.z, 0;
    pcl::demeanPointCloud(query, fp_eigen, query);

    // normal estimation for fix point
    pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree_query;
    kdtree_query.setInputCloud (query_Ptr);
    std::vector<int> pointIdxNKNSearch(KSearch);
    std::vector<float> pointNKNSquaredDistance(KSearch);
    kdtree_query.nearestKSearch (fixPoint, KSearch, pointIdxNKNSearch, pointNKNSquaredDistance);
    pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> nEst2;
    float nx, ny, nz, curvature;
    nEst2.computePointNormal (query, pointIdxNKNSearch, nx, ny, nz, curvature);

    // rotate query point cloud to align normal vector with z-axis (this will simplify some calculations)
    Eigen::Vector3f a, b;
    if (nx != 0)
        a << -ny/nx, 1, 0;
    else if (ny != 0)
        a << 0, -nz/ny, 1;
    else
        a << 1, 0, -nx/nz;
    a = a / a.norm();
    b << a[1]*nz - a[2]*ny, a[2]*nx - a[0]*nz, a[0]*ny - a[1]*nx;
    Eigen::Matrix< float, 4, 4 > transform;
    transform.row(0) << a[0], a[1], a[2], 0;
    transform.row(1) << b[0], b[1], b[2], 0;
    transform.row(2) << nx, ny, nz, 0;
    transform.row(3) << 0, 0, 0, 1;
    pcl::transformPointCloud(query, query, transform);


    // variable to store parameters
    Eigen::VectorXf params(n);

    // initialize parameters
    params[0] = M_PI/4;
    params[1] = rand() % model.points.size();



    /*
    pcl::PointCloud<pcl::PointXYZRGB> aligned;
    //pcl::copyPointCloud<pcl::PointXYZRGB>(query, aligned);
    homogeneousTransform(params, model_normals, query, aligned);
    */

    /*
    //check if it worked
    // normal estimation for fix point
    pcl::copyPointCloud<pcl::PointXYZRGB>(query, *query_Ptr);
    kdtree_query.setInputCloud (query_Ptr);
    kdtree_query.nearestKSearch (query.points[idx], KSearch, pointIdxNKNSearch, pointNKNSquaredDistance);
    pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> nEst3;
    nx, ny, nz, curvature;
    nEst3.computePointNormal (query, pointIdxNKNSearch, nx, ny, nz, curvature);
    std::cout << nx << ", " << ny << ", " << nz << std::endl;
    */



    // visualization
    pcl::copyPointCloud<pcl::PointXYZRGB>(model, *model_Ptr);
    viewer->addPointCloud<pcl::PointXYZRGB> (model_Ptr, "model");
    //viewer->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal> (model_Ptr, normals, 10, 0.05, "normals");
    //pcl::PointCloud<pcl::PointXYZRGB>::Ptr aligned_Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
    //pcl::copyPointCloud<pcl::PointXYZRGB>(aligned, *aligned_Ptr);
    //viewer->addPointCloud<pcl::PointXYZRGB> (aligned_Ptr, "aligned");
    //viewer->addCoordinateSystem(0.1);
    //viewer->spin();



    // optimize parameters
    float cost = simuatedAnnealing(params, &model, &query, &kdtree, model_normals);
    std::cout << "final result:" << std::endl;
    std::cout << params << std::endl;
    std::cout << "value of cost function: " << cost << std::endl;



    /*
    // show results for various combinations af parameters
    float best = INFINITY;
    float best0, best1;
    Eigen::Vector2f test;

    int numstep0 = 11;
    double step0 = M_PI/4;
    int step1 = 500;
    int numstep1 = 50;

    cv::Mat results = cv::Mat::zeros(numstep0,numstep1,CV_32FC1);
    for (int i = 0; i < numstep0 ; i++)
    {
        std::cout << i << std::endl;
        for (int j = 0; j < numstep1; j++)
        {
            test[0] = i * step0;
            test[1] = j * step1;
            float newc = cost(test, &model, &query, &kdtree, model_normals);
            std::cout << newc << std::endl;
            results.at<float>(i,j) = newc;
            if (newc < best)
            {
                best = newc;
                best0 = i * step0;
                best1 = j * step1;
            }
            //std::cout << test[0] << ", " << test[1] << ", " << newc << std::endl;
        }
    }
    double min, max;
    cv::minMaxLoc(results, &min, &max);

    std::cout << "best result: " << best0 << ", " << best1 << ", " << best << std::endl;

    cv::Mat mapped;
    results.convertTo(mapped,CV_8U,255.0/(max-min),-255.0*min/(max-min));
    cv::namedWindow("results", cv::WINDOW_NORMAL);
    cv::imshow("results", mapped);
    cv::waitKey();

    params << best0, best1;
    */

    // transform query using the final parameters and save result to pcd file
    pcl::PointCloud<pcl::PointXYZRGB> aligned;
    homogeneousTransform(params, model_normals, query, aligned);
    pcl::PointCloud<pcl::PointXYZRGB> result;
    pcl::copyPointCloud<pcl::PointXYZRGB>(aligned, result);
    result += model;
    pcl::io::savePCDFileASCII("3D_match.pcd", result);



    return 0;
}





// compute costs of a given set of parameters
float cost(const Eigen::VectorXf params, pcl::PointCloud<pcl::PointXYZRGB> * model,
           pcl::PointCloud<pcl::PointXYZRGB> * query, pcl::KdTreeFLANN<pcl::PointXYZRGB> * kdtree, pcl::PointCloud<pcl::PointXYZRGBNormal> & model_normals)
{

    // transform query
    pcl::PointCloud<pcl::PointXYZRGB> aligned;
    homogeneousTransform(params, model_normals, *query, aligned);

    // compute cost function
    float cost = 0;
    int K = 1;         // for k-nearest neighbors
    float normConst = sqrt(3) * 255.0 / p1;  //normalization constant, RGB distances should be comparable to XYZ distances


    pcl::PointXYZRGB neighbor;
    pcl::PointCloud<pcl::PointXYZRGB>::iterator it;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr model_Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::copyPointCloud<pcl::PointXYZRGB>(*model, *model_Ptr);
    pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree_model;
    kdtree_model.setInputCloud (model_Ptr);


    // iterate over transformed cloud
    for (it = aligned.begin(); it < aligned.end(); it++)
    {
        pcl::PointXYZRGB searchPoint = *it;
        std::vector<int> pointIdxNKNSearch(K);
        std::vector<float> pointNKNSquaredDistance(K);
        kdtree_model.nearestKSearch (searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance);
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
           pcl::PointCloud<pcl::PointXYZRGB> * query, pcl::KdTreeFLANN<pcl::PointXYZRGB> * kdtree,  pcl::PointCloud<pcl::PointXYZRGBNormal> & model_normals)
{

    // transform query
    pcl::PointCloud<pcl::PointXYZRGB> aligned;
    homogeneousTransform(params, model_normals, *query, aligned);

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
                        pcl::PointCloud<pcl::PointXYZRGB> *query, pcl::KdTreeFLANN<pcl::PointXYZRGB> * kdtree,  pcl::PointCloud<pcl::PointXYZRGBNormal> & model_normals)
{
    // random seed
    generator sampler(boost::mt19937(time(0)),boost::normal_distribution<float>(0,1));

    // temperature
    float T = T0;

    // some variables
    Eigen::VectorXf params_best(n);  // best parameter values found
    Eigen::VectorXf neighbor(n); // new candidate state
    float f_old, f_new, f_best;  // old, current and best value of the cost function
    f_best = f_old = cost(params, model, query, kdtree, model_normals);
    std::cout << "initial costs: " << f_old << std::endl;
    int it = 1;  //number of iterations
    float dE;
    float f_before[it_interval+1] ;

    do
    {
        std::cout << "aktuelle Kosten: " << f_old << std::endl;
        f_before[(it-1) % (it_interval+1)] = f_old;


            // draw new candidate state and compute corresponding value of the cost function
            sample(params, neighbor, it, &sampler, 0, model, kdtree);
            f_new = cost(neighbor, model, query, kdtree, model_normals);

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
                    homogeneousTransform(neighbor, model_normals, *query, aligned);
                    pcl::PointCloud<pcl::PointXYZRGB>::Ptr ref(new pcl::PointCloud<pcl::PointXYZRGB>);
                    pcl::copyPointCloud<pcl::PointXYZRGB>(aligned, *ref);
                    viewer->addPointCloud<pcl::PointXYZRGB> (ref, "aligned");
                    std::cout << "press q to continue" << std::endl;
                    viewer->spin();
                    viewer->removePointCloud("aligned");
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



/*
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
    std::cout << "index: " << i << std::endl;
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
*/


void sample(const Eigen::VectorXf & params, Eigen::VectorXf & neighbor, int it, generator * sampler, int i, pcl::PointCloud<pcl::PointXYZRGB> *model, pcl::KdTreeFLANN<pcl::PointXYZRGB> * kdtree)
{

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr model_Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::copyPointCloud<pcl::PointXYZRGB>(*model, *model_Ptr);
    pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree_model;
    kdtree_model.setInputCloud (model_Ptr);

    static double fact1 = 1;
    fact1 *= beta2;
    static double fact2 = 1;
    fact2 *= beta3;

        neighbor = params;
        neighbor[0] += fact1 * rotStd * (*sampler)();

        int K_now = K * fact2;
        std::vector<int> pointIdxNKNSearch(K_now);
        std::vector<float> pointNKNSquaredDistance(K_now);
        kdtree_model.nearestKSearch (model->points[params[1]], K_now, pointIdxNKNSearch, pointNKNSquaredDistance);
        neighbor = params;
        neighbor[1] = pointIdxNKNSearch[rand() % K_now];

}


void homogeneousTransform(const Eigen::VectorXf & params, pcl::PointCloud<pcl::PointXYZRGBNormal> & model_normals, pcl::PointCloud<pcl::PointXYZRGB> & query, pcl::PointCloud<pcl::PointXYZRGB> & aligned)
{
    pcl::PointXYZRGBNormal anchor = model_normals.points[params[1]];

    Eigen::Vector3f a, b;
    float nx, ny ,nz;
    nx = anchor.normal_x;
    ny = anchor.normal_y;
    nz = anchor.normal_z;

    if (fabs(nx) > fabs(ny) && fabs(nx) > fabs(nz))
        a << -(ny*ny + nz*nz)/nx , ny, nz;
    else if (fabs(ny) > fabs(nx) && fabs(ny) > fabs(nz))
        a << nx, -(nx*nx+nz*nz)/ny, nz;
    else
        a << nx, ny, -(nx*nx+ny*ny)/nz;
    a = a / a.norm();
    b << a[1]*nz - a[2]*ny, a[2]*nx - a[0]*nz, a[0]*ny - a[1]*nx;

    Eigen::Matrix< float, 4, 4 > transform_normal;
    transform_normal.row(0) << a[0], b[0], nx, anchor.x;
    transform_normal.row(1) << a[1], b[1], ny, anchor.y;
    transform_normal.row(2) << a[2], b[2], nz, anchor.z;
    transform_normal.row(3) << 0, 0, 0, 1;


    Eigen::Matrix< float, 4, 4 > transform_rot;
    transform_rot.row(0) << cos(params[0]) , -sin(params[0]) , 0 , 0;
    transform_rot.row(1) << sin(params[0]) , cos(params[0]) , 0 , 0;
    transform_rot.row(2) << 0, 0, 1, 0;
    transform_rot.row(3) << 0, 0, 0, 1;

    Eigen::Matrix< float, 4, 4 > transform_total = transform_normal * transform_rot;
    pcl::transformPointCloud(query, aligned, transform_total);

    /*
    Eigen::Vector4f model_normal;
    model_normal << model_normals.points[params[1]].normal_x, model_normals.points[params[1]].normal_y, model_normals.points[params[1]].normal_z, 1;
    Eigen::Vector4f query_normal;
    query_normal << 0, 0, 1, 1;
    Eigen::Vector4f normal_rot = transform_total * query_normal;
    std::cout << model_normal[0] << ", " << model_normal[1] << ", " << model_normal[2] << std::endl;
    std::cout << normal_rot[0] << ", " << normal_rot[1] << ", " << normal_rot[2] << std::endl;
    std::cout << transform_total << std::endl;
    */
}





