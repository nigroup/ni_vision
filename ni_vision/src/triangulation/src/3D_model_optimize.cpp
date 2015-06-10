// load point clouds and align them to generate full 3D model
// define a suitable cost function and optimize it using simulated annealing

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


bool optimize = 1; // if false, test different combinations of parameters and visualize the cost function (for n==2)

// information about the data set
std::string setName = "/home/anna/Soennecken";   // name of the folder
static const int numClouds = 8;  //number of point clouds
static const double rotAng = 45.0 * 2.0 * M_PI / 360.0;  //angle of rotation (turn table) in radians


// information about parameters (arguments of cost function)
// the problem is parametrized as follows:
// params[0] and params[1] define the offset of the axes of rotation in x- and z-direction
// the offset is added to the center of mass of the first point cloud in the data set
// params[2], params[3] and params[4] define the orientation of the axes of rotation and the angle of rotation
// if the axes of rotation is a = (a1, a2, a3), then params[2] is the ratio a1/a2 and params[3] is the ratio a3/a2
// params[4] is the magnitude of a, which is interpreted as the angle of rotation, params[4] = sqrt(a1*a1 + a2*a2 + a3*a3)
// params[4] should be fixed, if the angle of rotation is known
// the components of a can be computed analytically from params[2], params[3] and params[4] :
// a2 = params[4] / sqrt(1 + params[2]*params[2] + params[3] * params[3])
// a1 = a2 * params[2]
// a3 = a2 * params[3]
static const int n = 5;  // degrees of freedom
static const bool vary[n] = {1, 1, 1, 1, 0};  // specify which parameters should be optimized (true / 1) and which should be considered fixed (false / 0)

// initial values and step sizes
static const float pRange[n][2] =  { {0.0, 0.01},
                                     {0.0, 0.01},
                                     {0.0, 0.1},
                                     {0.0, 0.1},
                                     {rotAng, 0.01} };


// pairs of point clouds which are considered in the cost function
static const int numPairs = 2;
static const int pairs[numPairs][2] = { {0,1} , {1,2} };

// construct kd-trees for efficient nearest neighbor search
pcl::KdTreeFLANN<pcl::PointXYZRGB> * kdtree[numPairs];


// parameters for cost function and simulated annealing
double p1 = 0.01;          // maximum distance of neighbors which are considered a match in cost function
double cWeight = 1;        // weight of RGB distance relative to XYZ distance in cost function
float T0 = 1;              // initial temperature
double beta1 = 0.995;      // temperature decay in each iteration
double beta2 = 0.998;      // variance decay (gaussian sampling of the next candidate state in simulated annealing)
int it_interval = 100;      // stopping rule for simulated annealing: the algorithm stopps if the relative change of the function value
float min_change = 0.0001;    // within it_interval iterations is smaller than min_change



// function declarations
float cost(const Eigen::VectorXf params, pcl::PointCloud<pcl::PointXYZRGB> * clouds, const Eigen::Vector4f centroid);
float simuatedAnnealing(Eigen::VectorXf & params, pcl::PointCloud<pcl::PointXYZRGB> * clouds, Eigen::Vector4f centroid);
void sample_ds(const Eigen::VectorXf & params, Eigen::VectorXf & neighbor);
void sample_cs(const Eigen::VectorXf & params, Eigen::VectorXf & neighbor);
void sample_gaussian(const Eigen::VectorXf & params, Eigen::VectorXf & neighbor, int it, generator * sampler);
void rotate( const pcl::PointCloud<pcl::PointXYZRGB> *clouds, pcl::PointCloud<pcl::PointXYZRGB> * clouds_rot, const Eigen::VectorXf & centroid,
             const Eigen::VectorXf & params);


int main()
{
     // load data
    pcl::PointCloud<pcl::PointXYZRGB> clouds[numClouds];
    for (int i = 0; i < numClouds; i++)
    {
        std::string num = static_cast<std::ostringstream*>( &(std::ostringstream() << (i+1)))->str();
        std::string fn_in = setName + "/cloudShifted_" + num + ".pcd";
        pcl::io::loadPCDFile (fn_in, clouds[i]);
    }


    // construct kd-trees
    for (int p = 0; p < numPairs; p++)
    {
        kdtree[p] = new pcl::KdTreeFLANN<pcl::PointXYZRGB>;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr ref(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::copyPointCloud<pcl::PointXYZRGB>(clouds[pairs[p][0]], *ref);
        kdtree[p]->setInputCloud (ref);
    }

     // compute the mean of the reference point cloud
     Eigen::Vector4f centroid; //ignore 4th entry
     pcl::compute3DCentroid(clouds[0], centroid);

     // variable to store parameters
     Eigen::VectorXf params(n);

     //----------------------------------------------------------------------------------------
     /* if (!optimize)
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
     } */
     //------------------------------------------------------------------------------------------------


     if (optimize)
     {
         // initialize parameters
         for (int i = 0; i < n; i++)
             params[i] = pRange[i][0];

         // optimize parameters
         float cost = simuatedAnnealing(params, clouds, centroid);
         std::cout << "final result:" << std::endl;
         std::cout << params << std::endl;
         std::cout << "value of cost function: " << cost << std::endl;
         std::cout << "angle of rotation: " << params[4] * 360 / (2*M_PI) << std::endl;
     }

     //---------------------------------------------------------------------------------------------------

     // rotate all point clouds using the final parameters and save result to pcd file
     pcl::PointCloud<pcl::PointXYZRGB> result;
     pcl::copyPointCloud<pcl::PointXYZRGB>(clouds[0], result);
     pcl::PointCloud<pcl::PointXYZRGB> clouds_rot[numClouds];
     rotate(clouds, clouds_rot, centroid, params);

     for (int i = 1; i < numClouds; i++)
         result += clouds_rot[i];

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
    float a2 = params[4] / sqrt(1 + params[2]*params[2] + params[3]*params[3]);
    float a1 = a2 * params[2];
    float a3 = a2 * params[3];
    float axis_angle[3] = {a1, a2, a3};
    float rotation_matrix[9];
    pcl::recognition::aux::axisAngleToRotationMatrix(axis_angle, rotation_matrix);
    Eigen::Matrix< float, 4, 4 > transform;
    transform.row(0) << rotation_matrix[0] , rotation_matrix[1] ,rotation_matrix[2] , 0;
    transform.row(1) << rotation_matrix[3] , rotation_matrix[4] ,rotation_matrix[5] , 0;
    transform.row(2) << rotation_matrix[6] , rotation_matrix[7] ,rotation_matrix[8] , 0;
    transform.row(3) << 0, 0, 0, 1;

    // transform the clouds
    pcl::PointCloud<pcl::PointXYZRGB> cloud_temp1;
    pcl::PointCloud<pcl::PointXYZRGB> cloud_temp2;
    pcl::PointCloud<pcl::PointXYZRGB> clouds_rot[numPairs];

    for (int i = 0; i < numPairs; i++)
    {
        pcl::demeanPointCloud(clouds[pairs[i][1]], fix, cloud_temp1);
        for (int j = 1; j <= (pairs[i][1]-pairs[i][0]) + (pairs[i][1]-pairs[i][0] < 0) * numClouds + numClouds; j++)
        {
            pcl::copyPointCloud<pcl::PointXYZRGB>(cloud_temp1, cloud_temp2);
            pcl::transformPointCloud(cloud_temp2, cloud_temp1, transform);
        }
        pcl::demeanPointCloud(cloud_temp1, fix_neg, clouds_rot[i]);
    }

    // compute cost function
    float cost = 0;
    int K = 1;         // for k-nearest neighbors
    float normConst = sqrt(3) * 255.0 / p1;  //normalization constant, RGB distances should be comparable to XYZ distances


    pcl::PointXYZRGB neighbor;
    pcl::PointCloud<pcl::PointXYZRGB>::iterator it;
    for (int p = 0; p < numPairs; p++)
    {
        // use XYZ to find the nearest neighbor
        //pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;
        //pcl::PointCloud<pcl::PointXYZRGB>::Ptr ref(new pcl::PointCloud<pcl::PointXYZRGB>);
        //pcl::copyPointCloud<pcl::PointXYZRGB>(clouds[pairs[p][0]], *ref);
        //kdtree.setInputCloud (ref);

        // iterate over second point cloud
        for (it = clouds_rot[p].begin(); it < clouds_rot[p].end(); it++)
        {
            pcl::PointXYZRGB searchPoint = *it;
            std::vector<int> pointIdxNKNSearch(K);
            std::vector<float> pointNKNSquaredDistance(K);
            kdtree[p]->nearestKSearch (searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance);
            neighbor = clouds[pairs[p][0]].points[pointIdxNKNSearch[0]];
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
            double cdist = norm(rgb1-rgb2) / normConst;    //is the euclidean metric a sensible choice?

            // update cost
            if (dist < p1)
                cost = cost + dist - p1 + cWeight * cdist;
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

    // temperature
    float T = T0;

    // some variables
    Eigen::VectorXf params_best(n);  // best parameter values found
    Eigen::VectorXf neighbor(n); // new candidate state
    float f_old, f_new, f_best;  // old, current and best value of the cost function
    f_best = f_old = cost(params, clouds, centroid);
    int it = 1;  //number of iterations
    float dE;
    float f_before[it_interval+1];

    do
    {
        std::cout << "aktuelle Kosten: " << f_old << std::endl;
        f_before[(it-1) % (it_interval+1)] = f_old;


        // draw new candidate state and compute corresponding value of the cost function
        sample_gaussian(params, neighbor, it, &sampler);
        f_new = cost(neighbor, clouds, centroid);
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




// rotate all point clouds using the parameters in params
void rotate( const pcl::PointCloud<pcl::PointXYZRGB> *clouds, pcl::PointCloud<pcl::PointXYZRGB> * clouds_rot, const Eigen::VectorXf & centroid,
             const Eigen::VectorXf & params)
{
    // position of the axis of rotation, relative to computed centroid
    Eigen::Vector4f offset(params[0], 0, params[1], 0);
    Eigen::Vector4f fix = centroid + offset;
    Eigen::Vector4f fix_neg = -fix;

    // specify axis of rotation (magnitude = angle in radians) and construct transformation matrix
    float a2 = params[4] / sqrt(1 + params[2]*params[2] + params[3]*params[3]);
    float a1 = a2 * params[2];
    float a3 = a2 * params[3];
    float axis_angle[3] = {a1, a2, a3};
    float rotation_matrix[9];
    pcl::recognition::aux::axisAngleToRotationMatrix(axis_angle, rotation_matrix);
    Eigen::Matrix< float, 4, 4 > transform;
    transform.row(0) << rotation_matrix[0] , rotation_matrix[1] ,rotation_matrix[2] , 0;
    transform.row(1) << rotation_matrix[3] , rotation_matrix[4] ,rotation_matrix[5] , 0;
    transform.row(2) << rotation_matrix[6] , rotation_matrix[7] ,rotation_matrix[8] , 0;
    transform.row(3) << 0, 0, 0, 1;

    // transform the clouds
    pcl::PointCloud<pcl::PointXYZRGB> cloud_temp1;
    pcl::PointCloud<pcl::PointXYZRGB> cloud_temp2;

    for (int i = 0; i < numClouds; i++)
    {
        pcl::demeanPointCloud(clouds[i], fix, cloud_temp1);
        for (int j = 1; j <= i; j++)
        {
            pcl::copyPointCloud<pcl::PointXYZRGB>(cloud_temp1, cloud_temp2);
            pcl::transformPointCloud(cloud_temp2, cloud_temp1, transform);
        }
        pcl::demeanPointCloud(cloud_temp1, fix_neg, clouds_rot[i]);
    }

}







// other functions tested

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


















