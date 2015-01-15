#include <pcl/io/vtk_io.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>


// parameters
int KSearch= 20;                      // 20
double searchRadius = 0.025;          // 0.025
double mu = 2.5;                      // 2.5
int maximumNearestNeighbors = 100;   // 100
double maximumSurfaceAngle = M_PI/4;  // M_PI/4
double minimumAngle = M_PI/18;        // M_PI/18
double maximumAngle = 2*M_PI/3;       // 2*M_PI/3
bool normalConsistency = false;       // false



int main (int argc, char** argv)
{
    // no RGB data available or ignoring available RGB data
    if (argc < 3 || std::string(argv[2]) == "xyz")
    {
        // Load input file into a PointCloud<T> with an appropriate type
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PCLPointCloud2 cloud_blob;
        std::string fn_in =argv[1];
        pcl::io::loadPCDFile (fn_in, cloud_blob);
        pcl::fromPCLPointCloud2 (cloud_blob, *cloud);
        //* the data should be available in cloud

        // Normal estimation*
        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;
        pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
        tree->setInputCloud (cloud);
        n.setInputCloud (cloud);
        n.setSearchMethod (tree);
        n.setKSearch (KSearch);
        n.compute (*normals);
        //* normals should not contain the point normals + surface curvatures

        // Concatenate the XYZ and normal fields*
        pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::PointNormal>);
        pcl::concatenateFields (*cloud, *normals, *cloud_with_normals);
        //* cloud_with_normals = cloud + normals

        // Create search tree*
        pcl::search::KdTree<pcl::PointNormal>::Ptr tree2 (new pcl::search::KdTree<pcl::PointNormal>);
        tree2->setInputCloud (cloud_with_normals);

        // Initialize objects
        pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;
        pcl::PolygonMesh triangles;

        // Set the maximum distance between connected points (maximum edge length)
        gp3.setSearchRadius (searchRadius);

        // Set typical values for the parameters
        gp3.setMu (mu);
        gp3.setMaximumNearestNeighbors (maximumNearestNeighbors);
        gp3.setMaximumSurfaceAngle(maximumSurfaceAngle);
        gp3.setMinimumAngle(minimumAngle);
        gp3.setMaximumAngle(maximumAngle);
        gp3.setNormalConsistency(normalConsistency);

        // Get result
        gp3.setInputCloud (cloud_with_normals);
        gp3.setSearchMethod (tree2);
        gp3.reconstruct (triangles);

        // save result into VTK file
        std::string fn_out = fn_in.substr(0, fn_in.size() - 4) + "_mesh.vtk";
        pcl::io::saveVTKFile (fn_out, triangles);
    }

    // using available RGB data
    else if (std::string(argv[2]) == "rgb")
    {
        // Load input file into a PointCloud<T> with an appropriate type
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::PCLPointCloud2 cloud_blob;
        std::string fn_in =argv[1];
        pcl::io::loadPCDFile (fn_in, cloud_blob);
        pcl::fromPCLPointCloud2 (cloud_blob, *cloud);
        //* the data should be available in cloud

        // Normal estimation*
        pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> n;
        pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
        pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);
        tree->setInputCloud (cloud);
        n.setInputCloud (cloud);
        n.setSearchMethod (tree);
        n.setKSearch (KSearch);
        n.compute (*normals);
        //* normals should not contain the point normals + surface curvatures

        // Concatenate the XYZRGB and normal fields*
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        pcl::concatenateFields (*cloud, *normals, *cloud_with_normals);
        //* cloud_with_normals = cloud + normals

        // Create search tree*
        pcl::search::KdTree<pcl::PointXYZRGBNormal>::Ptr tree2 (new pcl::search::KdTree<pcl::PointXYZRGBNormal>);
        tree2->setInputCloud (cloud_with_normals);

        // Initialize objects
        pcl::GreedyProjectionTriangulation<pcl::PointXYZRGBNormal> gp3;
        pcl::PolygonMesh triangles;

        // Set the maximum distance between connected points (maximum edge length)
        gp3.setSearchRadius (searchRadius);

        // Set typical values for the parameters
        gp3.setMu (mu);
        gp3.setMaximumNearestNeighbors (maximumNearestNeighbors);
        gp3.setMaximumSurfaceAngle(maximumSurfaceAngle);
        gp3.setMinimumAngle(minimumAngle);
        gp3.setMaximumAngle(maximumAngle);
        gp3.setNormalConsistency(normalConsistency);

        // Get result
        gp3.setInputCloud (cloud_with_normals);
        gp3.setSearchMethod (tree2);
        gp3.reconstruct (triangles);

        // save result into VTK file
        std::string fn_out = fn_in.substr(0, fn_in.size() - 4) + "_mesh.vtk";
        pcl::io::saveVTKFile (fn_out, triangles);
    }

  // Finish
  return (0);
}
