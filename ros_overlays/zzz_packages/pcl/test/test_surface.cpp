/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2009, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * $Id: test_surface.cpp 36144 2011-02-22 05:59:19Z aaldoma $
 *
 */
/** \author Zoltan-Csaba Marton */

#include <gtest/gtest.h>

#include <boost/make_shared.hpp>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/mls.h>
#include <pcl/surface/gp3.h>
#include <pcl/surface/grid_projection.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/surface/concave_hull.h>
#include <pcl/common/common.h>

using namespace pcl;
using namespace pcl::io;
using namespace std;

typedef KdTree<PointXYZ>::Ptr KdTreePtr;

PointCloud<PointXYZ> cloud;
vector<int> indices;
KdTreePtr tree;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST (PCL, MovingLeastSquares)
{
  // Init objects
  PointCloud<PointXYZ> mls_points;
  PointCloud<Normal>::Ptr mls_normals (new PointCloud<Normal> ());
  MovingLeastSquares<PointXYZ, Normal> mls;

  // Set parameters
  mls.setInputCloud (cloud.makeShared ());
  mls.setOutputNormals (mls_normals);
  mls.setIndices (boost::make_shared <vector<int> > (indices));
  mls.setPolynomialFit (true);
  mls.setSearchMethod (tree);
  mls.setSearchRadius (0.03);

  // Reconstruct
  mls.reconstruct (mls_points);
  EXPECT_NEAR (mls_points.points[0].x, 0.005, 1e-3);
  EXPECT_NEAR (mls_points.points[0].y, 0.111, 1e-3);
  EXPECT_NEAR (mls_points.points[0].z, 0.038, 1e-3);
  EXPECT_NEAR (fabs (mls_normals->points[0].normal[0]), 0.1176, 1e-3);
  EXPECT_NEAR (fabs (mls_normals->points[0].normal[1]), 0.6193, 1e-3);
  EXPECT_NEAR (fabs (mls_normals->points[0].normal[2]), 0.7762, 1e-3);
  EXPECT_NEAR (mls_normals->points[0].curvature, 0.012, 1e-3);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST (PCL, GreedyProjectionTriangulation)
{
  // Create a simple test dataset
  PointCloud<PointNormal> cloud_with_normals;
  cloud_with_normals.height = 1;
  cloud_with_normals.is_dense = true;
  for (float x = 0; x <= 0.01; x += 0.01)
  {
    for (float y = 0; y <= 0.01; y += 0.01)
    {
      PointNormal p;
      p.x = x;
      p.y = y;
      p.z = 0;
      p.normal[0] = 0;
      p.normal[1] = 0;
      p.normal[2] = 1;
      p.curvature = 0;
      cloud_with_normals.points.push_back (p);
    }
  }
  cloud_with_normals.width = cloud_with_normals.points.size ();
  PointCloud<PointNormal>::ConstPtr cloud_ptr = cloud_with_normals.makeShared ();

  // Create search tree
  KdTree<PointNormal>::Ptr tree2 = boost::make_shared<KdTreeFLANN<PointNormal> > ();
  tree2->setInputCloud (cloud_ptr);

  // Init objects
  PolygonMesh triangles;
  GreedyProjectionTriangulation<PointNormal> gp3;

  // Set parameters
  gp3.setInputCloud (cloud_ptr);
  gp3.setSearchMethod (tree2);
  gp3.setSearchRadius (0.02);
  gp3.setMu (2.5);
  gp3.setMaximumNearestNeighbors (50);
  gp3.setMaximumSurfaceAgle(M_PI/4); // 45 degrees
  gp3.setMinimumAngle(M_PI/18); // 10 degrees
  gp3.setMaximumAngle(2*M_PI/3); // 120 degrees
  gp3.setNormalConsistency(false);

  // Reconstruct
  gp3.reconstruct (triangles);
  EXPECT_EQ (triangles.cloud.width, cloud_with_normals.width);
  EXPECT_EQ (triangles.cloud.height, cloud_with_normals.height);
  EXPECT_EQ ((int)triangles.polygons.size(), 2);
  EXPECT_EQ ((int)triangles.polygons.at(0).vertices.size(), 3);
  EXPECT_EQ ((int)triangles.polygons.at(0).vertices.at(0), 0);
  EXPECT_EQ ((int)triangles.polygons.at(0).vertices.at(1), 1);
  EXPECT_EQ ((int)triangles.polygons.at(0).vertices.at(2), 2);

  // Additional vertex information
  std::vector<int> parts = gp3.getPartIDs();
  std::vector<int> states = gp3.getPointStates();
  int nr_points = cloud_with_normals.width * cloud_with_normals.height;
  EXPECT_EQ (parts.size (), nr_points);
  EXPECT_EQ (states.size (), nr_points);
  for (int i = 0; i < nr_points; i++)
  {
    EXPECT_EQ (parts[i], 0);
    EXPECT_EQ (states[i], gp3.BOUNDARY);
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST (PCL, GridProjection)
{
  // Create a simple test dataset
  PointCloud<PointNormal> cloud_with_normals;
  cloud_with_normals.height = 1;
  cloud_with_normals.is_dense = true;
  for (float x = 0; x <= 0.01; x += 0.01)
  {
    for (float y = 0; y <= 0.01; y += 0.01)
    {
      PointNormal p;
      p.x = x;
      p.y = y;
      p.z = 0;
      p.normal[0] = 0;
      p.normal[1] = 0;
      p.normal[2] = 1;
      p.curvature = 0;
      cloud_with_normals.points.push_back (p);
    }
  }
  cloud_with_normals.width = cloud_with_normals.points.size ();
  PointCloud<PointNormal>::ConstPtr cloud_ptr = cloud_with_normals.makeShared ();

  // Create search tree
  KdTree<PointNormal>::Ptr tree2 = boost::make_shared<KdTreeFLANN<PointNormal> > ();
  tree2->setInputCloud (cloud_ptr);

  // Init objects
  PolygonMesh triangles;
  GridProjection<PointNormal> gp3;

  // Set parameters
  gp3.setInputCloud (cloud_ptr);
  gp3.setSearchMethod (tree2);
  gp3.setResolution (0.1);

  // Reconstruct
  gp3.reconstruct (triangles);
//  EXPECT_EQ (triangles.cloud.width, cloud_with_normals.width);
//  EXPECT_EQ (triangles.cloud.height, cloud_with_normals.height);
//  EXPECT_EQ ((int)triangles.polygons.size(), 2);
///  EXPECT_EQ ((int)triangles.polygons.at(0).vertices.size(), 3);
//  EXPECT_EQ ((int)triangles.polygons.at(0).vertices.at(0), 0);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST (PCL, ConvexHull_bunny)
{
  pcl::PointCloud<pcl::PointXYZ> hull;
  std::vector<pcl::Vertices> polygons;

  pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloudptr (new pcl::PointCloud<pcl::PointXYZ> (cloud));
  pcl::ConvexHull<pcl::PointXYZ> chull;
  chull.setInputCloud (cloudptr);
  chull.reconstruct (hull, polygons);

  EXPECT_EQ (polygons.size (), 206);

  //check distance between min and max in the hull
  Eigen::Vector4f min_pt_hull, max_pt_hull;
  pcl::getMinMax3D (hull, min_pt_hull, max_pt_hull);

  Eigen::Vector4f min_pt, max_pt;
  pcl::getMinMax3D (hull, min_pt, max_pt);

  EXPECT_EQ((min_pt - max_pt).norm (), (min_pt_hull - max_pt_hull).norm ());
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST (PCL, ConvexHull_LTable)
{
  //construct dataset
  pcl::PointCloud<pcl::PointXYZ> cloud_out_ltable;
  cloud_out_ltable.points.resize (100);

  int npoints = 0;
  for (size_t i = 0; i < 8; i++) 
  {
    for (size_t j = 0; j <= 2; j++) 
    {
      cloud_out_ltable.points[npoints].x = (double)(i)*0.5;
      cloud_out_ltable.points[npoints].y = -(double)(j)*0.5;
      cloud_out_ltable.points[npoints].z = 0;
      npoints++;
    }
  }

  for (size_t i = 0; i <= 2; i++) 
  {
    for (size_t j = 3; j < 8; j++) 
    {
      cloud_out_ltable.points[npoints].x = (double)(i)*0.5;
      cloud_out_ltable.points[npoints].y = -(double)(j)*0.5;
      cloud_out_ltable.points[npoints].z = 0;
      npoints++;
    }
  }

  cloud_out_ltable.points.resize (npoints);

  pcl::PointCloud<pcl::PointXYZ> hull;
  std::vector<pcl::Vertices> polygons;
  pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloudptr (new pcl::PointCloud<pcl::PointXYZ> (cloud_out_ltable));
  pcl::ConvexHull<pcl::PointXYZ> chull;
  chull.setInputCloud (cloudptr);
  chull.reconstruct (hull, polygons);

  EXPECT_EQ (polygons.size (), 1);
  EXPECT_EQ (hull.points.size (), 11);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST (PCL, ConcaveHull_LTable)
{
  //construct dataset
  pcl::PointCloud<pcl::PointXYZ> cloud_out_ltable;
  cloud_out_ltable.points.resize (100);

  int npoints = 0;
  for (size_t i = 0; i < 8; i++) 
  {
    for (size_t j = 0; j <= 2; j++) 
    {
      cloud_out_ltable.points[npoints].x = (double)(i)*0.5;
      cloud_out_ltable.points[npoints].y = -(double)(j)*0.5;
      cloud_out_ltable.points[npoints].z = 0;
      npoints++;
    }
  }

  for (size_t i = 0; i <= 2; i++) 
  {
    for(size_t j = 3; j < 8; j++) 
    {
      cloud_out_ltable.points[npoints].x = (double)(i)*0.5;
      cloud_out_ltable.points[npoints].y = -(double)(j)*0.5;
      cloud_out_ltable.points[npoints].z = 0;
      npoints++;
    }
  }

  cloud_out_ltable.points.resize (npoints);

  pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloudptr (new pcl::PointCloud<pcl::PointXYZ> (cloud_out_ltable));

  pcl::PointCloud<pcl::PointXYZ> alpha_shape, voronoi_centers;
  std::vector<pcl::Vertices> polygons_alpha;

  pcl::ConcaveHull<pcl::PointXYZ> concave_hull;
  concave_hull.setInputCloud (cloudptr);
  concave_hull.setAlpha (0.5);
  concave_hull.setVoronoiCenters (voronoi_centers);
  concave_hull.reconstruct (alpha_shape, polygons_alpha);

  EXPECT_EQ (alpha_shape.points.size (), 27);

  pcl::PointCloud<pcl::PointXYZ> alpha_shape1, voronoi_centers1;
  std::vector<pcl::Vertices> polygons_alpha1;

  pcl::ConcaveHull<pcl::PointXYZ> concave_hull1;
  concave_hull1.setInputCloud (cloudptr);
  concave_hull1.setAlpha (1.5);
  concave_hull1.setVoronoiCenters (voronoi_centers1);
  concave_hull1.reconstruct (alpha_shape1, polygons_alpha1);

  EXPECT_EQ (alpha_shape1.points.size (), 23);

  pcl::PointCloud<pcl::PointXYZ> alpha_shape2, voronoi_centers2;
  std::vector<pcl::Vertices> polygons_alpha2;
  pcl::ConcaveHull<pcl::PointXYZ> concave_hull2;
  concave_hull2.setInputCloud (cloudptr);
  concave_hull2.setAlpha (3);
  concave_hull2.setVoronoiCenters (voronoi_centers2);
  concave_hull2.reconstruct (alpha_shape2, polygons_alpha2);

  EXPECT_EQ (alpha_shape2.points.size (), 19);
}

/* ---[ */
int
  main (int argc, char** argv)
{
  sensor_msgs::PointCloud2 cloud_blob;
  loadPCDFile ("./test/bun0.pcd", cloud_blob);
  fromROSMsg (cloud_blob, cloud);

  indices.resize (cloud.points.size ());
  for (size_t i = 0; i < indices.size (); ++i) { indices[i] = i; }

  tree = boost::make_shared<KdTreeFLANN<PointXYZ> > (false);
  tree->setInputCloud (cloud.makeShared ());

  testing::InitGoogleTest (&argc, argv);
  return (RUN_ALL_TESTS ());
}
/* ]--- */
