/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2010, Willow Garage, Inc.
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
 * $Id: test_mls.cpp 36039 2011-02-17 20:56:28Z marton $
 *
 */

#include <boost/make_shared.hpp>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/mls.h>
#include <pcl/surface/gp3.h>
#include <pcl/surface/grid_projection.h>

using namespace pcl;
using namespace pcl::io;
using namespace std;

typedef KdTree<PointXYZ>::Ptr KdTreePtr;

PointCloud<PointXYZ>::Ptr cloud (new PointCloud<PointXYZ> ());
vector<int> indices;
KdTreePtr tree;

/* ---[ */
int
  main (int argc, char** argv)
{
  //*
  // Input
  sensor_msgs::PointCloud2 cloud_blob;
  loadPCDFile ("./test/bun0.pcd", cloud_blob);
  fromROSMsg (cloud_blob, *cloud);

  // Indices
  indices.resize (cloud->points.size ());
  for (size_t i = 0; i < indices.size (); ++i) { indices[i] = i; }

  // KD-Tree
  tree = boost::make_shared<KdTreeFLANN<PointXYZ> > ();
  tree->setInputCloud (cloud);

  // Init objects
  PointCloud<PointXYZ> mls_points;
  PointCloud<Normal>::Ptr mls_normals (new PointCloud<Normal> ());
  MovingLeastSquares<PointXYZ, Normal> mls;

  // Set parameters
  mls.setInputCloud (cloud);
  mls.setIndices (boost::make_shared <vector<int> > (indices));
  mls.setPolynomialFit (true);
  mls.setSearchMethod (tree);
  mls.setSearchRadius (0.03);

  // Reconstruct
  //ros::Time::init();
  //ros::Time t = ros::Time::now ();
  mls.setOutputNormals (mls_normals);
  mls.reconstruct (mls_points);
  //ROS_INFO ("MLS fit done in %g seconds.", (ros::Time::now () - t).toSec ());
  //ROS_INFO ("MLS fit of %zu points and %zu normals.", mls_points.points.size (), mls_normals->points.size ());
  
  PointCloud<PointNormal> mls_cloud;
  pcl::concatenateFields (mls_points, *mls_normals, mls_cloud);

  // Save output
  savePCDFile ("./test/bun0-mls.pcd", mls_cloud);

//*/
//  sensor_msgs::PointCloud2 cloud_blob;
//  PointCloud<PointNormal> mls_points;
//  loadPCDFile ("./test/bun0-mls.pcd", cloud_blob);
//  fromROSMsg (cloud_blob, mls_points);
//  indices.resize (mls_points.points.size ());
//  for (size_t i = 0; i < indices.size (); ++i) { indices[i] = i; }

  // Test triangulation
  std::cerr << "TESTING TRIANGULATION" << std::endl;
  KdTree<PointNormal>::Ptr tree2 = boost::make_shared<KdTreeFLANN<PointNormal> > ();
  tree2->setInputCloud (mls_cloud.makeShared ());
  PolygonMesh triangles;
  GreedyProjectionTriangulation<PointNormal> gp3;
/*  GridProjection<PointNormal> gp3;
    gp3.setInputCloud (mls_points.makeShared ());
    gp3.setSearchMethod (tree2);
    gp3.setResolution (0.01);
    gp3.setPaddingSize (3);*/
  gp3.setInputCloud (mls_cloud.makeShared ());
  gp3.setSearchMethod (tree2);
  gp3.setIndices (boost::make_shared <vector<int> > (indices));
  gp3.setSearchRadius (0.02);
  gp3.setMu (2.5);
  gp3.setMaximumNearestNeighbors (50);
  gp3.setMaximumSurfaceAgle(M_PI/4); // 45 degrees
  gp3.setMinimumAngle(M_PI/18); // 10 degrees
  gp3.setMaximumAngle(2*M_PI/3); // 120 degrees
  gp3.setNormalConsistency(false);
  gp3.reconstruct (triangles);
  saveVTKFile ("./test/bun0-mls.vtk", triangles);
  //std::cerr << "INPUT: ./test/bun0-mls.pcd" << std::endl;
  std::cerr << "OUTPUT: ./test/bun0-mls.vtk" << std::endl;

  // Finish
  return (0);
}
/* ]--- */
