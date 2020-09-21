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
 * $Id: test_filters.cpp 35810 2011-02-08 00:03:46Z rusu $
 *
 */
/** \author Radu Bogdan Rusu */

#include <gtest/gtest.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/conditional_removal.h>

using namespace pcl;
using namespace pcl::io;
using namespace std;

sensor_msgs::PointCloud2 cloud_blob_;
sensor_msgs::PointCloud2::Ptr cloud_blob_ptr_;
PointCloud<PointXYZ> cloud_;
PointCloud<PointXYZ>::Ptr cloud_ptr_;
vector<int> indices_;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST (ExtractIndices, Filters)
{
  // Test the PointCloud<PointT> method
  ExtractIndices<PointXYZ> ei;
  vector<int> indices (2);
  indices[0] = 0; indices[1] = cloud_.points.size () - 1;

  PointCloud<PointXYZ> output;
  ei.setInputCloud (cloud_ptr_);
  ei.setIndices (boost::make_shared<std::vector<int> > (indices));
  ei.filter (output);

  EXPECT_EQ ((int)output.points.size (), 2);
  EXPECT_EQ ((int)output.width,  2);
  EXPECT_EQ ((int)output.height, 1);
 
  EXPECT_EQ (cloud_.points[0].x, output.points[0].x);
  EXPECT_EQ (cloud_.points[0].y, output.points[0].y);
  EXPECT_EQ (cloud_.points[0].z, output.points[0].z);

  EXPECT_EQ (cloud_.points[cloud_.points.size () - 1].x, output.points[1].x);
  EXPECT_EQ (cloud_.points[cloud_.points.size () - 1].y, output.points[1].y);
  EXPECT_EQ (cloud_.points[cloud_.points.size () - 1].z, output.points[1].z);

  ei.setNegative (true);
  ei.filter (output);

  EXPECT_EQ (output.points.size (), cloud_.points.size () - 2);
  EXPECT_EQ (output.width,  cloud_.points.size () - 2);
  EXPECT_EQ ((int)output.height, 1);
 
  EXPECT_EQ (cloud_.points[1].x, output.points[0].x);
  EXPECT_EQ (cloud_.points[1].y, output.points[0].y);
  EXPECT_EQ (cloud_.points[1].z, output.points[0].z);

  EXPECT_EQ (cloud_.points[cloud_.points.size () - 2].x, output.points[output.points.size () - 1].x);
  EXPECT_EQ (cloud_.points[cloud_.points.size () - 2].y, output.points[output.points.size () - 1].y);
  EXPECT_EQ (cloud_.points[cloud_.points.size () - 2].z, output.points[output.points.size () - 1].z);

  // Test the sensor_msgs::PointCloud2 method
  ExtractIndices<sensor_msgs::PointCloud2> ei2;

  sensor_msgs::PointCloud2 output_blob;
  ei2.setInputCloud (cloud_blob_ptr_);
  ei2.setIndices (boost::make_shared<std::vector<int> > (indices));
  ei2.filter (output_blob);

  fromROSMsg (output_blob, output);

  EXPECT_EQ ((int)output.points.size (), 2);
  EXPECT_EQ ((int)output.width,  2);
  EXPECT_EQ ((int)output.height, 1);

  EXPECT_EQ (cloud_.points[0].x, output.points[0].x);
  EXPECT_EQ (cloud_.points[0].y, output.points[0].y);
  EXPECT_EQ (cloud_.points[0].z, output.points[0].z);

  EXPECT_EQ (cloud_.points[cloud_.points.size () - 1].x, output.points[1].x);
  EXPECT_EQ (cloud_.points[cloud_.points.size () - 1].y, output.points[1].y);
  EXPECT_EQ (cloud_.points[cloud_.points.size () - 1].z, output.points[1].z);

  ei2.setNegative (true);
  ei2.filter (output_blob);

  fromROSMsg (output_blob, output);

  EXPECT_EQ (output.points.size (), cloud_.points.size () - 2);
  EXPECT_EQ (output.width,  cloud_.points.size () - 2);
  EXPECT_EQ ((int)output.height, 1);

  EXPECT_EQ (cloud_.points[1].x, output.points[0].x);
  EXPECT_EQ (cloud_.points[1].y, output.points[0].y);
  EXPECT_EQ (cloud_.points[1].z, output.points[0].z);

  EXPECT_EQ (cloud_.points[cloud_.points.size () - 2].x, output.points[output.points.size () - 1].x);
  EXPECT_EQ (cloud_.points[cloud_.points.size () - 2].y, output.points[output.points.size () - 1].y);
  EXPECT_EQ (cloud_.points[cloud_.points.size () - 2].z, output.points[output.points.size () - 1].z);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST (PassThrough, Filters)
{
  // Test the PointCloud<PointT> method
  PointCloud<PointXYZ> output;
  PassThrough<PointXYZ> pt;

  pt.setInputCloud (cloud_ptr_);
  pt.filter (output);

  EXPECT_EQ (output.points.size (), cloud_.points.size ());
  EXPECT_EQ (output.width,  cloud_.width);
  EXPECT_EQ (output.height, cloud_.height);

  pt.setFilterFieldName ("z");
  pt.setFilterLimits (0.05, 0.1);
  pt.filter (output);
  
  EXPECT_EQ ((int)output.points.size (), 42); 
  EXPECT_EQ ((int)output.width, 42);
  EXPECT_EQ ((int)output.height, 1);
  EXPECT_EQ ((bool)output.is_dense, true);

  EXPECT_NEAR (output.points[0].x, -0.074556, 1e-5);
  EXPECT_NEAR (output.points[0].y,  0.13415,  1e-5);
  EXPECT_NEAR (output.points[0].z,  0.051046, 1e-5);

  EXPECT_NEAR (output.points[41].x, -0.030331, 1e-5);
  EXPECT_NEAR (output.points[41].y,  0.039749, 1e-5);
  EXPECT_NEAR (output.points[41].z,  0.052133, 1e-5);
  
  pt.setFilterLimitsNegative (true);
  pt.filter (output);

  EXPECT_EQ ((int)output.points.size (), 355); 
  EXPECT_EQ ((int)output.width, 355);
  EXPECT_EQ ((int)output.height, 1);
  EXPECT_EQ ((bool)output.is_dense, true);

  EXPECT_NEAR (output.points[0].x, 0.0054216, 1e-5);
  EXPECT_NEAR (output.points[0].y, 0.11349,   1e-5);
  EXPECT_NEAR (output.points[0].z, 0.040749,  1e-5);

  EXPECT_NEAR (output.points[354].x, -0.07793, 1e-5);
  EXPECT_NEAR (output.points[354].y,  0.17516, 1e-5);
  EXPECT_NEAR (output.points[354].z, -0.0444,  1e-5);

  // Test the keep organized structure
  pt.setUserFilterValue (std::numeric_limits<float>::quiet_NaN ());
  pt.setFilterFieldName ("");
  pt.filter (output);

  EXPECT_EQ (output.points.size (), cloud_.points.size ()); 
  EXPECT_EQ (output.width, cloud_.width);
  EXPECT_EQ (output.height, cloud_.height);
  EXPECT_EQ (output.is_dense, cloud_.is_dense);
  EXPECT_NEAR (output.points[0].x, cloud_.points[0].x, 1e-5);
  EXPECT_NEAR (output.points[output.points.size () - 1].x, cloud_.points[cloud_.points.size () - 1].x, 1e-5);

  pt.setFilterFieldName ("z");
  pt.setFilterLimitsNegative (false);
  pt.setKeepOrganized (true);
  pt.filter (output);
  
  EXPECT_EQ (output.points.size (), cloud_.points.size ()); 
  EXPECT_EQ (output.width, cloud_.width);
  EXPECT_EQ (output.height, cloud_.height);
  EXPECT_EQ ((bool)output.is_dense, false);   // NaN was set as a user filter value

  if (!pcl_isnan (output.points[0].x)) EXPECT_EQ (1, 0);
  if (!pcl_isnan (output.points[0].y)) EXPECT_EQ (1, 0);
  if (!pcl_isnan (output.points[0].z)) EXPECT_EQ (1, 0);

  if (!pcl_isnan (output.points[41].x)) EXPECT_EQ (1, 0);
  if (!pcl_isnan (output.points[41].y)) EXPECT_EQ (1, 0);
  if (!pcl_isnan (output.points[41].z)) EXPECT_EQ (1, 0);

  pt.setFilterLimitsNegative (true);
  pt.filter (output);

  EXPECT_EQ (output.points.size (), cloud_.points.size ()); 
  EXPECT_EQ (output.width, cloud_.width);
  EXPECT_EQ (output.height, cloud_.height);
  EXPECT_EQ ((bool)output.is_dense, false);   // NaN was set as a user filter value

  EXPECT_NEAR (output.points[0].x, cloud_.points[0].x, 1e-5);
  EXPECT_NEAR (output.points[0].y, cloud_.points[0].y, 1e-5);
  EXPECT_NEAR (output.points[0].z, cloud_.points[0].z, 1e-5);

  EXPECT_NEAR (output.points[41].x, cloud_.points[41].x, 1e-5);
  EXPECT_NEAR (output.points[41].y, cloud_.points[41].y, 1e-5);
  EXPECT_NEAR (output.points[41].z, cloud_.points[41].z, 1e-5);


  // Test the sensor_msgs::PointCloud2 method
  PassThrough<sensor_msgs::PointCloud2> pt2;

  sensor_msgs::PointCloud2 output_blob;
  pt2.setInputCloud (cloud_blob_ptr_);
  pt2.filter (output_blob);

  fromROSMsg (output_blob, output);

  EXPECT_EQ (output.points.size (), cloud_.points.size ());
  EXPECT_EQ (output.width,  cloud_.width);
  EXPECT_EQ (output.height, cloud_.height);

  pt2.setFilterFieldName ("z");
  pt2.setFilterLimits (0.05, 0.1);
  pt2.filter (output_blob);
  
  fromROSMsg (output_blob, output);

  EXPECT_EQ ((int)output.points.size (), 42); 
  EXPECT_EQ ((int)output.width, 42);
  EXPECT_EQ ((int)output.height, 1);
  EXPECT_EQ ((bool)output.is_dense, true);

  EXPECT_NEAR (output.points[0].x, -0.074556, 1e-5);
  EXPECT_NEAR (output.points[0].y,  0.13415,  1e-5);
  EXPECT_NEAR (output.points[0].z,  0.051046, 1e-5);

  EXPECT_NEAR (output.points[41].x, -0.030331, 1e-5);
  EXPECT_NEAR (output.points[41].y,  0.039749, 1e-5);
  EXPECT_NEAR (output.points[41].z,  0.052133, 1e-5);
  
  pt2.setFilterLimitsNegative (true);
  pt2.filter (output_blob);
  
  fromROSMsg (output_blob, output);

  EXPECT_EQ ((int)output.points.size (), 355); 
  EXPECT_EQ ((int)output.width, 355);
  EXPECT_EQ ((int)output.height, 1);
  EXPECT_EQ ((bool)output.is_dense, true);

  EXPECT_NEAR (output.points[0].x, 0.0054216, 1e-5);
  EXPECT_NEAR (output.points[0].y, 0.11349,   1e-5);
  EXPECT_NEAR (output.points[0].z, 0.040749,  1e-5);

  EXPECT_NEAR (output.points[354].x, -0.07793, 1e-5);
  EXPECT_NEAR (output.points[354].y,  0.17516, 1e-5);
  EXPECT_NEAR (output.points[354].z, -0.0444,  1e-5);

  // Test the keep organized structure
  pt2.setUserFilterValue (std::numeric_limits<float>::quiet_NaN ());
  pt2.setFilterFieldName ("");
  pt2.filter (output_blob);
  fromROSMsg (output_blob, output);

  EXPECT_EQ (output.points.size (), cloud_.points.size ()); 
  EXPECT_EQ (output.width, cloud_.width);
  EXPECT_EQ (output.height, cloud_.height);
  EXPECT_EQ (output.is_dense, cloud_.is_dense);
  EXPECT_NEAR (output.points[0].x, cloud_.points[0].x, 1e-5);
  EXPECT_NEAR (output.points[output.points.size () - 1].x, cloud_.points[cloud_.points.size () - 1].x, 1e-5);

  pt2.setFilterFieldName ("z");
  pt2.setFilterLimitsNegative (false);
  pt2.setKeepOrganized (true);
  pt2.filter (output_blob);
  fromROSMsg (output_blob, output);
  
  EXPECT_EQ (output.points.size (), cloud_.points.size ()); 
  EXPECT_EQ (output.width, cloud_.width);
  EXPECT_EQ (output.height, cloud_.height);
  EXPECT_EQ ((bool)output.is_dense, false);   // NaN was set as a user filter value

  if (!pcl_isnan (output.points[0].x)) EXPECT_EQ (1, 0);
  if (!pcl_isnan (output.points[0].y)) EXPECT_EQ (1, 0);
  if (!pcl_isnan (output.points[0].z)) EXPECT_EQ (1, 0);

  if (!pcl_isnan (output.points[41].x)) EXPECT_EQ (1, 0);
  if (!pcl_isnan (output.points[41].y)) EXPECT_EQ (1, 0);
  if (!pcl_isnan (output.points[41].z)) EXPECT_EQ (1, 0);

  pt2.setFilterLimitsNegative (true);
  pt2.filter (output_blob);
  fromROSMsg (output_blob, output);

  EXPECT_EQ (output.points.size (), cloud_.points.size ()); 
  EXPECT_EQ (output.width, cloud_.width);
  EXPECT_EQ (output.height, cloud_.height);
  EXPECT_EQ ((bool)output.is_dense, false);   // NaN was set as a user filter value

  EXPECT_NEAR (output.points[0].x, cloud_.points[0].x, 1e-5);
  EXPECT_NEAR (output.points[0].y, cloud_.points[0].y, 1e-5);
  EXPECT_NEAR (output.points[0].z, cloud_.points[0].z, 1e-5);

  EXPECT_NEAR (output.points[41].x, cloud_.points[41].x, 1e-5);
  EXPECT_NEAR (output.points[41].y, cloud_.points[41].y, 1e-5);
  EXPECT_NEAR (output.points[41].z, cloud_.points[41].z, 1e-5);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST (VoxelGrid, Filters)
{
  // Test the PointCloud<PointT> method
  PointCloud<PointXYZ> output;
  VoxelGrid<PointXYZ> grid;

  grid.setLeafSize (0.02, 0.02, 0.02);
  grid.setInputCloud (cloud_ptr_);
  grid.filter (output);

  EXPECT_EQ ((int)output.points.size (), 103);
  EXPECT_EQ ((int)output.width,  103);
  EXPECT_EQ ((int)output.height, 1);
  EXPECT_EQ ((bool)output.is_dense, true);

  grid.setFilterFieldName ("z");
  grid.setFilterLimits (0.05, 0.1);
  grid.filter (output);
  
  EXPECT_EQ ((int)output.points.size (), 14);
  EXPECT_EQ ((int)output.width, 14);
  EXPECT_EQ ((int)output.height, 1);
  EXPECT_EQ ((bool)output.is_dense, true);

  EXPECT_NEAR (output.points[0].x, -0.026125, 1e-4);
  EXPECT_NEAR (output.points[0].y,  0.039788, 1e-4);
  EXPECT_NEAR (output.points[0].z,  0.052827, 1e-4);

  EXPECT_NEAR (output.points[13].x, -0.073202, 1e-4);
  EXPECT_NEAR (output.points[13].y,  0.1296,   1e-4);
  EXPECT_NEAR (output.points[13].z,  0.051333, 1e-4);
  
  grid.setFilterLimitsNegative (true);
  grid.setSaveLeafLayout(true);
  grid.filter (output);

  EXPECT_EQ ((int)output.points.size (), 100);
  EXPECT_EQ ((int)output.width, 100);
  EXPECT_EQ ((int)output.height, 1);
  EXPECT_EQ ((bool)output.is_dense, true);

  EXPECT_NEAR (output.points[0].x, -0.070192, 1e-4);
  EXPECT_NEAR (output.points[0].y,  0.17653,  1e-4);
  EXPECT_NEAR (output.points[0].z, -0.048774, 1e-4);

  EXPECT_NEAR (output.points[99].x, -0.068948, 1e-4);
  EXPECT_NEAR (output.points[99].y,  0.1447,   1e-4);
  EXPECT_NEAR (output.points[99].z,  0.042178, 1e-4);
  
  // centroids should be identified correctly
  EXPECT_EQ (grid.getCentroidIndex (output.points[0]), 0);
  EXPECT_EQ (grid.getCentroidIndex (output.points[99]), 99);
  EXPECT_EQ (grid.getCentroidIndexAt (grid.getGridCoordinates (-1,-1,-1), false), -1);
  //ROS_ERROR ("IGNORE PREVIOUS ERROR: testing it's functionality!");

  // input point 195 [0.04872199893, 0.07376000285, 0.01743399911]
  int centroidIdx = grid.getCentroidIndex (cloud_ptr_->points[195]);

  // for arbitrary points, the centroid should be close
  EXPECT_LE (fabs (output.points[centroidIdx].x - cloud_ptr_->points[195].x), 0.02);
  EXPECT_LE (fabs (output.points[centroidIdx].y - cloud_ptr_->points[195].y), 0.02);
  EXPECT_LE (fabs (output.points[centroidIdx].z - cloud_ptr_->points[195].z), 0.02);

  // if getNeighborCentroidIndices works then the other helper functions work as well
  EXPECT_EQ (grid.getNeighborCentroidIndices (output.points[0], Eigen::MatrixXi::Zero(3,1))[0], 0);
  EXPECT_EQ (grid.getNeighborCentroidIndices (output.points[99], Eigen::MatrixXi::Zero(3,1))[0], 99);

  // neighboring centroid should be in the right position
  Eigen::MatrixXi directions = Eigen::Vector3i (0, 0, 1);
  vector<int> neighbors = grid.getNeighborCentroidIndices (cloud_ptr_->points[195], directions);
  EXPECT_EQ (neighbors.size (), (size_t)directions.cols ());
  EXPECT_NE (neighbors.at (0), -1);
  EXPECT_LE (fabs (output.points[neighbors.at (0)].x - output.points[centroidIdx].x), 0.02);
  EXPECT_LE (fabs (output.points[neighbors.at (0)].y - output.points[centroidIdx].y), 0.02);
  EXPECT_LE (      output.points[neighbors.at (0)].z - output.points[centroidIdx].z,  0.02 * 2);

  // Test the sensor_msgs::PointCloud2 method
  VoxelGrid<sensor_msgs::PointCloud2> grid2;

  sensor_msgs::PointCloud2 output_blob;

  grid2.setLeafSize (0.02, 0.02, 0.02);
  grid2.setInputCloud (cloud_blob_ptr_);
  grid2.filter (output_blob);

  fromROSMsg (output_blob, output);

  EXPECT_EQ ((int)output.points.size (), 103);
  EXPECT_EQ ((int)output.width,  103);
  EXPECT_EQ ((int)output.height, 1);
  EXPECT_EQ ((bool)output.is_dense, true);

  grid2.setFilterFieldName ("z");
  grid2.setFilterLimits (0.05, 0.1);
  grid2.filter (output_blob);
  
  fromROSMsg (output_blob, output);

  EXPECT_EQ ((int)output.points.size (), 14);
  EXPECT_EQ ((int)output.width, 14);
  EXPECT_EQ ((int)output.height, 1);
  EXPECT_EQ ((bool)output.is_dense, true);

  EXPECT_NEAR (output.points[0].x, -0.026125, 1e-4);
  EXPECT_NEAR (output.points[0].y,  0.039788, 1e-4);
  EXPECT_NEAR (output.points[0].z,  0.052827, 1e-4);

  EXPECT_NEAR (output.points[13].x, -0.073202, 1e-4);
  EXPECT_NEAR (output.points[13].y,  0.1296,   1e-4);
  EXPECT_NEAR (output.points[13].z,  0.051333, 1e-4);
  
  grid2.setFilterLimitsNegative (true);
  grid2.setSaveLeafLayout(true);
  grid2.filter (output_blob);

  fromROSMsg (output_blob, output);

  EXPECT_EQ ((int)output.points.size (), 100);
  EXPECT_EQ ((int)output.width, 100);
  EXPECT_EQ ((int)output.height, 1);
  EXPECT_EQ ((bool)output.is_dense, true);

  EXPECT_NEAR (output.points[0].x, -0.070192, 1e-4);
  EXPECT_NEAR (output.points[0].y,  0.17653,  1e-4);
  EXPECT_NEAR (output.points[0].z, -0.048774, 1e-4);

  EXPECT_NEAR (output.points[99].x, -0.068948, 1e-4);
  EXPECT_NEAR (output.points[99].y,  0.1447,   1e-4);
  EXPECT_NEAR (output.points[99].z,  0.042178, 1e-4);

  // centroids should be identified correctly
  EXPECT_EQ (grid2.getCentroidIndex (output.points[0].x, output.points[0].y, output.points[0].z), 0);
  EXPECT_EQ (grid2.getCentroidIndex (output.points[99].x, output.points[99].y, output.points[99].z), 99);
  EXPECT_EQ (grid2.getCentroidIndexAt (grid2.getGridCoordinates (-1,-1,-1), false), -1);
  //ROS_ERROR ("IGNORE PREVIOUS ERROR: testing it's functionality!");

  // input point 195 [0.04872199893, 0.07376000285, 0.01743399911]
  int centroidIdx2 = grid2.getCentroidIndex (0.048722, 0.073760, 0.017434);
  EXPECT_NE (centroidIdx2, -1);

  // for arbitrary points, the centroid should be close
  EXPECT_LE (fabs (output.points[centroidIdx2].x - 0.048722), 0.02);
  EXPECT_LE (fabs (output.points[centroidIdx2].y - 0.073760), 0.02);
  EXPECT_LE (fabs (output.points[centroidIdx2].z - 0.017434), 0.02);

  // if getNeighborCentroidIndices works then the other helper functions work as well
  EXPECT_EQ (grid2.getNeighborCentroidIndices (output.points[0].x, output.points[0].y, output.points[0].z, Eigen::MatrixXi::Zero(3,1))[0], 0);
  EXPECT_EQ (grid2.getNeighborCentroidIndices (output.points[99].x, output.points[99].y, output.points[99].z, Eigen::MatrixXi::Zero(3,1))[0], 99);

  // neighboring centroid should be in the right position
  Eigen::MatrixXi directions2 = Eigen::Vector3i (0, 0, 1);
  vector<int> neighbors2 = grid2.getNeighborCentroidIndices (0.048722, 0.073760, 0.017434, directions2);
  EXPECT_EQ (neighbors2.size (), (size_t)directions2.cols ());
  EXPECT_NE (neighbors2.at (0), -1);
  EXPECT_LE (fabs (output.points[neighbors2.at (0)].x - output.points[centroidIdx2].x), 0.02);
  EXPECT_LE (fabs (output.points[neighbors2.at (0)].y - output.points[centroidIdx2].y), 0.02);
  EXPECT_LE (      output.points[neighbors2.at (0)].z - output.points[centroidIdx2].z,  0.02 * 2);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST (VoxelGrid_RGB, Filters)
{
  sensor_msgs::PointCloud2 cloud_rgb_blob_;
  sensor_msgs::PointCloud2::Ptr cloud_rgb_blob_ptr_;
  PointCloud<PointXYZRGB> cloud_rgb_;
  PointCloud<PointXYZRGB>::Ptr cloud_rgb_ptr_;

  int col_r[] = {214,193, 180, 164, 133, 119, 158, 179, 178, 212};
  int col_g[] = { 10, 39, 219, 231, 142, 169,  84, 158, 139, 214};
  int col_b[] = {101, 26,  46, 189, 211, 154, 246,  16, 139, 153};
  float ave_r = 0.0; float ave_b = 0.0; float ave_g = 0.0;
  for (int i = 0; i < 10; ++i)
  {
    ave_r += col_r[i]; ave_g += col_g[i]; ave_b += col_b[i];
  }
  ave_r /= 10; ave_g /= 10; ave_b /= 10;

  for (int i=0; i < 10; ++i)
  {
    PointXYZRGB pt;
    int rgb = (col_r[i] << 16) | (col_g[i]  << 8) | col_b[i];
    pt.x = 0.0; pt.y = 0.0; pt.z = 0.0;
    pt.rgb = *reinterpret_cast<float*>(&rgb);
    cloud_rgb_.points.push_back(pt);
  }

  toROSMsg (cloud_rgb_, cloud_rgb_blob_);
  cloud_rgb_blob_ptr_.reset (new sensor_msgs::PointCloud2 (cloud_rgb_blob_));
  cloud_rgb_ptr_.reset (new PointCloud<PointXYZRGB> (cloud_rgb_));

  PointCloud<PointXYZRGB> output_rgb;
  VoxelGrid<PointXYZRGB> grid_rgb;

  grid_rgb.setLeafSize (0.03, 0.03, 0.03);
  grid_rgb.setInputCloud (cloud_rgb_ptr_);
  grid_rgb.filter (output_rgb);

  EXPECT_EQ ((int)output_rgb.points.size (), 1);
  EXPECT_EQ ((int)output_rgb.width, 1);
  EXPECT_EQ ((int)output_rgb.height, 1);
  EXPECT_EQ ((bool)output_rgb.is_dense, true);
  {
    int rgb;
    int r,g,b;
    memcpy (&rgb, &(output_rgb.points[0].rgb), sizeof(int));
    r = (rgb >> 16) & 0xFF; g = (rgb >> 8 ) & 0xFF; b = (rgb >> 0 ) & 0xFF;
    EXPECT_NEAR (output_rgb.points[0].x, 0.0, 1e-4);
    EXPECT_NEAR (output_rgb.points[0].y, 0.0, 1e-4);
    EXPECT_NEAR (output_rgb.points[0].z, 0.0, 1e-4);
    EXPECT_NEAR (r, ave_r, 1.0);
    EXPECT_NEAR (g, ave_g, 1.0);
    EXPECT_NEAR (b, ave_b, 1.0);
  }

  VoxelGrid<sensor_msgs::PointCloud2> grid2;
  sensor_msgs::PointCloud2 output_rgb_blob;

  grid2.setLeafSize (0.03, 0.03, 0.03);
  grid2.setInputCloud (cloud_rgb_blob_ptr_);
  grid2.filter (output_rgb_blob);

  fromROSMsg (output_rgb_blob, output_rgb);

  EXPECT_EQ ((int)output_rgb.points.size (), 1);
  EXPECT_EQ ((int)output_rgb.width, 1);
  EXPECT_EQ ((int)output_rgb.height, 1);
  EXPECT_EQ ((bool)output_rgb.is_dense, true);
  {
    int rgb;
    int r,g,b;
    memcpy (&rgb, &(output_rgb.points[0].rgb), sizeof(int));
    r = (rgb >> 16) & 0xFF; g = (rgb >> 8 ) & 0xFF; b = (rgb >> 0 ) & 0xFF;
    EXPECT_NEAR (output_rgb.points[0].x, 0.0, 1e-4);
    EXPECT_NEAR (output_rgb.points[0].y, 0.0, 1e-4);
    EXPECT_NEAR (output_rgb.points[0].z, 0.0, 1e-4);
    EXPECT_NEAR (r, ave_r, 1.0);
    EXPECT_NEAR (g, ave_g, 1.0);
    EXPECT_NEAR (b, ave_b, 1.0);
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST (ProjectInliers, Filters)
{
  // Test the PointCloud<PointT> method
  ProjectInliers<PointXYZ> proj;
  PointCloud<PointXYZ> output;

  ModelCoefficients::Ptr coefficients (new ModelCoefficients ());
  coefficients->values.resize (4);
  coefficients->values[0] = coefficients->values[1] = 0;
  coefficients->values[2] = 1.0;
  coefficients->values[3] = 0;

  proj.setModelType (SACMODEL_PLANE);
  proj.setInputCloud (cloud_ptr_);
  proj.setModelCoefficients (coefficients);
  proj.filter (output);

  for (size_t i = 0; i < output.points.size (); ++i)
    EXPECT_NEAR (output.points[i].z, 0.0, 1e-4);
  
  // Test the sensor_msgs::PointCloud2 method
  ProjectInliers<sensor_msgs::PointCloud2> proj2;

  sensor_msgs::PointCloud2 output_blob;

  proj2.setModelType (SACMODEL_PLANE);
  proj2.setInputCloud (cloud_blob_ptr_);
  proj2.setModelCoefficients (coefficients);
  proj2.filter (output_blob);

  fromROSMsg (output_blob, output);

  for (size_t i = 0; i < output.points.size (); ++i)
    EXPECT_NEAR (output.points[i].z, 0.0, 1e-4);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST (RadiusOutlierRemoval, Filters)
{
  // Test the PointCloud<PointT> method
  PointCloud<PointXYZ> cloud_out;
  // Remove outliers using a spherical density criterion
  RadiusOutlierRemoval<PointXYZ> outrem;
  outrem.setInputCloud (cloud_ptr_);
  outrem.setRadiusSearch (0.02);
  outrem.setMinNeighborsInRadius (15);
  outrem.filter (cloud_out);
  
  EXPECT_EQ ((int)cloud_out.points.size (), 307);
  EXPECT_EQ ((int)cloud_out.width, 307);
  EXPECT_EQ ((bool)cloud_out.is_dense, true);
  EXPECT_NEAR (cloud_out.points[cloud_out.points.size () - 1].x, -0.077893, 1e-4);
  EXPECT_NEAR (cloud_out.points[cloud_out.points.size () - 1].y,  0.16039,  1e-4);
  EXPECT_NEAR (cloud_out.points[cloud_out.points.size () - 1].z, -0.021299, 1e-4);

  // Test the sensor_msgs::PointCloud2 method
  sensor_msgs::PointCloud2 cloud_out2;
  RadiusOutlierRemoval<sensor_msgs::PointCloud2> outrem2;
  outrem2.setInputCloud (cloud_blob_ptr_);
  outrem2.setRadiusSearch (0.02);
  outrem2.setMinNeighborsInRadius (15);
  outrem2.filter (cloud_out2);

  fromROSMsg (cloud_out2, cloud_out);
  EXPECT_EQ ((int)cloud_out.points.size (), 307);
  EXPECT_EQ ((int)cloud_out.width, 307);
  EXPECT_EQ ((bool)cloud_out.is_dense, true);
  EXPECT_NEAR (cloud_out.points[cloud_out.points.size () - 1].x, -0.077893, 1e-4);
  EXPECT_NEAR (cloud_out.points[cloud_out.points.size () - 1].y,  0.16039,  1e-4);
  EXPECT_NEAR (cloud_out.points[cloud_out.points.size () - 1].z, -0.021299, 1e-4);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST (StatisticalOutlierRemoval, Filters)
{
  // Test the PointCloud<PointT> method
  PointCloud<PointXYZ> output;
  // Remove outliers using a spherical density criterion
  StatisticalOutlierRemoval<PointXYZ> outrem;
  outrem.setInputCloud (cloud_ptr_);
  outrem.setMeanK (50);
  outrem.setStddevMulThresh (1.0);
  outrem.filter (output);

  EXPECT_EQ ((int)output.points.size (), 352);
  EXPECT_EQ ((int)output.width, 352);
  EXPECT_EQ ((bool)output.is_dense, true);
  EXPECT_NEAR (output.points[output.points.size () - 1].x, -0.034667,   1e-4);
  EXPECT_NEAR (output.points[output.points.size () - 1].y,  0.15131,    1e-4);
  EXPECT_NEAR (output.points[output.points.size () - 1].z, -0.00071029, 1e-4);

  outrem.setNegative (true);
  outrem.filter (output);

  EXPECT_EQ ((int)output.points.size (), (int)cloud_.points.size () - 352);
  EXPECT_EQ ((int)output.width, (int)cloud_.width - 352);
  EXPECT_EQ ((bool)output.is_dense, true);
  EXPECT_NEAR (output.points[output.points.size () - 1].x, -0.07793, 1e-4);
  EXPECT_NEAR (output.points[output.points.size () - 1].y,  0.17516, 1e-4);
  EXPECT_NEAR (output.points[output.points.size () - 1].z, -0.0444,  1e-4);

  // Test the sensor_msgs::PointCloud2 method
  sensor_msgs::PointCloud2 output2;
  StatisticalOutlierRemoval<sensor_msgs::PointCloud2> outrem2;
  outrem2.setInputCloud (cloud_blob_ptr_);
  outrem2.setMeanK (50);
  outrem2.setStddevMulThresh (1.0);
  outrem2.filter (output2);

  fromROSMsg (output2, output);

  EXPECT_EQ ((int)output.points.size (), 352);
  EXPECT_EQ ((int)output.width, 352);
  EXPECT_EQ ((bool)output.is_dense, true);
  EXPECT_NEAR (output.points[output.points.size () - 1].x, -0.034667,   1e-4);
  EXPECT_NEAR (output.points[output.points.size () - 1].y,  0.15131,    1e-4);
  EXPECT_NEAR (output.points[output.points.size () - 1].z, -0.00071029, 1e-4);

  outrem2.setNegative (true);
  outrem2.filter (output2);

  fromROSMsg (output2, output);

  EXPECT_EQ ((int)output.points.size (), (int)cloud_.points.size () - 352);
  EXPECT_EQ ((int)output.width, (int)cloud_.width - 352);
  EXPECT_EQ ((bool)output.is_dense, true);
  EXPECT_NEAR (output.points[output.points.size () - 1].x, -0.07793, 1e-4);
  EXPECT_NEAR (output.points[output.points.size () - 1].y,  0.17516, 1e-4);
  EXPECT_NEAR (output.points[output.points.size () - 1].z, -0.0444,  1e-4);
}

//////////////////////////////////////////////////////////////////////////////////////////////
TEST (ConditionalRemoval, Filters)
{
  // Test the PointCloud<PointT> method
  PointCloud<PointXYZ> output;

  // build the condition
  ConditionAnd<PointXYZ>::Ptr range_cond (new ConditionAnd<PointXYZ> ());
  range_cond->addComparison (FieldComparison<PointXYZ>::ConstPtr (new FieldComparison<PointXYZ>("z", ComparisonOps::GT, 0.02)) );
  range_cond->addComparison (FieldComparison<PointXYZ>::ConstPtr (new FieldComparison<PointXYZ>("z", ComparisonOps::LT, 0.04)) );
  range_cond->addComparison (FieldComparison<PointXYZ>::ConstPtr (new FieldComparison<PointXYZ>("y", ComparisonOps::GT, 0.10)) );
  range_cond->addComparison (FieldComparison<PointXYZ>::ConstPtr (new FieldComparison<PointXYZ>("y", ComparisonOps::LT, 0.12)) );

  // build the filter
  ConditionalRemoval<PointXYZ> condrem (range_cond); 
  condrem.setInputCloud (cloud_ptr_);

  // try the dense version
  condrem.setKeepOrganized (true);
  condrem.filter (output);

  EXPECT_EQ ((int)output.points.size (), 28);
  EXPECT_NEAR (output.points[output.points.size () - 1].x, -0.087292, 1e-4);
  EXPECT_NEAR (output.points[output.points.size () - 1].y,  0.103140, 1e-4);
  EXPECT_NEAR (output.points[output.points.size () - 1].z,  0.020825, 1e-4);

  // try the not dense version
  condrem.setKeepOrganized (false);
  condrem.filter (output);

  int num_not_nan = 0;
  for (size_t i = 0; i < output.points.size (); i++) 
  {
    if (pcl_isfinite (output.points[i].x) && 
        pcl_isfinite (output.points[i].y) &&
        pcl_isfinite (output.points[i].z))
      num_not_nan++;
  }

  EXPECT_EQ ((int)output.points.size (), (int)cloud_ptr_->points.size ());
  EXPECT_EQ ((int)output.width, (int)cloud_ptr_->width);
  EXPECT_EQ ((int)output.height, (int)cloud_ptr_->height);
  EXPECT_EQ (num_not_nan, 28);
}

/* ---[ */
int
  main (int argc, char** argv)
{
  // Load a standard PCD file from disk
  loadPCDFile ("./test/bun0.pcd", cloud_blob_);
  cloud_blob_ptr_.reset (new sensor_msgs::PointCloud2 (cloud_blob_));
  fromROSMsg (cloud_blob_, cloud_);
  cloud_ptr_.reset (new PointCloud<PointXYZ> (cloud_));

  indices_.resize (cloud_.points.size ());
  for (size_t i = 0; i < indices_.size (); ++i) { indices_[i] = i; }

  testing::InitGoogleTest (&argc, argv);
  return (RUN_ALL_TESTS ());
}
/* ]--- */