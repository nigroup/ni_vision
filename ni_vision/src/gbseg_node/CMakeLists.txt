cmake_minimum_required(VERSION 2.8.3)
project(gbseg_node)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  image_transport
  cv_bridge
  message_generation
)

add_message_files(
  FILES
  params_GbSeg.msg
  point.msg
  segment.msg
  segmentation.msg
)


 generate_messages(
#  DEPENDENCIES
#  sensor_msgs
 )

catkin_package(
  #CATKIN_DEPENDS message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(gb_segmentation_node src/gb_segmentation_node.cpp)
target_link_libraries(gb_segmentation_node ${catkin_LIBRARIES})
target_link_libraries(gb_segmentation_node ${OpenCV_LIBRARIES})

add_executable(params_GbSeg_publisher src/params_GbSeg_publisher.cpp)
target_link_libraries(params_GbSeg_publisher ${catkin_LIBRARIES})

add_executable(result_GbSeg_subscriber src/result_GbSeg_subscriber.cpp)
target_link_libraries(result_GbSeg_subscriber ${catkin_LIBRARIES})
target_link_libraries(gb_segmentation_node ${OpenCV_LIBRARIES})
