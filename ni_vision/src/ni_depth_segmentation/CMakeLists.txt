cmake_minimum_required(VERSION 2.8.3)
project(ni_depth_segmentation)

# ----------------------------------------------------------------------------
# Include some CMake utilities
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Set global CMake variables
# ----------------------------------------------------------------------------
# Print all warnings and enable C++11 support
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++0x")
set(CMAKE_VERBOSE ON CACHE BOOL "Verbose mode")

# ----------------------------------------------------------------------------
# NI Libraries
# ----------------------------------------------------------------------------
set(NI_LIBS ni_core ni_layers)

# ----------------------------------------------------------------------------
# ELM Libraries (must preceed OpenCV)
# ----------------------------------------------------------------------------
# ELM Layers describe the model pipeline
# ----------------------------------------------------------------------------
include(../ni/cmake/DetectELM.cmake)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  ni
  roscpp
  rospy
  std_msgs
  message_generation
  cv_bridge
  OpenCV               # for complete link directories
  image_transport
  pcl_conversions
  pcl_ros
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)


## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend and a run_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependencies might have been
##     pulled in transitively but can be declared for certainty nonetheless:
##     * add a build_depend tag for "message_generation"
##     * add a run_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
add_message_files(
   FILES
   Num.msg
   NodeKVP.msg
)

## Generate services in the 'srv' folder
add_service_files(
  FILES
  AddTwoInts.srv
)

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
)

generate_dynamic_reconfigure_options(
  cfg/gen_params_NodeKVP.cfg
)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES ni_seg_track
#  CATKIN_DEPENDS ni roscpp rospy std_msgs
  CATKIN_DEPENDS dynamic_reconfigure message_runtime
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)
include_directories(
  ${catkin_INCLUDE_DIRS} ${ELM_INCLUDE_DIRS}
)

# Declare a cpp library
# glob non-node code only
file(GLOB_RECURSE SRC_LIST src/${PROJECT_NAME}/*.c*)
file(GLOB_RECURSE HEADERS  src/${PROJECT_NAME}/*.h*)

# Filter out test related sources
set(REGEX_EXCLUDE_FROM_MODULE "(.*test.*)|(.*~.*)(.*node.*)")
list_filterout(SRC_LIST ${REGEX_EXCLUDE_FROM_MODULE})
list_filterout(HEADERS  ${REGEX_EXCLUDE_FROM_MODULE})

add_library(ni_depth_segmentation
  ${SRC_LIST}
  ${HEADERS}
)

# Declare a cpp executable for nodes

set(NODE_NAMES "")

# glob node code only
file(GLOB SRC_LIST src/*node.c*)
file(GLOB HEADERS  src/*node.h*)

# Filter out test related sources
set(REGEX_EXCLUDE_FROM_MODULE "(.*test.*)|(.*~.*)")
list_filterout(SRC_LIST ${REGEX_EXCLUDE_FROM_MODULE})
list_filterout(HEADERS  ${REGEX_EXCLUDE_FROM_MODULE})

#add_executable(ni_seg_track_node
#  ${SRC_LIST}
#  ${HEADERS}
#)

set(NODE_NAME depth_map_node)
add_executable(${NODE_NAME} src/${NODE_NAME}.cpp)
target_link_libraries(${NODE_NAME} ${catkin_LIBRARIES} ${ELM_LIBS} ${NI_LIBS})
list(APPEND NODE_NAMES ${NODE_NAME})

set(NODE_NAME depth_segmentation_node)
add_executable(${NODE_NAME} src/${NODE_NAME}.cpp)
target_link_libraries(${NODE_NAME} ${catkin_LIBRARIES} ${ELM_LIBS} ${NI_LIBS})
add_dependencies(${NODE_NAME} ${PROJECT_NAME}_gencfg)
list(APPEND NODE_NAMES ${NODE_NAME})

set(NODE_NAME surface_tracking_node)
add_executable(${NODE_NAME} src/${NODE_NAME}.cpp)
target_link_libraries(${NODE_NAME} ${catkin_LIBRARIES} ${ELM_LIBS} ${NI_LIBS})
list(APPEND NODE_NAMES ${NODE_NAME})

set(NODE_NAME surface_segmentation_and_tracking_node)
add_executable(${NODE_NAME} src/${NODE_NAME}.cpp)
target_link_libraries(${NODE_NAME} ${catkin_LIBRARIES} ${ELM_LIBS} ${NI_LIBS})
add_dependencies(${NODE_NAME} ${PROJECT_NAME}_gencfg)
list(APPEND NODE_NAMES ${NODE_NAME})

#status(${ELM_DEPS_LIBS})
if(ELM_WITH_MATIO)
    set(NODE_NAME nyu_depth_v2_node)
    add_executable(${NODE_NAME} src/${NODE_NAME}.cpp)
    target_link_libraries(${NODE_NAME} ${catkin_LIBRARIES} ${ELM_LIBS} ${NI_LIBS} ${ELM_DEPS_LIBS})
    list(APPEND NODE_NAMES ${NODE_NAME})
else(ELM_WITH_MATIO)
    status("Skipping node: nyu_depth_v2_node")
endif(ELM_WITH_MATIO)

add_executable(talker_node src/talker_node.cpp)
target_link_libraries(talker_node ${catkin_LIBRARIES})
add_dependencies(talker_node beginner_tutorials_generate_messages_cpp)

add_executable(listener_node src/listener_node.cpp)
target_link_libraries(listener_node ${catkin_LIBRARIES})
add_dependencies(listener_node beginner_tutorials_generate_messages_cpp)

status("")
status("Build:")
status("    nodes: ${NODE_NAMES}")

## Add cmake target dependencies of the executable/library
## as an example, message headers may need to be generated before nodes
# add_dependencies(ni_seg_track_node ni_seg_track_generate_messages_cpp)

# Specify libraries to link a library or executable target against
#target_link_libraries(ni_seg_track_node
#  ${catkin_LIBRARIES}
#)

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
# install(TARGETS ni_seg_track ni_seg_track_node
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_ni_seg_track.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
