cmake_minimum_required(VERSION 2.8.3)

set(ROOT_PROJECT ni)
project(${ROOT_PROJECT})

# ----------------------------------------------------------------------------
# Define root project variables
# ----------------------------------------------------------------------------
set(${ROOT_PROJECT}_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/modules) # enforce includes to start with project's parent dir (e.g. #include <elm/core/core.h>)
set(${ROOT_PROJECT}_LIBS    "")
set(${ROOT_PROJECT}_MODULES "")

# ----------------------------------------------------------------------------
# Include some CMake utilities
# ----------------------------------------------------------------------------

include(cmake/FileSystemUtils.cmake)
include(cmake/ListUtils.cmake)
include(cmake/MessageUtils.cmake)

# ----------------------------------------------------------------------------
# Set global CMake variables
# ----------------------------------------------------------------------------
# Print all warnings and enable C++11 support
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++0x")
set(CMAKE_VERBOSE ON CACHE BOOL "Verbose mode")

set(BUILD_NO_ROS OFF CACHE BOOL "Enable building NI libs independent of ROS.")
if(BUILD_NO_ROS)
# TODO: write out alternate CMakeLists file for building without ROS
else(BUILD_NO_ROS)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED
  roscpp
  cv_bridge
  pcl_ros
)

# ----------------------------------------------------------------------------
# System dependencies found with CMake's conventions
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# GTest package
# ----------------------------------------------------------------------------
# Testing framework
# ----------------------------------------------------------------------------
# NOTE: This needs to happen before OpenCV as OpenCV introduced a bug:
# Their ts module results in multiple definitions at linking time
# ----------------------------------------------------------------------------
include(cmake/DetectGTest.cmake)

# ----------------------------------------------------------------------------
# ELM Libraries (must preceed OpenCV)
# ----------------------------------------------------------------------------
# ELM Layers describe the model pipeline
# ----------------------------------------------------------------------------
include(cmake/DetectELM.cmake)

# Non-catkin dependencies covered
# ----------------------------------------------------------------------------

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
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
# generate_messages(
#   DEPENDENCIES
#   std_msgs  # Or other packages containing msgs
# )

###########
## Build ##
###########

## Specify additional locations of header files
include_directories(${${ROOT_PROJECT}_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS})

## configure sub modules (i.e. libraries)

set(${ROOT_PROJECT}_LIBS ${${ROOT_PROJECT}_LIBS} ${catkin_LIBRARIES})

add_subdirectory(modules/${ROOT_PROJECT})

## Building of libraries is handled by modules

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
  INCLUDE_DIRS modules/${ROOT_PROJECT}
  LIBRARIES ni_core ni_layers
  CATKIN_DEPENDS cv_bridge
#  DEPENDS system_lib
)

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
# install(TARGETS ni_lib ni_lib_node
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

status("Configure test targets:")

## Add gtest based cpp test target and link libraries

# Setting up test target spearately
# because catkin's setup uses gtest symbols found in OpenCV
# and therefore runs into a malloc error after running the tests.

status("")

include(cmake/BuildTests.cmake)
status("    Configure unittest project")
build_tests("run_${ROOT_PROJECT}_unittests" "*unittest*")

#set(filter "*unittest*")

#status("Scanning for files: ${filter} in folder: modules/${ROOT_PROJECT}")

#file(GLOB_RECURSE TEST_SRCS modules/${ROOT_PROJECT}/${filter})
#catkin_add_gtest(${ROOT_PROJECT}-test ${TEST_SRCS})

#if(TARGET ${ROOT_PROJECT}-test)

#  target_link_libraries(${ROOT_PROJECT}-test ${${ROOT_PROJECT}_MODULES} ${${ROOT_PROJECT}_LIBS})

#endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)

# ----------------------------------------------------------------------------
# Done configuring the build.
# Let's print some summary info
# ----------------------------------------------------------------------------

status("")

status("Build settings:")
status("")
status("    Build type:\t${CMAKE_BUILD_TYPE}")
status("    C Flags:\t${CMAKE_C_FLAGS}")
status("    C++ Flags:\t${CMAKE_CXX_FLAGS}")

status("")
status("Build:")
set(MODULE_NAMES ${${ROOT_PROJECT}_MODULES})
strip_lib_prefix(MODULE_NAMES ${MODULE_NAMES})
status("    modules:  "      ${MODULE_NAMES})
status("    unit-tests:  "   run_${ROOT_PROJECT}_unittests)

status("")

status("Dependencies:")

status("")
status("    GTest:    "   GTEST_FOUND THEN ${GTEST_ROOT} ELSE NO)

status("")
status("    ELM:    "   ELM_FOUND THEN "Yes"  ELSE NO)
status("        lib Dir.:  "   ELM_FOUND THEN "${ELM_LIB_DIR}" ELSE "-")
set(ELM_LIB_NAMES ${ELM_LIBS})
convert_to_lib_name(ELM_LIB_NAMES ${ELM_LIB_NAMES})
list_unique(ELM_LIB_NAMES ${ELM_LIB_NAMES})
status("        components :  "      ELM_FOUND THEN ${ELM_LIB_NAMES} ELSE "-")

status("")

endif(BUILD_NO_ROS) # end file with this line.
