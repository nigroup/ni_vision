# The NI Vision Libraries and Packages #

## Getting your build up and running ##

### Prepwork ###

Install [ROS Hydro](http://wiki.ros.org/hydro/Installation)

Download and build GTest:

Indeed testing with GTest is enabled by catkin. However, since
we also rely on OpenCV, OpenCV's GTest symbols conflict with those
provided by catkin and executing the tests results in a malloc error.

[Using gtest with cv_bridge fails](http://answers.ros.org/question/76453/using-gtest-with-cv_bridge-fails/)
[Issue: cv_bridge exposes internal GTest symbols from OpenCV and/or links incorrectly](https://github.com/ros-perception/vision_opencv/issues/22)

GTest source and build can go outside your catkin workspace.

* mkdir ~/src
* [Download GTest](https://googletest.googlecode.com/files/gtest-1.7.0.zip)
* unzip to ~/src
* cd ~/src/gtest-1.7.0
* cmake .	# perform an in-source build
* make

Download and build the ELM libraries:

ELM source and build can go outside your catkin workspace.

* mkdir ~/src # unless you've alreaded created it
* cd ~/src
* git clone git@github.com:kashefy/elm.git
* mkdir -p ~/build/elm/ros_static
* cd ~/build/elm/ros_static

For CMake >= 2.8.9:

* cmake -DBUILD_SHARED_LIBS=OFF -DGTEST_ROOT=~/src/gtest-1.7.0 ~/src/elm

For earlier versions of CMake:

* cmake -DBUILD_SHARED_LIBS=OFF -DCMAKE_CXX_FLAGS="-fPIC" -DGTEST_ROOT=~/src/gtest-1.7.0 -DOpenCV_DIR=/opt/ros/hydro/share/OpenCV/OpenCVConfig.cmake ~/src/elm

* make -j2

Building the NI Vision Libraries and packages:

* mkdir ~/catkin_ws 	# Create a catkin workspace
* mkdir ~/catkin_ws/src 		# Create a src sub directory
* clone this repository to ~/catkin_ws/src/. You'll end up with ~/catkin_ws/src/ni_vision
* catkin_make --cmake-args -DGTEST_ROOT=~/src/gtest-1.7.0 -DELM_DIR=~/build/elm/ros_static

Running the unittests for target ni:
~/catkin_ws/devel/lib/ni/run_ni_unittests
