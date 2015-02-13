# The NI Vision Libraries and Packages #

## Getting your build up and running ##

### Prepwork ###

Install ROS Hydro[ROS](http://www.ros.org/)

Download and build GTest:

Indeed testing with GTest is enabled by catkin. However, since
we also rely on OpenCV, OpenCV's GTest symbols conflict with those
provided by catkin and executing the tests results in a malloc error.

[Using gtest with cv_bridge fails](http://answers.ros.org/question/76453/using-gtest-with-cv_bridge-fails/)
[Issue: cv_bridge exposes internal GTest symbols from OpenCV and/or links incorrectly](https://github.com/ros-perception/vision_opencv/issues/22)


* [Download GTest](https://googletest.googlecode.com/files/gtest-1.7.0.zip)
* unzip to ~/src
* cd ~/src/gtest-1.7.0
* cmake .
* make

Building the NI Vision Libraries and packages:

* mkdir ~/catkin_ws 	# Create a catkin workspace
* mkdir ~/catkin_ws/src 		# Create a src sub directory
* clone this repository to ~/catkin_ws/src/. You'll end up with ~/catkin_ws/src/ni_vision
* --cmake-args -DGTEST_ROOT=~/src/gtest-1.7.0

Running the unittests for target ni:
~/catkin_ws/devel/lib/ni/run_ni_unittests
