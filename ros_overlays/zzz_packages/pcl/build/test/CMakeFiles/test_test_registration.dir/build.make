# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canoncical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ni/ros_overlays/zzz_packages/pcl

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ni/ros_overlays/zzz_packages/pcl/build

# Utility rule file for test_test_registration.

test/CMakeFiles/test_test_registration: ../bin/test_registration
	cd /home/ni/ros_overlays/zzz_packages/pcl && /opt/ros/electric/ros/tools/rosunit/bin/rosunit --name=test_registration --time-limit=60.0 /home/ni/ros_overlays/zzz_packages/pcl/bin/test_registration

test_test_registration: test/CMakeFiles/test_test_registration
test_test_registration: test/CMakeFiles/test_test_registration.dir/build.make
.PHONY : test_test_registration

# Rule to build all files generated by this target.
test/CMakeFiles/test_test_registration.dir/build: test_test_registration
.PHONY : test/CMakeFiles/test_test_registration.dir/build

test/CMakeFiles/test_test_registration.dir/clean:
	cd /home/ni/ros_overlays/zzz_packages/pcl/build/test && $(CMAKE_COMMAND) -P CMakeFiles/test_test_registration.dir/cmake_clean.cmake
.PHONY : test/CMakeFiles/test_test_registration.dir/clean

test/CMakeFiles/test_test_registration.dir/depend:
	cd /home/ni/ros_overlays/zzz_packages/pcl/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ni/ros_overlays/zzz_packages/pcl /home/ni/ros_overlays/zzz_packages/pcl/test /home/ni/ros_overlays/zzz_packages/pcl/build /home/ni/ros_overlays/zzz_packages/pcl/build/test /home/ni/ros_overlays/zzz_packages/pcl/build/test/CMakeFiles/test_test_registration.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/CMakeFiles/test_test_registration.dir/depend

