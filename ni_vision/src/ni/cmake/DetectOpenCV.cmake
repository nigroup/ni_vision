# ----------------------------------------------------------------------------
# OpenCV
# ----------------------------------------------------------------------------
# find_package OpenCV
# Defines: OpenCV_FOUND, OpenCV_INCLUDE_DIRS, OpenCV_LIBS, OpenCV_LINK_LIBRARIES
# ----------------------------------------------------------------------------
# Force the user to tell us which OpenCV they want (otherwise find_package can find the wrong one, cache it and changes to OpenCV_DIR are ignored)

set(OPENCV_REQUESTED_COMPONENTS core highgui imgproc)

if(DEFINED OpenCV_DIR)
    find_package(OpenCV REQUIRED ${OPENCV_REQUESTED_COMPONENTS} PATHS ${OpenCV_DIR})
else(DEFINED OpenCV_DIR)
    find_package(OpenCV REQUIRED ${OPENCV_REQUESTED_COMPONENTS})
endif(DEFINED OpenCV_DIR)

if(OpenCV_FOUND)
    list(APPEND ${ROOT_PROJECT}_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS})
    list(APPEND ${ROOT_PROJECT}_LIBS ${OpenCV_LIBS})
else(OpenCV_FOUND)
    message(SEND_ERROR "Failed to find OpenCV. Double check that \"OpenCV_DIR\" to the root build directory of OpenCV.")
endif(OpenCV_FOUND)
