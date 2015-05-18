# ----------------------------------------------------------------------------
#  CMake file for 3rdparty module
# ----------------------------------------------------------------------------

set(MODULE_NAME ${ROOT_PROJECT}_3rdparty)

project(${MODULE_NAME})

file(GLOB_RECURSE SRC_LIST *.c*)
file(GLOB_RECURSE HEADERS  *.h*)

# Filter out test related sources
set(REGEX_EXCLUDE_FROM_MODULE "(.*test.*)|(.*~.*)|(.*matrix.cpp)|(.*siftfast.*)")
list_filterout(SRC_LIST ${REGEX_EXCLUDE_FROM_MODULE})
list_filterout(HEADERS  ${REGEX_EXCLUDE_FROM_MODULE})

list(APPEND SRC_LIST siftfast/libsiftfast.cpp)
list(APPEND HEADERS ${CMAKE_CURRENT_LIST_DIR}/siftfast/siftfast.h)

status("HEADERS:${HEADERS}")

add_library(${MODULE_NAME} ${SRC_LIST} ${HEADERS})
target_link_libraries(${MODULE_NAME} ${${ROOT_PROJECT}_LIBS})

list(APPEND ${ROOT_PROJECT}_MODULES ${MODULE_NAME})
set(${ROOT_PROJECT}_MODULES ${${ROOT_PROJECT}_MODULES} PARENT_SCOPE)

# add module's install targets, header installation is centralized
install(TARGETS ${MODULE_NAME} DESTINATION lib)
