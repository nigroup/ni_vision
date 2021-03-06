# ----------------------------------------------------------------------------
#  CMake function for building tests
# 
#  Recursively scans ${CMAKE_SOURCE_DIR} for source files according to filter (e.g. *unittest*)
#  And builds them as part of a single binary for push-button execution
#  Args:
#    test_project (string): test project name
#    filter (string): source file filter to use with CMake's file(GLOB_RECURSE ...) routine
# ----------------------------------------------------------------------------

FUNCTION(build_tests test_project filter)

    message(STATUS "------- Test Project: ${test_project} ------- ")
    message(STATUS "Scanning for files: ${filter} in folder: ${CMAKE_SOURCE_DIR}")
    file(GLOB_RECURSE SRCS ${CMAKE_SOURCE_DIR}/${filter})
     
    list(LENGTH SRCS nbTestFiles )
    if(nbTestFiles GREATER 0)
        message(STATUS  "Identified ${nbTestFiles} test source files to build.")
        add_executable(${test_project} ${SRCS})

        add_dependencies(${test_project} ${${ROOT_PROJECT}_MODULES})
            set(${test_project}_LIBS ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES} ${${ROOT_PROJECT}_LIBS} ${${ROOT_PROJECT}_MODULES})
            list(REMOVE_DUPLICATES ${test_project}_LIBS)
            target_link_libraries(${test_project} ${${test_project}_LIBS})

    else(nbTestFiles GREATER 0)
        message( WARNING  "Unable to locate any test files. No tests to add to this project")
    endif(nbTestFiles GREATER 0)
    
ENDFUNCTION( build_tests )
