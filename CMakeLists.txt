cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)
project(towr_plus VERSION 1.3.2)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

find_package(Eigen)
include_directories(${EIGEN_INCLUDE_DIRS})
find_package(ifopt 2.0.1 REQUIRED)

###########
## Build ##
###########
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)
IF(NOT CMAKE_BUILD_TYPE MATCHES Release)
  message(STATUS "CMAKE_BUILD_TYPE not set to Release -> impacts performance")
endif()

include_directories("${PROJECT_SOURCE_DIR}")
include_directories("${PROJECT_SOURCE_DIR}/util")
include_directories("${PROJECT_SOURCE_DIR}/pnc/planner/locomotion/towr_plus/include")

add_subdirectory(util)
add_subdirectory(pnc/planner/locomotion/towr_plus)

add_executable(atlas_two_step_test test/atlas_two_step_test.cc)
target_link_libraries(atlas_two_step_test
    PRIVATE
    towr_plus
    ifopt::ifopt_ipopt
)

add_executable(atlas_two_step_yaml_test test/atlas_two_step_yaml_test.cc)
target_link_libraries(atlas_two_step_yaml_test
    PRIVATE
    towr_plus
    ifopt::ifopt_ipopt
    my_yaml
)

add_executable(atlas_no_step_yaml_test test/atlas_no_step_yaml_test.cc)
target_link_libraries(atlas_no_step_yaml_test
    PRIVATE
    towr_plus
    ifopt::ifopt_ipopt
    my_yaml
)

SET (THIS_COM "${PROJECT_SOURCE_DIR}/" )
CONFIGURE_FILE(${PROJECT_SOURCE_DIR}/config.h.cmake ${PROJECT_SOURCE_DIR}/configuration.h)
