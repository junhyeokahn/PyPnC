cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)
project(towr_plus VERSION 1.3.2)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

find_package(Eigen)
include_directories(${EIGEN_INCLUDE_DIRS})
find_package(ifopt 2.0.1 REQUIRED)

if(TARGET ifopt::ifopt_snopt)
    set(SNOPT_FOUND TRUE)
else()
    set(SNOPT_FOUND FALSE)
endif()

###########
## Build ##
###########
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE DEBUG)
IF(NOT CMAKE_BUILD_TYPE MATCHES Release)
  message(STATUS "CMAKE_BUILD_TYPE not set to Release -> impacts performance")
endif()

include_directories("${PROJECT_SOURCE_DIR}")
include_directories("${PROJECT_SOURCE_DIR}/external_source")
include_directories("${PROJECT_SOURCE_DIR}/pnc/planner/locomotion/towr_plus/include")
include_directories("${PROJECT_SOURCE_DIR}/pnc/planner/dubins")

add_subdirectory(external_source)
add_subdirectory(util)
#add_subdirectory(pnc/planner/locomotion/towr_plus)
add_subdirectory(pnc/planner/dubins)

# Add Executables

add_executable(dubins_test test/dubins.cc)
target_link_libraries(dubins_test PRIVATE dubins)

target_link_libraries(dubins_test PRIVATE dubins)

add_executable(casadi_file_test test/casadi_file_test.cc
    pnc/planner/locomotion/towr_plus/src/atlas_crbi.c)

#add_executable(atlas_two_step_yaml_test test/atlas_two_step_yaml_test.cc)
#target_link_libraries(atlas_two_step_yaml_test
    #PRIVATE
    #towr_plus
    #ifopt::ifopt_ipopt
    #my_yaml
#)
#if(SNOPT_FOUND)
    #target_link_libraries(atlas_two_step_yaml_test PRIVATE ifopt::ifopt_snopt)
#endif()

#add_executable(atlas_no_step_yaml_test test/atlas_no_step_yaml_test.cc)
#target_link_libraries(atlas_no_step_yaml_test
    #PRIVATE
    #towr_plus
    #ifopt::ifopt_ipopt
    #my_yaml
#)
#if(SNOPT_FOUND)
    #target_link_libraries(atlas_no_step_yaml_test PRIVATE ifopt::ifopt_snopt)
#endif()

#add_executable(atlas_one_step_yaml_test test/atlas_one_step_yaml_test.cc)
#target_link_libraries(atlas_one_step_yaml_test
    #PRIVATE
    #towr_plus
    #ifopt::ifopt_ipopt
    #my_yaml
#)
#if(SNOPT_FOUND)
    #target_link_libraries(atlas_one_step_yaml_test PRIVATE ifopt::ifopt_snopt)
#endif()

add_executable(mlp_model_test test/mlp_model_test.cc)
target_link_libraries(mlp_model_test my_util)

SET (THIS_COM "${PROJECT_SOURCE_DIR}/" )
if (SNOPT_FOUND)
    SET(BUILD_WITH_SNOPT 1)
else()
    SET(BUILD_WITH_SNOPT 0)
endif()
CONFIGURE_FILE(${PROJECT_SOURCE_DIR}/config.h.cmake ${PROJECT_SOURCE_DIR}/configuration.h)
