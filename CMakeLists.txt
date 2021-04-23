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
add_subdirectory(pnc/planner/locomotion/towr_plus)
add_subdirectory(pnc/planner/dubins)

# Add Executables
add_executable(dubin_test test/dubins.cc)
target_link_libraries(dubin_test dubins)

foreach(example atlas_half_step atlas_one_step atlas_two_step atlas_three_step
                atlas_four_step atlas_five_step atlas_six_step
                atlas_forward_walk atlas_turning atlas_side_walk
                atlas_steer_walk atlas_round_walk
                atlas_block atlas_stair atlas_slope
                atlas_chimney atlas_lr_chimney atlas_chimney_jump atlas_lr_chimney_jump
                valkyrie_forward_walk valkyrie_turning
                valkyrie_round_walk valkyrie_steer_walk
                valkyrie_slope
                nao_forward_walk nao_side_walk)

    add_executable(${example} simulator/towr_plus/${example}.cpp)
    target_link_libraries(${example}
        PRIVATE
        towr_plus
        ifopt::ifopt_ipopt
        my_yaml
    )
    if(SNOPT_FOUND)
        target_link_libraries(${example} PRIVATE ifopt::ifopt_snopt)
    endif()

endforeach(example)

SET (THIS_COM "${PROJECT_SOURCE_DIR}/" )
if (SNOPT_FOUND)
    SET(BUILD_WITH_SNOPT 1)
else()
    SET(BUILD_WITH_SNOPT 0)
endif()
CONFIGURE_FILE(${PROJECT_SOURCE_DIR}/config.h.cmake ${PROJECT_SOURCE_DIR}/configuration.h)
