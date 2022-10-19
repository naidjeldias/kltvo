cmake_minimum_required(VERSION 3.5      )
project(kltvo)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

find_package(OpenCV 4.0.0 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Pangolin 0.4 REQUIRED)

include_directories(
        ${PROJECT_SOURCE_DIR}/include
        $(OpenCV_INCLUDE_DIRS)
        ${EIGEN3_INCLUDE_DIR}
        ${Pangolin_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME} SHARED
        src/utils.cpp
        src/tracking.cpp
        src/eightPoint.cpp
        src/ORBextractor.cc
        src/viewer.cpp
)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")

target_link_libraries(
        ${PROJECT_NAME}
        ${OpenCV_LIBS}
        ${EIGEN3_LIBS}
        ${Pangolin_LIBRARIES}
)

add_executable(stereo_kitti
        examples/kitti/main_kitti.cpp)
target_link_libraries(stereo_kitti ${PROJECT_NAME})

add_executable(stereo_euroc
        examples/euroc/main_euroc.cpp)
target_link_libraries(stereo_euroc ${PROJECT_NAME})