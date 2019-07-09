cmake_minimum_required(VERSION 3.8)
project(kltVO)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

find_package(OpenCV 4.0.0 REQUIRED)

include_directories(
        ${PROJECT_SOURCE_DIR}/include
        $(OpenCV_INCLUDE_DIRS)
)

add_library(${PROJECT_NAME} SHARED
        src/utils.cpp
        src/tracking.cpp
        src/eightPoint.cpp
        src/epnp.cpp
        src/ORBextractor.cc
)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -pthread")

target_link_libraries(
        ${PROJECT_NAME}
        ${OpenCV_LIBS}
)

add_executable(stereo_kitti
        main_kitti.cpp)
target_link_libraries(stereo_kitti ${PROJECT_NAME})

add_executable(stereo_euroc
        main_euroc.cpp)
target_link_libraries(stereo_euroc ${PROJECT_NAME})