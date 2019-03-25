cmake_minimum_required(VERSION 3.8)
project(kltVO)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

find_package(OpenCV 4.0.0 REQUIRED)

set(SOURCE_FILES
        main_kitti.cpp
        src/utils.cpp
        src/tracking.cpp
        src/eightPoint.cpp
        src/epnp.cpp
        src/ORBextractor.cc
        )

include_directories(
        ${PROJECT_SOURCE_DIR}/include
        $(OpenCV_INCLUDE_DIRS)
)

add_executable(kltVO ${SOURCE_FILES})
target_link_libraries(
        ${PROJECT_NAME}
        ${OpenCV_LIBS}
)