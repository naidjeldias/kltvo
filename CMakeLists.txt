cmake_minimum_required(VERSION 3.8)
project(kltVO)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)

set(SOURCE_FILES main.cpp utils.cpp utils.h tracking.cpp tracking.h eightpoint.hpp eightPoint.cpp epnp.h epnp.cpp)
add_executable(kltVO ${SOURCE_FILES})

include_directories($(OpenCV_INCLUDE_DIRS))
target_link_libraries(kltVO ${OpenCV_LIBS})