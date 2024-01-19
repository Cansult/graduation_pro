cmake_minimum_required(VERSION 3.15)
project(Hough)
add_executable(Hough Hough.cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11") # set C++ 11

# find_package(PkgConfig REQUIRED)
find_package(OpenCV REQUIRED)

# pkg_search_module(CV REQUIRED opencv4)

# MESSAGE(STATUS ${CV_INCLUDE_DIRS})
# MESSAGE(STATUS ${CV_LIBRARY_DIRS})

# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_STANDARD 11)
# set(incOpen "/opt/homebrew/Cellar/opencv/4.8.1_5/include/opencv4/")
# include_directories(${CV_INCLUDE_DIRS})
# link_directories(${CV_LIBRARY_DIRS})
target_link_libraries(Hough ${OpenCV_LIBS})
# include_directories(${OpenCV_INCLUDE_DIRS}) # Not needed for CMake >= 2.8.11
