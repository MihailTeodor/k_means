cmake_minimum_required(VERSION 3.17)

project(k_means)
#project(k_means LANGUAGES CXX CUDA)




#set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

add_executable(k_means_TEST test.cpp point.h k-means_SEQ.h k-means_SEQ.cpp k-means_OPENMP.h k-means_OPENMP.cpp k-means_CUDA.cuh k-means_CUDA.cu utilities.h utilities.cpp)
add_executable(k_means_TEST_REPORT test_report.cpp point.h k-means_SEQ.h k-means_SEQ.cpp k-means_OPENMP.h k-means_OPENMP.cpp k-means_CUDA.cuh k-means_CUDA.cu utilities.h utilities.cpp)


#find_package(CUDA REQUIRED)
#include_directories("${CUDA_INCLUDE_DIRS}")