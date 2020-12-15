#ifndef K_MEANS_K_MEANS_CUDA_CUH
#define K_MEANS_K_MEANS_CUDA_CUH

#include "cuda_runtime.h"
#include <tuple>

#define CUDA_CHECK_RETURN(value) check_cuda_error(__FILE__, __LINE__, #value, value)

__host__ void check_cuda_error(const char *file, const unsigned line, const char *statement, const cudaError_t err);

__device__ double doubleAtomicAdd(double *address, double val);

__global__ void update_centroids(double* device_centroids, double* centroids_details, bool* iterate);

__device__ double distance(double x1, double x2, double y1, double y2);

__device__ bool check_tollerance(double x_old, double x_new, double y_old, double y_new);

__global__ void assign_clusters(double* device_dataset, double* device_centroids, double* centroids_details);

__host__ std::tuple<double *, double *>
kmeans_cuda(double *device_dataset, const short num_clusters,
            double *device_centroids, const int num_points, const short point_size, const int max_epochs);


#endif //K_MEANS_K_MEANS_CUDA_CUH

