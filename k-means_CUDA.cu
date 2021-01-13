#include <stdio.h>
#include <time.h>
#include <iostream>
#include "k-means_CUDA.cuh"
#define THREAD_PER_BLOCK 1024

__constant__ int NUM_POINTS, POINT_SIZE, NUM_CENTROIDS, MAX_EPOCHS;

__device__ double TOLLERANCE = 0.00001;

__host__ void check_cuda_error(const char *file, const unsigned line, const char *statement, const cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << statement << " returned " << cudaGetErrorString(err) << "(" << err << ") at " << file << ":"
                  << line << "\n";
        exit(EXIT_FAILURE);
    }
}


__device__ double doubleAtomicAdd(double *address, double val) {
    auto *address_as_ull =
            (unsigned long long int *) address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double((long long int) assumed)));
    } while (assumed != old);
    return __longlong_as_double((long long int) old);
}


__global__ void update_centroids(double* device_centroids, double* centroids_details, bool* iterate) {

    int unsigned id_centroid = blockDim.x * blockIdx.x + threadIdx.x;

    double centroid_x_new, centroid_y_new, centroid_x_old, centroid_y_old = 0;

    if (id_centroid < NUM_CENTROIDS) {

        centroid_x_new = centroids_details[id_centroid + (NUM_CENTROIDS * 0)] / centroids_details[id_centroid + (NUM_CENTROIDS * 2)];
        centroid_y_new = centroids_details[id_centroid + (NUM_CENTROIDS * 1)] / centroids_details[id_centroid + (NUM_CENTROIDS * 2)];

        centroid_x_old = device_centroids[id_centroid + (NUM_CENTROIDS * 0)];
        centroid_y_old = device_centroids[id_centroid + (NUM_CENTROIDS * 1)];

        if(!check_tollerance(centroid_x_old, centroid_x_new, centroid_y_old, centroid_y_new)) {
            *iterate = true;
            device_centroids[id_centroid + (NUM_CENTROIDS * 0)] = centroid_x_new;
            device_centroids[id_centroid + (NUM_CENTROIDS * 1)] = centroid_y_new;
        }
    }
}


__device__ double distance(double x1, double x2, double y1, double y2) {
    return sqrt(((x1 - x2) * (x1 - x2)) + ((y1 - y2) * (y1 - y2)));
}


__device__ bool check_tollerance(double x_old, double x_new, double y_old, double y_new){
    if(fabs(x_old - x_new) < TOLLERANCE && fabs(y_old - y_new) < TOLLERANCE)
        return true;
    else
        return false;
}


__global__ void assign_clusters(double* device_dataset, double* device_centroids, double* centroids_details) {

    long id_punto = threadIdx.x + blockIdx.x * blockDim.x;

    if (id_punto < NUM_POINTS) {
        double punto_x, punto_y, centroid_x, centroid_y = 0;

        punto_x = device_dataset[id_punto + (NUM_POINTS * 0)];
        punto_y = device_dataset[id_punto + (NUM_POINTS * 1)];

        long best_centroid_id = 0;
        double distMIN = INFINITY;

        for (int i = 0; i < NUM_CENTROIDS; i++) {

            centroid_x = device_centroids[i + (NUM_CENTROIDS * 0)];
            centroid_y = device_centroids[i + (NUM_CENTROIDS * 1)];

            auto dist = distance(punto_x, centroid_x, punto_y, centroid_y);
            if (dist < distMIN) {
                best_centroid_id = i;
                distMIN = dist;
            }
        }

        device_dataset[id_punto + (NUM_POINTS * 2)] = best_centroid_id;
        doubleAtomicAdd(&centroids_details[best_centroid_id + (NUM_CENTROIDS * 0)], punto_x);
        doubleAtomicAdd(&centroids_details[best_centroid_id + (NUM_CENTROIDS * 1)], punto_y);
        doubleAtomicAdd(&centroids_details[best_centroid_id + (NUM_CENTROIDS * 2)], 1);
    }
}


__host__ std::tuple<double *, double *>
kmeans_cuda(double *device_dataset, const short num_centroids,
            double *device_centroids, const int num_points, const short point_size, const int max_epochs)
{

    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(NUM_CENTROIDS, &num_centroids, sizeof(int)));
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(POINT_SIZE, &point_size, sizeof(int)));
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(NUM_POINTS, &num_points, sizeof(int)));
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(MAX_EPOCHS, &max_epochs, sizeof(int)));

    double *centroids_details;
    bool* device_iterate;
    bool host_iterate;
    int iteration = 0;

    CUDA_CHECK_RETURN(cudaMalloc((void **) &centroids_details, num_centroids * point_size * sizeof(double)));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &device_iterate, sizeof(bool)));

    do{

        host_iterate = false;
        CUDA_CHECK_RETURN(cudaMemcpy(device_iterate, &host_iterate, sizeof(bool), cudaMemcpyHostToDevice));

        iteration ++;

        CUDA_CHECK_RETURN(cudaMemset(centroids_details, 0, num_centroids * point_size * sizeof(double)));

        assign_clusters <<<(num_points + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK, THREAD_PER_BLOCK>>> (device_dataset, device_centroids, centroids_details);
        cudaDeviceSynchronize();

        update_centroids <<<1, num_centroids >>> (device_centroids, centroids_details, device_iterate);
        cudaDeviceSynchronize();

        CUDA_CHECK_RETURN(cudaMemcpy(&host_iterate, device_iterate, sizeof(bool), cudaMemcpyDeviceToHost));

    } while(host_iterate && iteration < max_epochs);

    return {device_dataset, device_centroids};
}
