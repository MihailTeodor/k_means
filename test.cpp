#include <random>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <tuple>
#include <chrono>

#include "point.h"
#include "k-means_SEQ.h"
#include "k-means_OPENMP.h"
#include "utilities.h"

#include "k-means_CUDA.cuh"


int main () {

    int num_clusters = 100;
    int num_points = 100000;
    int max_epochs = 20;
    int nr_threads = 2;
    const auto num_dimensions = 3;

    std::string tmp = std::to_string(num_points);
    std::string dataset_name = "rand_dataset_" + tmp + ".txt";

    generate_random_dataset(num_points);

    std::vector<Point> dataset;

    std::ifstream dataset_file("../datasets/" + dataset_name, std::ifstream::in);
    if(dataset_file) {
        load_dataset(dataset, dataset_file);
        dataset_file.close();
    }else{
        std::cerr << "Error: Could not open dataset.\n";
    }

    std::vector<Point> initial_centroids = generate_initial_centroids(dataset, num_clusters);

    std::vector<Point> final_dataset, final_centroids;

    auto *testArrayCentroids = (double *) malloc(num_clusters * num_dimensions * sizeof(double));   //USED FOR DEBUGGING

    std::cout << "Executing SEQUENTIAL k-means.\n";
    auto start = std::chrono::high_resolution_clock::now();
    std::tie(final_dataset, final_centroids) = k_means_SEQ(dataset, initial_centroids, max_epochs);
    auto finish = std::chrono::high_resolution_clock::now();
    const double sequential_execution_time = std::chrono::duration<double>(finish - start).count();

    std::cout << "- Sequential K-Means:\n";
    std::cout << "Execution Time: " << sequential_execution_time << " s\n\n";

    std::cout << "PRINTING CENTROIDS\n";
    transform_into_array(final_centroids, num_clusters, testArrayCentroids);
    print_array(testArrayCentroids, num_clusters);

    try{
        printf("Drawing the chart...\n");
        draw_chart_gnu(final_dataset);
    }catch(int e){
        printf("Chart not available, gnuplot not found");
    }


    std::cout << "Executing OPENMP k-means.\n";
    start = std::chrono::high_resolution_clock::now();
    std::tie(final_dataset, final_centroids) = k_means_OPENMP(dataset, initial_centroids, max_epochs, nr_threads);
    finish = std::chrono::high_resolution_clock::now();
    const double OPENMP_execution_time = std::chrono::duration<double>(finish - start).count();

    std::cout << "- OPENMP K-Means:\n";
    std::cout << "Execution Time: " << OPENMP_execution_time << " s\n\n";

    std::cout << "PRINTING CENTROIDS\n";
   transform_into_array(final_centroids, num_clusters, testArrayCentroids);
    print_array(testArrayCentroids, num_clusters);

    try{
        printf("Drawing the chart...\n");
        draw_chart_gnu(final_dataset);
    }catch(int e){
        printf("Chart not available, gnuplot not found");
    }


    const auto num_bytes_dataset = num_points * num_dimensions * sizeof(double);
    const auto num_bytes_centroids = num_clusters * num_dimensions * sizeof(double);
    auto *host_dataset = (double *) malloc(num_bytes_dataset);
    auto *host_centroids = (double *) malloc(num_bytes_centroids);

    transform_into_array(dataset, num_points, host_dataset);
    transform_into_array(initial_centroids, num_clusters, host_centroids);

    double *device_dataset, *device_centroids;
    CUDA_CHECK_RETURN(cudaMalloc((void **) &device_dataset, num_bytes_dataset));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &device_centroids, num_bytes_centroids));
    CUDA_CHECK_RETURN(cudaMemcpy(device_dataset, host_dataset, num_bytes_dataset, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(device_centroids, host_centroids, num_bytes_centroids, cudaMemcpyHostToDevice));

    std::cout << "Executing CUDA k-means.\n";
    start = std::chrono::high_resolution_clock::now();
    std::tie(device_dataset, device_centroids) = kmeans_cuda(device_dataset, num_clusters, device_centroids, num_points, num_dimensions, max_epochs);
    finish = std::chrono::high_resolution_clock::now();
    const double CUDA_execution_time = std::chrono::duration<double>(finish - start).count();
    std::cout << "- CUDA K-Means:\n";
    std::cout << "Execution Time: " << CUDA_execution_time << " s\n\n";

    CUDA_CHECK_RETURN(cudaMemcpy(host_dataset, device_dataset, num_bytes_centroids, cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(host_centroids, device_centroids, num_bytes_centroids, cudaMemcpyDeviceToHost));

    std::cout << "PRINTING THE HOST_CENTROIDS AFTER CUDA \n";
    print_array(host_centroids, num_clusters);

    transform_from_array(host_dataset, final_dataset, num_points);
    transform_from_array(host_centroids, final_centroids, num_clusters);


    CUDA_CHECK_RETURN(cudaFree(device_dataset));
    CUDA_CHECK_RETURN(cudaFree(device_centroids));


    free(host_dataset);
    free(host_centroids);

    try{
        printf("Drawing the chart...\n");
        draw_chart_gnu(final_dataset);
    }catch(int e){
        printf("Chart not available, gnuplot not found");
    }
}
