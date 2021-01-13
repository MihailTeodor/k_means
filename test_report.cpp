#include <string>
#include <fstream>
#include <iostream>
#include <chrono>
#include "point.h"
#include "k-means_SEQ.h"
#include "k-means_OPENMP.h"
#include "utilities.h"
#include "k-means_CUDA.cuh"

int main() {
    int num_points[] = {100, 1000, 10000, 100000, 1000000, 10000000};
    int num_clusters[] = {10, 50, 100, 1000};
    int num_threads[] = {2, 4, 8};
    int max_epochs = 20;
    const auto num_dimensions = 3;
    const int num_iterations = 10;
    double tmp_time;
    double computation_time_MAX = 200;

    for (auto p : num_points) {
        std::string tmp = std::to_string(p);
        std::string dataset_name = "rand_dataset_" + tmp + ".txt";
        generate_random_dataset(p);

        std::vector<Point> dataset;
        std::ifstream dataset_file("../datasets/" + dataset_name, std::ifstream::in);
        if (dataset_file) {
            load_dataset(dataset, dataset_file);
            dataset_file.close();
        } else {
            std::cerr << "Error: Could not open dataset.\n";
        }

        for (auto c : num_clusters) {
            if (c < p) {

                std::vector<Point> initial_centroids = generate_initial_centroids(dataset, c);

                tmp_time = 0;
                int iterations_completed = 0;
                for(int k = 0; k < num_iterations; k++) {

                    auto start = std::chrono::high_resolution_clock::now();
                    k_means_SEQ(dataset, initial_centroids, max_epochs);
                    auto finish = std::chrono::high_resolution_clock::now();
                    tmp_time += std::chrono::duration<double>(finish - start).count();
                    iterations_completed ++;
                    if(std::chrono::duration<double>(finish - start).count() > computation_time_MAX)
                        k = num_iterations;
                }
                const double sequential_execution_time = tmp_time / iterations_completed;

                // starting CUDA section
                        const auto num_bytes_dataset = p * num_dimensions * sizeof(double);
                        const auto num_bytes_centroids = c * num_dimensions * sizeof(double);
                        auto *host_dataset = (double *) malloc(num_bytes_dataset);
                        auto *host_centroids = (double *) malloc(num_bytes_centroids);

                        transform_into_array(dataset, p, host_dataset);
                        transform_into_array(initial_centroids, c, host_centroids);

                        tmp_time = 0;
                        iterations_completed = 0;
                        for(int k = 0; k < num_iterations; k++) {
                            double *device_dataset, *device_centroids;
                            CUDA_CHECK_RETURN(cudaMalloc((void **) &device_dataset, num_bytes_dataset));
                            CUDA_CHECK_RETURN(cudaMalloc((void **) &device_centroids, num_bytes_centroids));
                            CUDA_CHECK_RETURN(cudaMemcpy(device_dataset, host_dataset, num_bytes_dataset,
                                               cudaMemcpyHostToDevice));
                            CUDA_CHECK_RETURN(cudaMemcpy(device_centroids, host_centroids, num_bytes_centroids,
                                                         cudaMemcpyHostToDevice));

                            auto start = std::chrono::high_resolution_clock::now();
                            std::tie(device_dataset, device_centroids) = kmeans_cuda(device_dataset, c,
                                                                                     device_centroids, p,
                                                                                     num_dimensions, max_epochs);
                            finish = std::chrono::high_resolution_clock::now();

                            tmp_time += std::chrono::duration<double>(finish - start).count();
                            iterations_completed ++;
                            if(std::chrono::duration<double>(finish - start).count() > computation_time_MAX)
                                k = num_iterations;

                            CUDA_CHECK_RETURN(cudaFree(device_dataset));
                            CUDA_CHECK_RETURN(cudaFree(device_centroids));
                        }

                    const double CUDA_execution_time = tmp_time / iterations_completed;
                    double CUDA_speedup = sequential_execution_time / CUDA_execution_time;
                    free(host_dataset);
                    free(host_centroids);




                for (auto t : num_threads) {
                    tmp_time = 0;
                    iterations_completed = 0;
                    for(int k = 0; k < num_iterations; k++ ) {
                        auto start = std::chrono::high_resolution_clock::now();
                        k_means_OPENMP(dataset, initial_centroids, max_epochs, t);
                        auto finish = std::chrono::high_resolution_clock::now();
                        tmp_time += std::chrono::duration<double>(finish - start).count();
                        iterations_completed++;
                        if(std::chrono::duration<double>(finish - start).count() > computation_time_MAX)
                            k = num_iterations;
                    }
                    const double OPENMP_execution_time = tmp_time / iterations_completed;

                    double OPENMP_speedup = sequential_execution_time / OPENMP_execution_time;

                    cout << "COMPUTATION ON DATASET OF " << p << " POINTS AND " << c << " CENTROIDS, USING " << t << " THREADS FOR OPENMP." << endl;
                    cout << "SEQUENTIAL TIME: " << sequential_execution_time << "; \nOPENMP TIME:     " << OPENMP_execution_time << ";    SPEEDUP OPENMP: " <<
                         OPENMP_speedup << "; \nCUDA TIME:       " << CUDA_execution_time << ";   SPEEDUP CUDA: " << CUDA_speedup << "; \n \n \n";
                }
            }
        }
    }
}