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
#include "k-means_CUDA.cuh"

void generate_random_dataset(int nr_points) {
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    const std::string rand_dataset_name =
            "rand_dataset_" + std::to_string(nr_points) +
            ".txt";

    const std::string rand_dataset_path = "../datasets/" + rand_dataset_name;
    if (!(std::ifstream(rand_dataset_path))) {
        std::ofstream rand_dataset_file(rand_dataset_path);
        for (auto i = 0; i < nr_points; i++) {
            for (auto j = 0; j < 2; j++) {
                rand_dataset_file << distribution(generator) << " ";
            }
            rand_dataset_file << "\n";
        }
        rand_dataset_file.close();

        std::cout << "Random Dataset Generated: " + rand_dataset_name + "\n";
    }



}


void load_dataset(std::vector<Point> &dataset, std::ifstream &dataset_file) {
    std::cout << "Loading dataset in progress ...\n";
    std::string file_line, word;

    while (getline(dataset_file, file_line)) {

        std::istringstream string_stream(file_line);
        std::vector<std::string> row;

        while(getline(string_stream, word, ' ')){
            row.push_back(word);
        }

        dataset.emplace_back(stod(row[0]), stod(row[1]));

    }
}


std::vector<Point> generate_initial_centroids(const std::vector<Point> &dataset, const long num_clusters) {
    std::vector<Point> initial_centroids(num_clusters);
    std::vector<int> random_vector(dataset.size());
    std::iota(random_vector.begin(), random_vector.end(), 0);
    std::shuffle(random_vector.begin(), random_vector.end(), std::mt19937(std::random_device()()));
    for (auto i = 0; i < num_clusters; i++) {
        initial_centroids[i] = dataset[random_vector[i]];
        initial_centroids[i].cluster = i;
    }
    return initial_centroids;
}


void
transform_into_array(const std::vector<Point> &data, const int num_rows, const int num_columns, double *array) {
    for (auto i = 0; i < num_rows; i++) {
        array[i * num_columns] = data[i].x;
        array[i * num_columns + 1] = data[i].y;
        array[i * num_columns + 2] = data[i].cluster;

        /*
        std::cout << array[i * num_columns] << std::endl;
        std::cout << array[i * num_columns + 1] << std::endl;
        std::cout << array[i * num_columns + 2] << std::endl;
*/

    }
}

void transform_from_array(const double *array, std::vector<Point> vector, const int num_elements, const int num_dimensions){
    double x, y = 0;
    int c = 0;
    for(auto i = 0; i < num_elements; i++){
        x = array[i * num_dimensions + 0];
        y = array[i * num_dimensions + 1];
        c = array[i * num_dimensions + 2];
        //std::cout << x << "' " << y << "' " << c << std::endl;
        vector.emplace_back(x, y, c);
    }
}

void print_array(const double *array, const int num_elements, const int num_dimensions){
    double x, y = 0;
    int c = 0;
    for(auto i = 0; i < num_elements; i++){
        x = array[i * num_dimensions + 0];
        y = array[i * num_dimensions + 1];
        c = array[i * num_dimensions + 2];
        std::cout << x << "' " << y << "' " << c << std::endl;

    }
}



void draw_chart_gnu(vector<Point> &points){

    ofstream outfile("data.txt");

    for(auto point : points){

        outfile << point.x << " " << point.y << " " << point.cluster<< std::endl;

    }

    outfile.close();
    system("gnuplot -p -e \"plot 'data.txt' using 1:2:3 with points palette notitle\"");
    remove("data.txt");

}


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
    transform_into_array(final_centroids, num_clusters, num_dimensions, testArrayCentroids);
    print_array(testArrayCentroids, num_clusters, num_dimensions);

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
    transform_into_array(final_centroids, num_clusters, num_dimensions, testArrayCentroids);
    print_array(testArrayCentroids, num_clusters, num_dimensions);

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

    transform_into_array(dataset, num_points, num_dimensions, host_dataset);
    transform_into_array(initial_centroids, num_clusters, num_dimensions, host_centroids);

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
    print_array(host_centroids, num_clusters, num_dimensions);

    transform_from_array(host_dataset, final_dataset, num_points, num_dimensions);
    transform_from_array(host_centroids, final_centroids, num_clusters, num_dimensions);


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
