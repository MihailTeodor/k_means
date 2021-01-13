#include <random>
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include "utilities.h"


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


void transform_into_array(const std::vector<Point> &data, const int num_rows, double *array) {
    for (auto i = 0; i < num_rows; i++) {
        array[i + (num_rows * 0)] = data[i].x;
        array[i + (num_rows * 1)] = data[i].y;
        array[i + (num_rows * 2)] = data[i].cluster;
    }
}


void transform_from_array(const double *array, std::vector<Point> vector, const int num_elements){
    double x, y = 0;
    int c = 0;
    for(auto i = 0; i < num_elements; i++){
        x = array[i + (num_elements * 0)];
        y = array[i + (num_elements * 1)];
        c = array[i + (num_elements * 2)];
        //std::cout << x << "' " << y << "' " << c << std::endl;
        vector.emplace_back(x, y, c);
    }
}


void print_array(const double *array, const int num_elements){
    double x, y = 0;
    int c = 0;
    for(auto i = 0; i < num_elements; i++){
        x = array[i + (num_elements * 0)];
        y = array[i + (num_elements * 1)];
        c = array[i + (num_elements * 2)];
        std::cout << x << "' " << y << "' " << c << std::endl;

    }
}


void draw_chart_gnu(std::vector<Point> &points){

    std::ofstream outfile("data.txt");

    for(auto point : points){

        outfile << point.x << " " << point.y << " " << point.cluster<< std::endl;

    }

    outfile.close();
    system("gnuplot -p -e \"plot 'data.txt' using 1:2:3 with points palette notitle\"");
    remove("data.txt");

}