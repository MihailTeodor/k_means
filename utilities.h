#ifndef K_MEANS_UTILITIES_H
#define K_MEANS_UTILITIES_H

#include "point.h"

void generate_random_dataset(int nr_points);

void load_dataset(std::vector<Point> &dataset, std::ifstream &dataset_file);

std::vector<Point> generate_initial_centroids(const std::vector<Point> &dataset, const long num_clusters);

void transform_into_array(const std::vector<Point> &data, const int num_rows, double *array);

void transform_from_array(const double *array, std::vector<Point> vector, const int num_elements);

void print_array(const double *array, const int num_elements);

void draw_chart_gnu(std::vector<Point> &points);

#endif //K_MEANS_UTILITIES_H
