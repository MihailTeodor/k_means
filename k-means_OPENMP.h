#ifndef K_MEANS_K_MEANS_OPENMP_H
#define K_MEANS_K_MEANS_OPENMP_H

#include <vector>
#include <tuple>
#include "point.h"

using namespace std;

tuple<vector<Point>, std::vector<Point>> k_means_OPENMP(vector<Point> points, vector<Point> centroids, int max_epochs, int nr_threads);

bool update_centroids_OPENMP(vector<Point>* centroids, vector<Point>* centroids_details);

void assign_clusters_OPENMP(vector<Point>* points, vector<Point>* centroids, vector<Point>* centroids_details);

#endif //K_MEANS_K_MEANS_OPENMP_H
