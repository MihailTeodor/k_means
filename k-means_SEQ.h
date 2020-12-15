#ifndef K_MEANS_K_MEANS_SEQ_H
#define K_MEANS_K_MEANS_SEQ_H

#include "point.h"
#include <tuple>
#include <vector>

using namespace std;

void draw_chart_gnu(vector<Point> &points);

void assign_clusters(vector<Point>* points, vector<Point>* centroids, vector<Point>* centroids_details);

bool update_centroids(vector<Point>* centroids, vector<Point>* centroids_details);

tuple<vector<Point>, vector<Point>> k_means_SEQ(vector<Point> points, vector<Point> centroids, int max_epochs);

#endif //K_MEANS_K_MEANS_SEQ_H
