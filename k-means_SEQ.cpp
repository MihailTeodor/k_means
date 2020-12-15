#include "k-means_SEQ.h"

void assign_clusters(vector<Point>* points, vector<Point>* centroids, vector<Point>* centroids_details){

    for(auto & i : *points){
        int bestCluster = 0;
        auto minDist = __DBL_MAX__;
        for(int j = 0; j < (*centroids).size(); j++){
            double dist = distance(i, (*centroids)[j]);
            if (dist <= minDist){
                minDist = dist;
                bestCluster = j;
            }
        }

        i.cluster = bestCluster;
        (*centroids_details)[bestCluster].x += i.x;
        (*centroids_details)[bestCluster].y += i.y;
        (*centroids_details)[bestCluster].cluster += 1;

    }
}


bool update_centroids(vector<Point>* centroids, vector<Point>* centroids_details){

    double centroid_x_new, centroid_y_new, centroid_x_old, centroid_y_old = 0;
    bool iterate = false;

    for(int i = 0; i < centroids->size(); i++){
        centroid_x_new = (*centroids_details)[i].x / (*centroids_details)[i].cluster;
        centroid_y_new = (*centroids_details)[i].y / (*centroids_details)[i].cluster;

        centroid_x_old = (*centroids)[i].x;
        centroid_y_old = (*centroids)[i].y;

        if(centroid_x_old != centroid_x_new || centroid_y_old != centroid_y_new) {
            iterate = true;
            (*centroids)[i].x = centroid_x_new;
            (*centroids)[i].y = centroid_y_new;
        }

        (*centroids_details)[i].x = 0;
        (*centroids_details)[i].y = 0;
        (*centroids_details)[i].cluster = 0;

    }

    return iterate;

}


tuple<std::vector<Point>, vector<Point>> k_means_SEQ(vector<Point> points, vector<Point> centroids, int max_epochs){

    vector<Point> centroids_details (centroids.size());
    bool iterate;
    int iteration = 0;

    do{
        iteration ++;

        assign_clusters(&points, &centroids, &centroids_details);

        iterate = update_centroids(&centroids, &centroids_details);

    } while(iterate && iteration < max_epochs);

    return {points, centroids};

}