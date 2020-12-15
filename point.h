#ifndef K_MEANS_POINT_H
#define K_MEANS_POINT_H

#include <cmath>

struct Point {

    double x, y;
    int cluster;

    Point() : x(0.0), y(0.0), cluster(0){}
    Point(double x, double y, int c = 0) : x(x), y(y), cluster(c){}

};

inline double distance(Point p1, Point p2){
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

#endif //K_MEANS_POINT_H
