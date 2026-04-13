#ifndef STRUCTURES_H
#define STRUCTURES_H

#include <vector>
using namespace std;

struct Point {
    double x;
    double y;
};

struct BoundingBox {
    double min_x;
    double max_x;
    double min_y;
    double max_y;
};

struct Polygon {
    int id;
    vector<Point> outer;
    vector<vector<Point>> holes;
    BoundingBox bbox;
};

struct MultiPolygon {
    int id;
    vector<Polygon> parts;
};

#endif