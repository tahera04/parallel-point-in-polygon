#include "../include/bounding-box.h"
#include <limits>
#include <stdexcept>

BoundingBox computeBoundingBox(const vector<Point>& vertices) {
    if (vertices.empty()) {
        throw invalid_argument("Cannot compute bounding box of empty polygon.");
    }

    BoundingBox box;
    box.min_x = numeric_limits<double>::max();
    box.max_x = numeric_limits<double>::lowest();
    box.min_y = numeric_limits<double>::max();
    box.max_y = numeric_limits<double>::lowest();

    for (const Point& p : vertices) {
        if (p.x < box.min_x) box.min_x = p.x;
        if (p.x > box.max_x) box.max_x = p.x;
        if (p.y < box.min_y) box.min_y = p.y;
        if (p.y > box.max_y) box.max_y = p.y;
    }

    return box;
}

void assignBoundingBox(Polygon& poly) {
    poly.bbox = computeBoundingBox(poly.outer);
}

bool pointInsideBoundingBox(const Point& pt, const BoundingBox& box) {
    return pt.x >= box.min_x &&
           pt.x <= box.max_x &&
           pt.y >= box.min_y &&
           pt.y <= box.max_y;
}