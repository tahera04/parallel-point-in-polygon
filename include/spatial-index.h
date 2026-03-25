#ifndef SPATIAL_INDEX_H
#define SPATIAL_INDEX_H

#include "structures.h"
#include <vector>

class Quadtree {
public:
    Quadtree(const BoundingBox& world, int capacity = 4, int maxDepth = 8);

    void insert(int polygonIndex, const BoundingBox& box);

    std::vector<int> query(const Point& pt) const;
};

BoundingBox computeWorldBoundingBox(const std::vector<Polygon>& polygons);

#endif