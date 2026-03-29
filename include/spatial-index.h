#ifndef SPATIAL_INDEX_H
#define SPATIAL_INDEX_H

#include "structures.h"
#include <vector>
#include <memory>

struct Entry {
    int polygonIndex;
    BoundingBox box;
};

struct QuadNode {
    BoundingBox region;
    std::vector<Entry> entries;
    std::unique_ptr<QuadNode> nw, ne, sw, se;
    bool divided = false;
    int depth;

    QuadNode(const BoundingBox& r, int d) : region(r), depth(d) {}
};

class Quadtree {
private:
    std::unique_ptr<QuadNode> root;
    int capacity;
    int maxDepth;

    bool boxContainsPoint(const BoundingBox& box, const Point& pt) const;
    bool boxIntersectsBox(const BoundingBox& a, const BoundingBox& b) const;
    bool boxContainsBox(const BoundingBox& outer, const BoundingBox& inner) const;
    void subdivide(QuadNode* node);
    QuadNode* fittingChild(QuadNode* node, const BoundingBox& box) const;
    void insert(QuadNode* node, int polygonIndex, const BoundingBox& box);
    void query(QuadNode* node, const Point& pt, std::vector<int>& result) const;

public:
    Quadtree(const BoundingBox& world, int capacity = 4, int maxDepth = 8);
    void insert(int polygonIndex, const BoundingBox& box);
    std::vector<int> query(const Point& pt) const;
};

BoundingBox computeWorldBoundingBox(const std::vector<Polygon>& polygons);

#endif