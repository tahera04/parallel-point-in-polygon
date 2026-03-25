#include "../include/spatial-index.h"
#include <memory>
#include <algorithm>

using namespace std;

struct Entry {
    int polygonIndex;
    BoundingBox box;
};

struct QuadNode {
    BoundingBox region;
    vector<Entry> entries;
    unique_ptr<QuadNode> nw, ne, sw, se;
    bool divided = false;
    int depth;

    QuadNode(const BoundingBox& r, int d) : region(r), depth(d) {}
};

class Quadtree {
private:
    unique_ptr<QuadNode> root;
    int capacity;
    int maxDepth;

    bool boxContainsPoint(const BoundingBox& box, const Point& pt) const {
        return pt.x >= box.min_x &&
               pt.x <= box.max_x &&
               pt.y >= box.min_y &&
               pt.y <= box.max_y;
    }

    bool boxIntersectsBox(const BoundingBox& a, const BoundingBox& b) const {
        return !(a.max_x < b.min_x ||
                 b.max_x < a.min_x ||
                 a.max_y < b.min_y ||
                 b.max_y < a.min_y);
    }

    bool boxContainsBox(const BoundingBox& outer, const BoundingBox& inner) const {
        return inner.min_x >= outer.min_x &&
               inner.max_x <= outer.max_x &&
               inner.min_y >= outer.min_y &&
               inner.max_y <= outer.max_y;
    }

    void subdivide(QuadNode* node) {
        double mid_x = (node->region.min_x + node->region.max_x) / 2.0;
        double mid_y = (node->region.min_y + node->region.max_y) / 2.0;

        node->nw = make_unique<QuadNode>(
            BoundingBox{node->region.min_x, mid_x, mid_y, node->region.max_y},
            node->depth + 1
        );
        node->ne = make_unique<QuadNode>(
            BoundingBox{mid_x, node->region.max_x, mid_y, node->region.max_y},
            node->depth + 1
        );
        node->sw = make_unique<QuadNode>(
            BoundingBox{node->region.min_x, mid_x, node->region.min_y, mid_y},
            node->depth + 1
        );
        node->se = make_unique<QuadNode>(
            BoundingBox{mid_x, node->region.max_x, node->region.min_y, mid_y},
            node->depth + 1
        );

        node->divided = true;
    }

    QuadNode* fittingChild(QuadNode* node, const BoundingBox& box) const {
        if (!node->divided) return nullptr;

        if (boxContainsBox(node->nw->region, box)) return node->nw.get();
        if (boxContainsBox(node->ne->region, box)) return node->ne.get();
        if (boxContainsBox(node->sw->region, box)) return node->sw.get();
        if (boxContainsBox(node->se->region, box)) return node->se.get();

        return nullptr;
    }

    void insert(QuadNode* node, int polygonIndex, const BoundingBox& box) {
        if (!boxIntersectsBox(node->region, box)) return;

        if (!node->divided &&
            ((int)node->entries.size() < capacity || node->depth >= maxDepth)) {
            node->entries.push_back({polygonIndex, box});
            return;
        }

        if (!node->divided) {
            subdivide(node);

            vector<Entry> oldEntries = node->entries;
            node->entries.clear();

            for (const Entry& e : oldEntries) {
                QuadNode* child = fittingChild(node, e.box);
                if (child) {
                    insert(child, e.polygonIndex, e.box);
                } else {
                    node->entries.push_back(e);
                }
            }
        }

        QuadNode* child = fittingChild(node, box);
        if (child) {
            insert(child, polygonIndex, box);
        } else {
            node->entries.push_back({polygonIndex, box});
        }
    }

    void query(QuadNode* node, const Point& pt, vector<int>& result) const {
        if (!node || !boxContainsPoint(node->region, pt)) return;

        for (const Entry& e : node->entries) {
            if (boxContainsPoint(e.box, pt)) {
                result.push_back(e.polygonIndex);
            }
        }

        if (!node->divided) return;

        query(node->nw.get(), pt, result);
        query(node->ne.get(), pt, result);
        query(node->sw.get(), pt, result);
        query(node->se.get(), pt, result);
    }

public:
    Quadtree(const BoundingBox& world, int cap = 4, int maxD = 8)
        : capacity(cap), maxDepth(maxD) {
        root = make_unique<QuadNode>(world, 0);
    }

    void insert(int polygonIndex, const BoundingBox& box) {
        insert(root.get(), polygonIndex, box);
    }

    vector<int> query(const Point& pt) const {
        vector<int> result;
        query(root.get(), pt, result);
        return result;
    }
};

BoundingBox computeWorldBoundingBox(const vector<Polygon>& polygons) {
    if (polygons.empty()) {
        return BoundingBox{0, 0, 0, 0};
    }

    BoundingBox world = polygons[0].bbox;

    for (const Polygon& poly : polygons) {
        world.min_x = min(world.min_x, poly.bbox.min_x);
        world.max_x = max(world.max_x, poly.bbox.max_x);
        world.min_y = min(world.min_y, poly.bbox.min_y);
        world.max_y = max(world.max_y, poly.bbox.max_y);
    }

    return world;
}