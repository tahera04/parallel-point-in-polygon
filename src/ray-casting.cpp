#include "../include/structures.h"
#include <cmath>
#include <iostream>
using namespace std;

bool onSegment(Point p, Point a, Point b) {
    double cross = (b.x - a.x)*(p.y - a.y) - (b.y - a.y)*(p.x - a.x);
    if (fabs(cross) > 1e-9) return false;

    double dot = (p.x - a.x)*(p.x - b.x) + (p.y - a.y)*(p.y - b.y);
    return dot <= 0;
}

bool doesRayIntersect(Point p, Point a, Point b) {
    if (a.y > b.y) swap(a, b);

    if (p.y == a.y || p.y == b.y)
        p.y += 1e-9;

    if (p.y < a.y || p.y > b.y)
        return false;

    if (fabs(b.y - a.y) < 1e-9)
        return false;

    double x_intersect = a.x + (p.y - a.y) * (b.x - a.x) / (b.y - a.y);

    return x_intersect > p.x;

}

bool isInsideRing(Point p, vector<Point>& ring) {
    int count = 0;
    int n = ring.size();

    for (int i = 0; i < n; i++) {
        Point a = ring[i];
        Point b = ring[(i + 1) % n];

        // edge case: on boundary
        if (onSegment(p, a, b))
            return true;

        if (doesRayIntersect(p, a, b))
            count++;
    }

    return count % 2 == 1;
}

bool isPointInsidePolygon(Point p, Polygon& poly) {
    // Step 1: check outer boundary
    if (!isInsideRing(p, poly.outer))
        return false;

    // Step 2: check holes
    for (auto& hole : poly.holes) {
        if (isInsideRing(p, hole))
            return false;
    }

    return true;
}

// int main() {
//     Polygon poly1;
//     poly1.id = 1;
//     poly1.outer = {
//         {0, 0}, {10, 0}, {10, 10}, {0, 10}
//     };

//     Polygon poly2;
//     poly2.id = 2;
//     poly2.outer = {
//         {12, 2}, {18, 2}, {18, 8}, {12, 8}
//     };

//     Polygon poly3;
//     poly3.id = 3;
//     poly3.outer = {
//         {3, 12}, {8, 15}, {5, 20}, {1, 17}
//     };

//     Point p1 = {5, 5};
//     Point p2 = {15, 5};
//     Point p3 = {4, 16};
//     Point p4 = {0, 5};
//     Point p5 = {20, 20};

//     cout << "Point (5,5) in poly1: " << isPointInsidePolygon(p1, poly1) << endl;
//     cout << "Point (15,5) in poly2: " << isPointInsidePolygon(p2, poly2) << endl;
//     cout << "Point (4,16) in poly3: " << isPointInsidePolygon(p3, poly3) << endl;
//     cout << "Point (0,5) in poly1: " << isPointInsidePolygon(p4, poly1) << endl;

//     cout << "Point (20,20) in poly1: " << isPointInsidePolygon(p5, poly1) << endl;
//     cout << "Point (20,20) in poly2: " << isPointInsidePolygon(p5, poly2) << endl;
//     cout << "Point (20,20) in poly3: " << isPointInsidePolygon(p5, poly3) << endl;

//     return 0;
// }