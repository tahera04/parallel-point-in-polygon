#ifndef INTEGRATION_H
#define INTEGRATION_H

#include "structures.h"
#include <vector>

using namespace std;

// Classifies a set of query points against a set of polygons using the full pipeline:
// 1. Spatial Index Query (Quadtree)
// 2. Bounding Box Filtering
// 3. Ray Casting Algorithm

vector<int> classifyPoints(vector<Polygon>& polygons, const vector<Point>& queryPoints);

#endif
