#ifndef INTEGRATION_H
#define INTEGRATION_H

#include "structures.h"
#include "spatial-index.h"
#include <vector>

using namespace std;

// Classifies a set of query points against a set of polygons using the full pipeline:
// 1. Spatial Index Query (Quadtree) - returns candidates whose bboxes overlap
// 2. Ray Casting Algorithm - exact point-in-polygon test
// 
// Note: Quadtree must be pre-built from polygon bounding boxes

vector<int> classifyPoints(vector<Polygon>& polygons, const Quadtree& spatialIndex, const vector<Point>& queryPoints);

#endif
