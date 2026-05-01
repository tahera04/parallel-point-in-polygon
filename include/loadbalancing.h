#ifndef LOADBALANCING_H
#define LOADBALANCING_H

#include <vector>
#include "structures.h"
#include "spatial-index.h"

using namespace std;

// Dynamic task queue based load-balanced point classification
vector<int> classifyWithDynamicQueue(
    vector<Polygon>& polygons,
    const Quadtree& spatialIndex,
    const vector<Point>& queryPoints,
    int batchSize,
    int numThreads
);

#endif