#include "../include/integration.h"
#include "../include/spatial-index.h"
#include "../include/bounding-box.h"
#include <iostream>
#include <omp.h>

using namespace std;

bool isPointInsidePolygon(Point p, Polygon& poly);

//   This function combines all components of the point-in-polygon system
//   
//   Pipeline:
//   1. Spatial Index Query (Quadtree) returns candidates whose bboxes overlap the point
//   2. Ray Casting Algorithm performs exact point-in-polygon test
//   
//   NOTE: Quadtree must be pre-built and passed in to avoid rebuilding for each call

vector<int> classifyPoints(vector<Polygon>& polygons, const Quadtree& spatialIndex, const vector<Point>& queryPoints) {
    vector<int> results;
    results.reserve(queryPoints.size());
    
    // For each query point, classify it against the polygon set
    for (const Point& queryPoint : queryPoints) {
        int foundPolygonID = -1;  
        
        // STAGE 1: Spatial Index Query
        // Returns only polygons whose bounding boxes overlap the point
        // This is O(log N) due to quadtree spatial partitioning
        vector<int> candidates = spatialIndex.query(queryPoint);
        
        // STAGE 2: Ray Casting Algorithm
        // Performs exact point-in-polygon test on candidates only
        // No redundant bbox check needed - spatial index already filtered by bbox
        for (int candidateIndex : candidates) {
            // Ray Casting Algorithm - exact test
            if (isPointInsidePolygon(queryPoint, polygons[candidateIndex])) {
                foundPolygonID = polygons[candidateIndex].id;
                break;  // Point found, stop checking other candidates
            }
        }
        
        results.push_back(foundPolygonID);
    }
    
    return results;
}

// ─────────────────────────────────────────────────────────────────────────────
// Milestone 2 - Task 1: Parallel Point Processing
// Parallelizes the classification loop using OpenMP.
//
// Thread safety:
//   - results[] is pre-allocated; each thread writes to its own unique index.
//   - spatialIndex.query() is const; each call allocates its own local vector.
//   - isPointInsidePolygon() only reads polygon data (no writes).
//   - polygons vector is never modified during classification.
//
// Compile: g++ -fopenmp ...  (GCC/MinGW)
//          cl /openmp ...    (MSVC)
// ─────────────────────────────────────────────────────────────────────────────
vector<int> classifyPointsParallel(vector<Polygon>& polygons, const Quadtree& spatialIndex, const vector<Point>& queryPoints) {
    int n = (int)queryPoints.size();
    vector<int> results(n, -1);  // Pre-allocate; each thread writes to unique index

    #pragma omp parallel for schedule(dynamic, 1024)
    for (int i = 0; i < n; i++) {
        int foundPolygonID = -1;

        // STAGE 1: Spatial Index Query (thread-local: returns new vector each call)
        vector<int> candidates = spatialIndex.query(queryPoints[i]);

        // STAGE 2: Ray Casting on candidates only
        for (int candidateIndex : candidates) {
            if (isPointInsidePolygon(queryPoints[i], polygons[candidateIndex])) {
                foundPolygonID = polygons[candidateIndex].id;
                break;
            }
        }

        results[i] = foundPolygonID;  // Safe: unique index per thread
    }

    return results;
}
