#include "../include/integration.h"
#include "../include/spatial-index.h"
#include "../include/bounding-box.h"
#include <iostream>

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
