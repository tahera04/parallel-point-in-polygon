#include "../include/integration.h"
#include "../include/spatial-index.h"
#include "../include/bounding-box.h"
#include <iostream>

using namespace std;

bool isPointInsidePolygon(Point p, Polygon& poly);

//   This function combines all components of the point-in-polygon system

vector<int> classifyPoints(vector<Polygon>& polygons, const vector<Point>& queryPoints) {
    vector<int> results;
    
    BoundingBox worldBBox = computeWorldBoundingBox(polygons);
    Quadtree spatialIndex(worldBBox);
    for (int i = 0; i < polygons.size(); i++) {
        spatialIndex.insert(i, polygons[i].bbox);
    }
    
    // Here we classify each query point
    for (const Point& queryPoint : queryPoints) {
        int foundPolygonID = -1;  
        
        // Now we query spatial index to get candidate polygons
        vector<int> candidates = spatialIndex.query(queryPoint);
        
        // And now check each candidate polygon
        for (int candidateIndex : candidates) {

            // Check if the point is inside the bounding box
            if (!pointInsideBoundingBox(queryPoint, polygons[candidateIndex].bbox)) {
                continue;
            }
            
            // Ray Casting Algo/filter
            if (isPointInsidePolygon(queryPoint, polygons[candidateIndex])) {
                foundPolygonID = polygons[candidateIndex].id;
                break;  // Point found, stop checking other candidates
            }
        }
        
        results.push_back(foundPolygonID);
    }
    
    return results;
}
