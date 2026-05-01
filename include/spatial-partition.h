#ifndef SPATIAL_PARTITION_H
#define SPATIAL_PARTITION_H

#include "structures.h"
#include "spatial-index.h"
#include <vector>

using namespace std;



struct GridCell {
    int row;
    int col;
    vector<int> pointIndices;
};

// Grid-based spatial partition of query points.
struct GridPartition {
    int rows;
    int cols;
    BoundingBox world;
    double cellWidth;
    double cellHeight;
    vector<GridCell> cells; // laid out row-major: cells[row * cols + col]

    // Build the partition: divide world into rows x cols cells.
    GridPartition(const BoundingBox& world, int rows, int cols);

    // Assign every point in `points` to its corresponding cell.
    // Call once before using the partition for classification.
    void partition(const vector<Point>& points);

    // Return the flat cell index for a given point.
    // Returns -1 if the point lies outside the world bounds.
    int getCellIndex(const Point& p) const;
};

// ---------------------------------------------------------------
//Basic parallel loop
// ---------------------------------------------------------------
vector<int> classifyPointsParallel(
    vector<Polygon>& polygons,
    const Quadtree& spatialIndex,
    const vector<Point>& queryPoints,
    int numThreads
);

// ---------------------------------------------------------------
// Grid-partitioned parallel classification
// ---------------------------------------------------------------
vector<int> classifyPointsGridPartitioned(
    vector<Polygon>& polygons,
    const Quadtree& spatialIndex,
    const vector<Point>& queryPoints,
    int gridRows,
    int gridCols,
    int numThreads
);

#endif
