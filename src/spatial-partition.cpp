#include "../include/spatial-partition.h"
#include <omp.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>

using namespace std;

// Forward declaration - implemented in ray-casting.cpp
bool isPointInsidePolygon(Point p, Polygon& poly);

// ============================================================
//  GridPartition implementation
// ============================================================

GridPartition::GridPartition(const BoundingBox& world, int rows, int cols)
    : world(world), rows(rows), cols(cols)
{
    if (rows <= 0 || cols <= 0)
        throw invalid_argument("Grid dimensions must be positive.");

    // Add a tiny epsilon so points exactly on the max boundary
    // still map to a valid cell index.
    double eps = 1e-9;
    cellWidth  = (world.max_x - world.min_x + eps) / cols;
    cellHeight = (world.max_y - world.min_y + eps) / rows;

    // Pre-allocate all cells
    cells.resize(rows * cols);
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            cells[r * cols + c].row = r;
            cells[r * cols + c].col = c;
        }
    }
}

int GridPartition::getCellIndex(const Point& p) const {
    // Points outside the world are placed in the nearest border cell
    // (clamp) so they still get classified rather than dropped.
    int col = (int)floor((p.x - world.min_x) / cellWidth);
    int row = (int)floor((p.y - world.min_y) / cellHeight);

    col = max(0, min(cols - 1, col));
    row = max(0, min(rows - 1, row));

    return row * cols + col;
}

void GridPartition::partition(const vector<Point>& points) {
    // Clear any previous contents
    for (auto& cell : cells)
        cell.pointIndices.clear();

    for (int i = 0; i < (int)points.size(); i++) {
        int idx = getCellIndex(points[i]);
        cells[idx].pointIndices.push_back(i);
    }
}

// ============================================================
//  Strategy 1: Basic parallel loop
// ============================================================

vector<int> classifyPointsParallel(
    vector<Polygon>& polygons,
    const Quadtree& spatialIndex,
    const vector<Point>& queryPoints,
    int numThreads)
{
    int n = (int)queryPoints.size();
    vector<int> results(n, -1);

    // The Quadtree and polygons vector are read-only here, so
    // concurrent access from multiple threads is safe.
    omp_set_num_threads(numThreads);

    #pragma omp parallel for schedule(dynamic, 512)
    for (int i = 0; i < n; i++) {
        // Stage 1: Spatial index query - O(log N)
        vector<int> candidates = spatialIndex.query(queryPoints[i]);

        // Stage 2: Exact ray-casting test on candidates only
        for (int candidateIndex : candidates) {
            if (isPointInsidePolygon(queryPoints[i], polygons[candidateIndex])) {
                results[i] = polygons[candidateIndex].id;
                break;
            }
        }
    }

    return results;
}

// ============================================================
//  Strategy 2: Grid-partitioned parallel classification
// ============================================================

vector<int> classifyPointsGridPartitioned(
    vector<Polygon>& polygons,
    const Quadtree& spatialIndex,
    const vector<Point>& queryPoints,
    int gridRows,
    int gridCols,
    int numThreads)
{
    int n = (int)queryPoints.size();
    vector<int> results(n, -1);

    // Step 1: Compute world bounding box that covers all query points.
    // We expand it slightly to guarantee every point lands in a cell.
    BoundingBox world;
    world.min_x = world.min_y =  1e18;
    world.max_x = world.max_y = -1e18;
    for (const Point& p : queryPoints) {
        world.min_x = min(world.min_x, p.x);
        world.max_x = max(world.max_x, p.x);
        world.min_y = min(world.min_y, p.y);
        world.max_y = max(world.max_y, p.y);
    }

    // Step 2: Partition query points into grid cells.
    GridPartition grid(world, gridRows, gridCols);
    grid.partition(queryPoints);

    int totalCells = gridRows * gridCols;

    // Step 3: Process cells in parallel.
    // Dynamic scheduling ensures that a thread that finishes a
    // sparse cell immediately picks up the next pending cell, which
    // naturally handles spatial skew without manual load balancing.
    omp_set_num_threads(numThreads);

    #pragma omp parallel for schedule(dynamic, 1)
    for (int cellIdx = 0; cellIdx < totalCells; cellIdx++) {
        const GridCell& cell = grid.cells[cellIdx];

        // Each cell is owned exclusively by one thread at a time,
        // so writing results[pointIdx] is race-free.
        for (int pointIdx : cell.pointIndices) {
            // Stage 1: Spatial index query
            vector<int> candidates = spatialIndex.query(queryPoints[pointIdx]);

            // Stage 2: Ray casting
            for (int candidateIndex : candidates) {
                if (isPointInsidePolygon(queryPoints[pointIdx], polygons[candidateIndex])) {
                    results[pointIdx] = polygons[candidateIndex].id;
                    break;
                }
            }
        }
    }

    return results;
}
