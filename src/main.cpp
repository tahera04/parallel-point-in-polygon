#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <random>
#include "include/structures.h"
#include "include/dataset.h"
#include "include/bounding-box.h"
#include "include/spatial-index.h"
#include "include/integration.h"
#include "include/spatial-partition.h"

using namespace std;

// Generate uniform points in memory (no file I/O)
vector<Point> generateUniformPointsInMemory(long count, double minX, double maxX, double minY, double maxY) {
    vector<Point> points;
    points.reserve(count);
    mt19937 gen(42);
    uniform_real_distribution<double> distX(minX, maxX);
    uniform_real_distribution<double> distY(minY, maxY);
    for (long i = 0; i < count; i++)
        points.push_back({distX(gen), distY(gen)});
    return points;
}

// Generate clustered points in memory (no file I/O)
vector<Point> generateClusteredPointsInMemory(long count, double minX, double maxX, double minY, double maxY) {
    vector<Point> points;
    points.reserve(count);
    mt19937 gen(42);
    long pointsPerCluster = count / 5;
    double clusterWidth  = (maxX - minX) / 5;
    double clusterHeight = (maxY - minY) / 2;
    for (int cluster = 0; cluster < 5; cluster++) {
        double centerX = minX + (cluster + 0.5) * clusterWidth;
        double centerY = minY + (cluster % 2 == 0 ? 0.25 : 0.75) * (maxY - minY);
        normal_distribution<double> distX(centerX, clusterWidth / 4);
        normal_distribution<double> distY(centerY, clusterHeight / 4);
        for (long i = 0; i < pointsPerCluster; i++)
            points.push_back({distX(gen), distY(gen)});
    }
    return points;
}

int main() {
    cout << "========================================\n";
    cout << "  POINT-IN-POLYGON BASELINE BENCHMARK\n";
    cout << "  Milestone 1 - Task 9: Performance Measurement\n";
    cout << "========================================\n\n";

    // ===== PHASE 0: Setup =====
    cout << "[PHASE 0] Loading polygons...\n";
    vector<Polygon> polygons = loadPolygons("data/polygons.txt");
    cout << "  > Loaded " << polygons.size() << " polygons\n\n";

    // ===== PHASE 1: Bounding Boxes =====
    cout << "[PHASE 1] Computing bounding boxes...\n";
    auto t0 = chrono::high_resolution_clock::now();
    for (auto& poly : polygons) assignBoundingBox(poly);
    auto t1 = chrono::high_resolution_clock::now();
    cout << "  > Done in " << chrono::duration_cast<chrono::milliseconds>(t1-t0).count() << " ms\n\n";

    // ===== PHASE 2: Build Spatial Index =====
    cout << "[PHASE 2] Building Quadtree spatial index...\n";
    t0 = chrono::high_resolution_clock::now();
    BoundingBox worldBounds = computeWorldBoundingBox(polygons);
    Quadtree spatialIndex(worldBounds);
    for (int i = 0; i < (int)polygons.size(); i++)
        spatialIndex.insert(i, polygons[i].bbox);
    t1 = chrono::high_resolution_clock::now();
    cout << "  > Done in " << chrono::duration_cast<chrono::milliseconds>(t1-t0).count() << " ms\n\n";

    // ===== PHASE 3: M1 Baseline (sequential, batch) =====
    const long M1_TOTAL    = 5000000; // 5M points per distribution
    const long BATCH_SIZE  = 1000000;

    auto runBatch = [&](bool uniform) -> pair<double, long> {
        long inside = 0;
        auto start  = chrono::high_resolution_clock::now();
        for (long done = 0; done < M1_TOTAL; ) {
            long cur  = min(BATCH_SIZE, M1_TOTAL - done);
            auto batch = uniform
                ? generateUniformPointsInMemory(cur, 0, 100, 0, 100)
                : generateClusteredPointsInMemory(cur, 0, 100, 0, 100);
            auto res  = classifyPoints(polygons, spatialIndex, batch);
            for (int r : res) if (r != -1) inside++;
            done += cur;
        }
        auto end = chrono::high_resolution_clock::now();
        return { chrono::duration<double,milli>(end - start).count(), inside };
    };

    cout << "========================================================================\n";
    cout << " M1 BASELINE: " << M1_TOTAL/1000000 << "M Uniform points (sequential)\n";
    cout << "========================================================================\n";
    auto [uMs, uIn] = runBatch(true);
    cout << "  Time : " << fixed << setprecision(1) << uMs << " ms\n";
    cout << "  Inside: " << uIn << " / " << M1_TOTAL << "\n\n";

    cout << "========================================================================\n";
    cout << " M1 BASELINE: " << M1_TOTAL/1000000 << "M Clustered points (sequential)\n";
    cout << "========================================================================\n";
    auto [cMs, cIn] = runBatch(false);
    cout << "  Time : " << fixed << setprecision(1) << cMs << " ms\n";
    cout << "  Inside: " << cIn << " / " << M1_TOTAL << "\n\n";

    // ====================================================================
    //  MILESTONE 2 - TASK 2: SPATIAL PARTITIONING STRATEGIES
    // ====================================================================
    cout << "\n########################################################################\n";
    cout << "#     MILESTONE 2 - TASK 2: SPATIAL PARTITIONING STRATEGIES           #\n";
    cout << "########################################################################\n\n";

    const long M2_COUNT = 2000000; // 2M points

    cout << "[M2 SETUP] Generating " << M2_COUNT/1000000 << "M uniform points...\n";
    vector<Point> m2Uniform   = generateUniformPointsInMemory(M2_COUNT, 0, 100, 0, 100);
    cout << "[M2 SETUP] Generating " << M2_COUNT/1000000 << "M clustered points...\n\n";
    vector<Point> m2Clustered = generateClusteredPointsInMemory(M2_COUNT, 0, 100, 0, 100);

    // Time a classification call; return elapsed ms
    auto timeIt = [](auto fn) -> double {
        auto s = chrono::high_resolution_clock::now();
        fn();
        auto e = chrono::high_resolution_clock::now();
        return chrono::duration<double,milli>(e - s).count();
    };

    // --- Sequential baseline ---
    cout << "--------------------------------------------------------------------\n";
    cout << " [SEQ] Sequential baseline\n";
    cout << "--------------------------------------------------------------------\n";
    vector<int> seqU, seqC;
    double seqUms = timeIt([&]{ seqU = classifyPoints(polygons, spatialIndex, m2Uniform); });
    double seqCms = timeIt([&]{ seqC = classifyPoints(polygons, spatialIndex, m2Clustered); });
    cout << "  Uniform   | " << fixed << setprecision(1) << seqUms << " ms\n";
    cout << "  Clustered | " << fixed << setprecision(1) << seqCms << " ms\n\n";

    // --- Strategy 1: Basic OpenMP parallel loop ---
    cout << "--------------------------------------------------------------------\n";
    cout << " [PAR-BASIC] Strategy 1: Basic OpenMP parallel loop\n";
    cout << "--------------------------------------------------------------------\n";
    for (int t : {2, 4}) {
        double pUms = timeIt([&]{ classifyPointsParallel(polygons, spatialIndex, m2Uniform, t); });
        double pCms = timeIt([&]{ classifyPointsParallel(polygons, spatialIndex, m2Clustered, t); });
        cout << "  " << t << " threads | Uniform "
             << fixed << setprecision(1) << pUms << " ms (x" << setprecision(2) << seqUms/pUms << ")"
             << "  |  Clustered "
             << setprecision(1) << pCms << " ms (x" << setprecision(2) << seqCms/pCms << ")\n";
    }
    cout << "\n";

    // --- Strategy 2: Grid-partitioned parallel (4x4 tiles) ---
    const int GR = 4, GC = 4;
    cout << "--------------------------------------------------------------------\n";
    cout << " [PAR-GRID] Strategy 2: Grid-partitioned parallel (" << GR << "x" << GC << " tiles)\n";
    cout << "--------------------------------------------------------------------\n";
    for (int t : {2, 4}) {
        double gUms = timeIt([&]{ classifyPointsGridPartitioned(polygons, spatialIndex, m2Uniform, GR, GC, t); });
        double gCms = timeIt([&]{ classifyPointsGridPartitioned(polygons, spatialIndex, m2Clustered, GR, GC, t); });
        cout << "  " << t << " threads | Uniform "
             << fixed << setprecision(1) << gUms << " ms (x" << setprecision(2) << seqUms/gUms << ")"
             << "  |  Clustered "
             << setprecision(1) << gCms << " ms (x" << setprecision(2) << seqCms/gCms << ")\n";
    }
    cout << "\n";

    // --- Correctness check ---
    cout << "--------------------------------------------------------------------\n";
    cout << " [VERIFY] Correctness: Sequential vs Grid-Partitioned (4 threads)\n";
    cout << "--------------------------------------------------------------------\n";
    vector<int> gridU = classifyPointsGridPartitioned(polygons, spatialIndex, m2Uniform,   GR, GC, 4);
    vector<int> gridC = classifyPointsGridPartitioned(polygons, spatialIndex, m2Clustered, GR, GC, 4);

    int mmU = 0, mmC = 0;
    for (int i = 0; i < M2_COUNT; i++) { if (seqU[i] != gridU[i]) mmU++; }
    for (int i = 0; i < M2_COUNT; i++) { if (seqC[i] != gridC[i]) mmC++; }

    if (mmU == 0 && mmC == 0) {
        cout << "  [PASS] All " << M2_COUNT << " uniform results match.\n";
        cout << "  [PASS] All " << M2_COUNT << " clustered results match.\n";
    } else {
        cout << "  [FAIL] Uniform mismatches:   " << mmU << "\n";
        cout << "  [FAIL] Clustered mismatches: " << mmC << "\n";
    }

    return 0;
}
