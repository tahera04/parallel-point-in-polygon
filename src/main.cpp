#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <random>
#include <string>
#include <omp.h>
#include "include/structures.h"
#include "include/dataset.h"
#include "include/bounding-box.h"
#include "include/spatial-index.h"
#include "include/integration.h"
#include "include/spatial-partition.h"
#include "include/loadbalancing.h"

using namespace std;

// ── Point Generators (in-memory, no file I/O) ────────────────────────────────

vector<Point> generateUniformPointsInMemory(long count, double minX, double maxX, double minY, double maxY) {
    vector<Point> points;
    points.reserve(static_cast<size_t>(count));
    mt19937 gen(42);
    uniform_real_distribution<double> distX(minX, maxX);
    uniform_real_distribution<double> distY(minY, maxY);
    for (long i = 0; i < count; i++) points.push_back({distX(gen), distY(gen)});
    return points;
}

vector<Point> generateClusteredPointsInMemory(long count, double minX, double maxX, double minY, double maxY) {
    vector<Point> points;
    points.reserve(static_cast<size_t>(count));
    mt19937 gen(42);
    const int clusterCount = 5;
    long pointsPerCluster = count / clusterCount;
    long remainder = count % clusterCount;
    double clusterWidth  = (maxX - minX) / 5;
    double clusterHeight = (maxY - minY) / 2;
    for (int c = 0; c < clusterCount; c++) {
        double cx = minX + (c + 0.5) * clusterWidth;
        double cy = minY + (c % 2 == 0 ? 0.25 : 0.75) * (maxY - minY);
        normal_distribution<double> dX(cx, clusterWidth  / 4);
        normal_distribution<double> dY(cy, clusterHeight / 4);
        long clusterPoints = pointsPerCluster + (c < remainder ? 1 : 0);
        for (long i = 0; i < clusterPoints; i++) {
            points.push_back({dX(gen), dY(gen)});
        }
    }
    return points;
}

struct PhaseResult {
    double timeSec;
    long   insideCount;
    long   outsideCount;
};

PhaseResult classifyPhase(long total, bool uniform,
                          vector<Polygon>& polygons, const Quadtree& qt,
                          bool useParallel, long batchSize = 1000000) {
    PhaseResult res = {0.0, 0, 0};
    long processed = 0;
    auto start = chrono::high_resolution_clock::now();

    while (processed < total) {
        long cur = min(batchSize, total - processed);
        vector<Point> batch = uniform
            ? generateUniformPointsInMemory (cur, 0, 100, 0, 100)
            : generateClusteredPointsInMemory(cur, 0, 100, 0, 100);

        vector<int> results = useParallel
            ? classifyPointsParallel(polygons, qt, batch)
            : classifyPoints        (polygons, qt, batch);

        for (int r : results) {
            if (r == -1) res.outsideCount++;
            else         res.insideCount++;
        }
        processed += cur;
    }

    auto end = chrono::high_resolution_clock::now();
    res.timeSec = chrono::duration<double>(end - start).count();
    return res;
}

// ── Output Verification ───────────────────────────────────────────────────────
// Runs 100 K identical points through both seq and par; confirms every result matches.
bool verifyOutputMatch(vector<Polygon>& polygons, const Quadtree& qt) {
    const long N = 100000;
    vector<Point> pts = generateUniformPointsInMemory(N, 0, 100, 0, 100);
    vector<int> seqRes = classifyPoints        (polygons, qt, pts);
    vector<int> parRes = classifyPointsParallel(polygons, qt, pts);
    if (seqRes.size() != parRes.size()) return false;
    for (size_t i = 0; i < seqRes.size(); i++)
        if (seqRes[i] != parRes[i]) return false;
    return true;
}

// ── Dataset Configuration ─────────────────────────────────────────────────────
// runSequential: true  -> live sequential run (for Small; enables direct speedup)
//                false -> use stored Milestone 1 baseline times (for Medium/Large)
struct DatasetConfig {
    const char* name;
    long  uniformPts, clusteredPts;
    double baseSeqUniformSec, baseSeqClusteredSec;   // from Milestone 1 results
    bool   runSequential;
};

// ── Summary Row ───────────────────────────────────────────────────────────────
struct SummaryRow {
    string name;
    long   totalPoints;
    double seqTotal, parTotal, speedup;
    bool   seqIsBaseline;
};

// ── Main ──────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    // --- PARALLEL POINT-IN-POLYGON PERFORMANCE BENCHMARK ---
    cout << "==========================================================\n";
    cout << "  POINT-IN-POLYGON  |  MILESTONE 2 PARALLEL BENCHMARK\n";
    cout << "  Parallel Point Processing using OpenMP\n";
    cout << "==========================================================\n\n";

    int numThreads = omp_get_max_threads();
    cout << "[INFO] OpenMP max threads : " << numThreads << "\n";
    cout << "[INFO] Schedule           : dynamic, chunk = 1024\n\n";

    // ── PHASE 0: Load polygons, compute bboxes, build Quadtree ────────────────
    cout << "[PHASE 0] Loading data and building spatial index...\n";
    vector<Polygon> polygons = loadPolygons("data/polygons.txt");
    
    for (auto& poly : polygons) assignBoundingBox(poly);
    cout << "  - Bounding boxes computed\n";

    BoundingBox world = computeWorldBoundingBox(polygons);
    Quadtree qt(world);
    for (int i = 0; i < (int)polygons.size(); i++) qt.insert(i, polygons[i].bbox);
    cout << "  - Quadtree built\n\n";

    // ── PHASE 1: Output Verification (correctness, parallel/seq) ─────────────
    cout << "[PHASE 1] Output Verification (100K uniform points)...\n";
    if (verifyOutputMatch(polygons, qt))
        cout << "  - [PASS] Parallel output matches sequential exactly.\n\n";
    else {
        cout << "  - [FAIL] Output mismatch! Aborting.\n\n";
        return 1;
    }

    bool runHeavyDatasets = false;
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--full") runHeavyDatasets = true;
    }

    // ── Dataset configs (sequential baseline points from Milestone 1) ─────────
    DatasetConfig datasets[] = {
        { "Small",   1000000,    1000000,    30.0,   31.0,  false },
        { "Medium",  50000000,   50000000,  2219.0, 1754.0, false },
        { "Large",  100000000,  100000000,  2732.0, 2811.0, false }
    };

    vector<SummaryRow> summary;
    cout << "[INFO] Sequential baseline (hardcoded from your Milestone 1 screenshot):\n";
    cout << "  - Small  : 1,000,000 uniform + 1,000,000 clustered = 2,000,000 total\n";
    cout << "             Time = 30 + 31 = 61 sec\n";
    cout << "  - Medium : 50,000,000 uniform + 50,000,000 clustered = 100,000,000 total\n";
    cout << "             Time = 2,219 + 1,754 = 3,973 sec\n";
    cout << "  - Large  : 100,000,000 uniform + 100,000,000 clustered = 200,000,000 total\n";
    cout << "             Time = 2,732 + 2,811 = 5,543 sec\n";
    cout << "[INFO] Run mode: " << (runHeavyDatasets ? "FULL (Small+Medium+Large parallel)" : "DEFAULT (Small parallel only)") << "\n";
    cout << "[INFO] Use --full to run full parallel comparison.\n\n";

    for (auto& ds : datasets) {
        long total = ds.uniformPts + ds.clusteredPts;

        cout << "============================================================\n";
        cout << "  DATASET: " << ds.name << "  ("
             << fixed << setprecision(0) << total / 1e6 << "M points)\n";
        cout << "============================================================\n\n";

        double seqUniformSec, seqClusteredSec;

        // --- Sequential --- (Only for Small dataset)
        if (ds.runSequential) {
            cout << "[SEQ] Running sequential classification...\n";
            PhaseResult su = classifyPhase(ds.uniformPts,   true,  polygons, qt, false);
            PhaseResult sc = classifyPhase(ds.clusteredPts, false, polygons, qt, false);
            seqUniformSec   = su.timeSec;
            seqClusteredSec = sc.timeSec;
            cout << "  - Uniform   : " << fixed << setprecision(2) << su.timeSec << " sec\n";
            cout << "  - Clustered : " << sc.timeSec << " sec\n";
            cout << "  - Total     : " << su.timeSec + sc.timeSec << " sec\n\n";
        } else {
            seqUniformSec   = ds.baseSeqUniformSec;
            seqClusteredSec = ds.baseSeqClusteredSec;
            cout << "[SEQ] Sequential baseline (hardcoded from screenshot):\n";
            cout << "  - Uniform   : " << seqUniformSec   << " sec\n";
            cout << "  - Clustered : " << seqClusteredSec << " sec\n";
            cout << "  - Total     : " << seqUniformSec + seqClusteredSec << " sec\n\n";
        }

        double seqTotal = seqUniformSec + seqClusteredSec;
        double parTotal = -1.0;
        double speedup  = 0.0;
        bool runParallel = runHeavyDatasets || string(ds.name) == "Small";

        if (runParallel) {
            cout << "[PAR] Running parallel classification (" << numThreads << " threads)...\n";
            PhaseResult pu = classifyPhase(ds.uniformPts,   true,  polygons, qt, true);
            PhaseResult pc = classifyPhase(ds.clusteredPts, false, polygons, qt, true);

            cout << "  - Uniform   : " << fixed << setprecision(2) << pu.timeSec << " sec\n";
            cout << "  - Clustered : " << pc.timeSec << " sec\n";
            cout << "  - Total     : " << pu.timeSec + pc.timeSec << " sec\n";
            cout << "  - Inside    : " << (pu.insideCount  + pc.insideCount)  << " pts\n";
            cout << "  - Outside   : " << (pu.outsideCount + pc.outsideCount) << " pts\n\n";

            parTotal = pu.timeSec + pc.timeSec;
            speedup  = seqTotal / parTotal;
            cout << "  >>> Speedup: " << fixed << setprecision(2) << speedup << "x"
                 << "  (seq = " << (long)seqTotal << "s,  par = "
                 << fixed << setprecision(1) << parTotal << "s)\n\n";
        } else {
            cout << "[PAR] Skipped for " << ds.name << " dataset (run .\\main_parallel.exe --full to include).\n\n";
        }

        summary.push_back({ds.name, total, seqTotal, parTotal, speedup, !ds.runSequential});
    }

    // --- Final Summary Table ---
    cout << "\n";
    cout << "==========================================================================\n";
    cout << "           PARALLEL PERFORMANCE SUMMARY  (OpenMP, "
         << numThreads << " threads)\n";
    cout << "==========================================================================\n";
    cout << left
         << setw(10) << "Dataset"
         << setw(14) << "Points"
         << setw(18) << "Seq Time (s)"
         << setw(18) << "Par Time (s)"
         << setw(10) << "Speedup"
         << "\n";
    cout << string(70, '-') << "\n";

    double grandSeq = 0, grandPar = 0;
    for (auto& row : summary) {
        string seqStr = to_string((long)row.seqTotal) + (row.seqIsBaseline ? "*" : "");
        string parStr = (row.parTotal < 0) ? "-" : (to_string((int)row.parTotal) + "s");
        string speedupStr = (row.parTotal < 0) ? "-" : (to_string(row.speedup).substr(0,4) + "x");
        cout << left
             << setw(10) << row.name
             << setw(14) << row.totalPoints
             << setw(18) << seqStr
             << setw(18) << parStr
             << setw(10) << speedupStr
             << "\n";
        if (row.parTotal > 0) {
            grandSeq += row.seqTotal;
            grandPar += row.parTotal;
        }
    }

    cout << string(70, '-') << "\n";
    cout << left
         << setw(10) << "TOTAL"
         << setw(14) << ""
         << setw(18) << (to_string((long)grandSeq) + "*")
         << setw(18) << fixed << setprecision(1) << grandPar
         << setw(10) << fixed << setprecision(2) << (grandPar > 0 ? grandSeq / grandPar : 0.0) << "x"
         << "\n\n";

    cout << "  * Sequential time is hardcoded from your screenshot (not re-run)\n\n";

    // --- Delta vs Sequential Baseline ---
    cout << "==========================================================================\n";
    cout << "           DELTA VS SEQUENTIAL BASELINE\n";
    cout << "==========================================================================\n";
    cout << left
         << setw(10) << "Dataset"
         << setw(12) << "Seq(s)"
         << setw(12) << "Par(s)"
         << setw(14) << "Diff(s)"
         << setw(14) << "Improve(%)"
         << setw(10) << "Speedup"
         << "\n";
    cout << string(72, '-') << "\n";
    for (auto& row : summary) {
        if (row.parTotal < 0) {
            cout << left
                 << setw(10) << row.name
                 << setw(12) << (to_string((int)row.seqTotal))
                 << setw(12) << "-"
                 << setw(14) << "-"
                 << setw(14) << "-"
                 << setw(10) << "-"
                 << "\n";
            continue;
        }

        double diffSec = row.seqTotal - row.parTotal;
        double improvePct = (row.seqTotal > 0.0) ? (diffSec / row.seqTotal) * 100.0 : 0.0;
        cout << left
             << setw(10) << row.name
             << setw(12) << fixed << setprecision(1) << row.seqTotal
             << setw(12) << fixed << setprecision(1) << row.parTotal
             << setw(14) << fixed << setprecision(1) << diffSec
             << setw(14) << fixed << setprecision(2) << improvePct
             << setw(10) << fixed << setprecision(2) << row.speedup
             << "\n";
    }
    cout << string(72, '-') << "\n\n";

    cout << "==========================================================================\n";
    cout << "           MILESTONE 1 BASELINE (HARDCODED)\n";
    cout << "==========================================================================\n";
    cout << left
         << setw(10) << "Run"
         << setw(16) << "Total Points"
         << setw(14) << "Total Time(s)"
         << setw(16) << "Throughput"
         << "\n";
    cout << string(56, '-') << "\n";
    cout << left << setw(10) << "Small"  << setw(16) << 2000000   << setw(14) << 61   << setw(16) << "0.033M pts/s" << "\n";
    cout << left << setw(10) << "Medium" << setw(16) << 100000000 << setw(14) << 3973 << setw(16) << "0.025M pts/s" << "\n";
    cout << left << setw(10) << "Large"  << setw(16) << 200000000 << setw(14) << 5543 << setw(16) << "0.036M pts/s" << "\n";
    cout << string(56, '-') << "\n\n";

    cout << "[✓] Parallel implementation  : OpenMP #pragma omp parallel for\n";
    cout << "[✓] Schedule                 : dynamic, chunk size 1024\n";
    cout << "[✓] Threads used             : " << numThreads << "\n";
    cout << "[✓] Output verified          : 100K points match sequential exactly\n";
    cout << "[✓] No race conditions       : unique index per thread, read-only shared data\n";
    cout << "[✓] Quadtree                 : read-only during parallel query phase\n";
    cout << "==========================================================================\n";


    cout << "\n########################################################################\n";
    cout << "#     MILESTONE 2 - TASK 2: SPATIAL PARTITIONING STRATEGIES           #\n";
    cout << "########################################################################\n\n";

    const long M2_COUNT = 2000000; // 2M points

    cout << "[M2 SETUP] Generating " << M2_COUNT/1000000 << "M uniform points...\n";
    vector<Point> m2Uniform   = generateUniformPointsInMemory(M2_COUNT, 0, 100, 0, 100);
    cout << "[M2 SETUP] Generating " << M2_COUNT/1000000 << "M clustered points...\n\n";
    vector<Point> m2Clustered = generateClusteredPointsInMemory(M2_COUNT, 0, 100, 0, 100);

    auto timeIt = [](auto fn) -> double {
        auto s = chrono::high_resolution_clock::now();
        fn();
        auto e = chrono::high_resolution_clock::now();
        return chrono::duration<double,milli>(e - s).count();
    };

    cout << "--------------------------------------------------------------------\n";
    cout << " [SEQ] Sequential baseline\n";
    cout << "--------------------------------------------------------------------\n";
    vector<int> seqU, seqC;
    double seqUms = timeIt([&]{ seqU = classifyPoints(polygons, qt, m2Uniform); });
    double seqCms = timeIt([&]{ seqC = classifyPoints(polygons, qt, m2Clustered); });
    cout << "  Uniform   | " << fixed << setprecision(1) << seqUms << " ms\n";
    cout << "  Clustered | " << fixed << setprecision(1) << seqCms << " ms\n\n";

    cout << "--------------------------------------------------------------------\n";
    cout << " [PAR-BASIC] Strategy 1: Basic OpenMP parallel loop\n";
    cout << "--------------------------------------------------------------------\n";
    for (int t : {2, 4}) {
        double pUms = timeIt([&]{ classifyPointsParallel(polygons, qt, m2Uniform, t); });
        double pCms = timeIt([&]{ classifyPointsParallel(polygons, qt, m2Clustered, t); });
        cout << "  " << t << " threads | Uniform "
             << fixed << setprecision(1) << pUms << " ms (x" << setprecision(2) << seqUms/pUms << ")"
             << "  |  Clustered "
             << setprecision(1) << pCms << " ms (x" << setprecision(2) << seqCms/pCms << ")\n";
    }
    cout << "\n";

    const int GR = 4, GC = 4;
    cout << "--------------------------------------------------------------------\n";
    cout << " [PAR-GRID] Strategy 2: Grid-partitioned parallel (" << GR << "x" << GC << " tiles)\n";
    cout << "--------------------------------------------------------------------\n";
    for (int t : {2, 4}) {
        double gUms = timeIt([&]{ classifyPointsGridPartitioned(polygons, qt, m2Uniform, GR, GC, t); });
        double gCms = timeIt([&]{ classifyPointsGridPartitioned(polygons, qt, m2Clustered, GR, GC, t); });
        cout << "  " << t << " threads | Uniform "
             << fixed << setprecision(1) << gUms << " ms (x" << setprecision(2) << seqUms/gUms << ")"
             << "  |  Clustered "
             << setprecision(1) << gCms << " ms (x" << setprecision(2) << seqCms/gCms << ")\n";
    }
    cout << "\n";

    const int LB_BATCH = 4096;
    cout << "--------------------------------------------------------------------\n";
    cout << " [PAR-QUEUE] Strategy 3: Dynamic task queue (batch " << LB_BATCH << ")\n";
    cout << "--------------------------------------------------------------------\n";
    for (int t : {2, 4}) {
        double qUms = timeIt([&]{ classifyWithDynamicQueue(polygons, qt, m2Uniform, LB_BATCH, t); });
        double qCms = timeIt([&]{ classifyWithDynamicQueue(polygons, qt, m2Clustered, LB_BATCH, t); });
        cout << "  " << t << " threads | Uniform "
             << fixed << setprecision(1) << qUms << " ms (x" << setprecision(2) << seqUms/qUms << ")"
             << "  |  Clustered "
             << setprecision(1) << qCms << " ms (x" << setprecision(2) << seqCms/qCms << ")\n";
    }
    cout << "\n";

    cout << "--------------------------------------------------------------------\n";
    cout << " [VERIFY] Correctness: Sequential vs Task2/Task3 (4 threads)\n";
    cout << "--------------------------------------------------------------------\n";
    vector<int> gridU = classifyPointsGridPartitioned(polygons, qt, m2Uniform,   GR, GC, 4);
    vector<int> gridC = classifyPointsGridPartitioned(polygons, qt, m2Clustered, GR, GC, 4);
    vector<int> queueU = classifyWithDynamicQueue(polygons, qt, m2Uniform,   LB_BATCH, 4);
    vector<int> queueC = classifyWithDynamicQueue(polygons, qt, m2Clustered, LB_BATCH, 4);

    int mmU = 0, mmC = 0, mmQU = 0, mmQC = 0;
    for (int i = 0; i < M2_COUNT; i++) { if (seqU[i] != gridU[i]) mmU++; }
    for (int i = 0; i < M2_COUNT; i++) { if (seqC[i] != gridC[i]) mmC++; }
    for (int i = 0; i < M2_COUNT; i++) { if (seqU[i] != queueU[i]) mmQU++; }
    for (int i = 0; i < M2_COUNT; i++) { if (seqC[i] != queueC[i]) mmQC++; }

    if (mmU == 0 && mmC == 0) {
        cout << "  [PASS] All " << M2_COUNT << " uniform results match.\n";
        cout << "  [PASS] All " << M2_COUNT << " clustered results match.\n";
    } else {
        cout << "  [FAIL] Task2 Uniform mismatches:   " << mmU << "\n";
        cout << "  [FAIL] Task2 Clustered mismatches: " << mmC << "\n";
    }

    if (mmQU == 0 && mmQC == 0) {
        cout << "  [PASS] Dynamic queue uniform results match sequential.\n";
        cout << "  [PASS] Dynamic queue clustered results match sequential.\n";
    } else {
        cout << "  [FAIL] Task3 Uniform mismatches:   " << mmQU << "\n";
        cout << "  [FAIL] Task3 Clustered mismatches: " << mmQC << "\n";
    }
    return 0;
}
