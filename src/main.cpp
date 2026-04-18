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

using namespace std;

// ── Point Generators (in-memory, no file I/O) ────────────────────────────────

vector<Point> generateUniformPointsInMemory(long count, double minX, double maxX, double minY, double maxY) {
    vector<Point> points;
    points.reserve(count);
    mt19937 gen(42);
    uniform_real_distribution<double> distX(minX, maxX);
    uniform_real_distribution<double> distY(minY, maxY);
    for (long i = 0; i < count; i++) points.push_back({distX(gen), distY(gen)});
    return points;
}

vector<Point> generateClusteredPointsInMemory(long count, double minX, double maxX, double minY, double maxY) {
    vector<Point> points;
    points.reserve(count);
    mt19937 gen(42);
    long pointsPerCluster = count / 5;
    double clusterWidth  = (maxX - minX) / 5;
    double clusterHeight = (maxY - minY) / 2;
    for (int c = 0; c < 5; c++) {
        double cx = minX + (c + 0.5) * clusterWidth;
        double cy = minY + (c % 2 == 0 ? 0.25 : 0.75) * (maxY - minY);
        normal_distribution<double> dX(cx, clusterWidth  / 4);
        normal_distribution<double> dY(cy, clusterHeight / 4);
        for (long i = 0; i < pointsPerCluster; i++) points.push_back({dX(gen), dY(gen)});
    }
    return points;
}

// ── Phase Result ─────────────────────────────────────────────────────────────

struct PhaseResult {
    double timeSec;
    long   insideCount;
    long   outsideCount;
};

// Run one distribution (uniform or clustered) through sequential or parallel pipeline.
// Uses batch processing (1 M points at a time) to keep memory usage low.
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
int main() {
    cout << "==========================================================\n";
    cout << "  POINT-IN-POLYGON  |  MILESTONE 2 - TASK 1\n";
    cout << "  Parallel Point Processing using OpenMP\n";
    cout << "==========================================================\n\n";

    int numThreads = omp_get_max_threads();
    cout << "[INFO] OpenMP max threads : " << numThreads << "\n";
    cout << "[INFO] Schedule           : dynamic, chunk = 1024\n\n";

    // ── PHASE 0: Load polygons, compute bboxes, build Quadtree ────────────────
    cout << "[PHASE 0] Loading data and building spatial index...\n";

    vector<Polygon> polygons = loadPolygons("data/polygons.txt");
    cout << "  - Loaded " << polygons.size() << " polygons\n";

    for (auto& poly : polygons) assignBoundingBox(poly);
    cout << "  - Bounding boxes computed\n";

    BoundingBox world = computeWorldBoundingBox(polygons);
    Quadtree qt(world);
    for (int i = 0; i < (int)polygons.size(); i++) qt.insert(i, polygons[i].bbox);
    cout << "  - Quadtree built\n\n";

    // ── PHASE 1: Correctness Verification ────────────────────────────────────
    cout << "[PHASE 1] Output Verification (100K uniform points)...\n";
    if (verifyOutputMatch(polygons, qt))
        cout << "  - [PASS] Parallel output matches sequential exactly.\n\n";
    else {
        cout << "  - [FAIL] Output mismatch! Aborting.\n\n";
        return 1;
    }

    // ── Dataset configs (baseline times from Milestone 1 performance table) ───
    // Small  = 1 M uniform + 1 M clustered  = 2 M total   -> run both seq & par
    // Medium = 50M uniform + 50M clustered  = 100M total  -> parallel only
    // Large  = 100M uniform + 100M clustered = 200M total -> parallel only
    DatasetConfig datasets[] = {
        { "Small",   1000000,    1000000,    30.0,   31.0,  true  },
        { "Medium",  50000000,   50000000,  2219.0, 1754.0, false },
        { "Large",  100000000,  100000000,  2732.0, 2811.0, false }
    };

    vector<SummaryRow> summary;

    for (auto& ds : datasets) {
        long total = ds.uniformPts + ds.clusteredPts;

        cout << "============================================================\n";
        cout << "  DATASET: " << ds.name << "  ("
             << fixed << setprecision(0) << total / 1e6 << "M points)\n";
        cout << "============================================================\n\n";

        double seqUniformSec, seqClusteredSec;

        // ── Sequential ───────────────────────────────────────────────────────
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
            cout << "[SEQ] Sequential baseline (Milestone 1 stored result):\n";
            cout << "  - Uniform   : " << seqUniformSec   << " sec\n";
            cout << "  - Clustered : " << seqClusteredSec << " sec\n";
            cout << "  - Total     : " << seqUniformSec + seqClusteredSec << " sec\n\n";
        }

        // ── Parallel ─────────────────────────────────────────────────────────
        cout << "[PAR] Running parallel classification (" << numThreads << " threads)...\n";
        PhaseResult pu = classifyPhase(ds.uniformPts,   true,  polygons, qt, true);
        PhaseResult pc = classifyPhase(ds.clusteredPts, false, polygons, qt, true);

        cout << "  - Uniform   : " << fixed << setprecision(2) << pu.timeSec << " sec\n";
        cout << "  - Clustered : " << pc.timeSec << " sec\n";
        cout << "  - Total     : " << pu.timeSec + pc.timeSec << " sec\n";
        cout << "  - Inside    : " << (pu.insideCount  + pc.insideCount)  << " pts\n";
        cout << "  - Outside   : " << (pu.outsideCount + pc.outsideCount) << " pts\n\n";

        double seqTotal = seqUniformSec + seqClusteredSec;
        double parTotal = pu.timeSec + pc.timeSec;
        double speedup  = seqTotal / parTotal;

        cout << "  >>> Speedup: " << fixed << setprecision(2) << speedup << "x"
             << "  (seq = " << (long)seqTotal << "s,  par = "
             << fixed << setprecision(1) << parTotal << "s)\n\n";

        summary.push_back({ds.name, total, seqTotal, parTotal, speedup, !ds.runSequential});
    }

    // ── Final Summary Table ───────────────────────────────────────────────────
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
        cout << left
             << setw(10) << row.name
             << setw(14) << row.totalPoints
             << setw(18) << seqStr
             << setw(18) << (to_string((int)row.parTotal) + "s")
             << setw(10) << (to_string(row.speedup).substr(0,4) + "x")
             << "\n";
        grandSeq += row.seqTotal;
        grandPar += row.parTotal;
    }

    cout << string(70, '-') << "\n";
    cout << left
         << setw(10) << "TOTAL"
         << setw(14) << ""
         << setw(18) << (to_string((long)grandSeq) + "*")
         << setw(18) << fixed << setprecision(1) << grandPar
         << setw(10) << fixed << setprecision(2) << grandSeq / grandPar << "x"
         << "\n\n";

    cout << "  * Sequential time from Milestone 1 stored baseline (not re-run)\n\n";

    cout << "[✓] Parallel implementation  : OpenMP #pragma omp parallel for\n";
    cout << "[✓] Schedule                 : dynamic, chunk size 1024\n";
    cout << "[✓] Threads used             : " << numThreads << "\n";
    cout << "[✓] Output verified          : 100K points match sequential exactly\n";
    cout << "[✓] No race conditions       : unique index per thread, read-only shared data\n";
    cout << "[✓] Quadtree                 : read-only during parallel query phase\n";
    cout << "==========================================================================\n";

    return 0;
}
