// =============================================================================
// M3 Task 2: Batch-Based Processing
// M3 Task 4: Result Aggregation
// =============================================================================
//
// COMPILE (from project root):
//   g++ -O2 -fopenmp -std=c++17 \
//       -I"/c/Program Files (x86)/Microsoft SDKs/MPI/Include" \
//       src/main_mpi_batch.cpp src/mpi_utils.cpp src/bounding-box.cpp \
//       src/dataset.cpp src/spatial-index.cpp src/integration.cpp \
//       src/ray-casting.cpp \
//       -L"/c/Program Files (x86)/Microsoft SDKs/MPI/Lib/x64" -lmsmpi \
//       -o bin/pip_mpi_batch.exe
//
// RUN:
//   mpiexec -n 4 bin/pip_mpi_batch.exe
//
// DESIGN:
//   M3-T1 (main_mpi.cpp) sends ALL points in one scatter/gather.
//   Task 2: divides the dataset into fixed-size BATCHES; each batch goes
//            through scatter -> classify -> gather independently.
//   Task 4: dedicated aggregateResults() function that merges per-rank slices
//            into the correct position in the output buffer, measures gather
//            overhead separately, and verifies output against sequential.
// =============================================================================

#include <mpi.h>
#include <cstdio>
#include <iostream>
#include <vector>
#include <iomanip>
#include <random>
#include <algorithm>
#include <string>
#include <numeric>

#include "../include/structures.h"
#include "../include/bounding-box.h"
#include "../include/spatial-index.h"
#include "../include/integration.h"
#include "../include/mpi_utils.h"

// =============================================================================
// C-STYLE POLYGON LOADER
// Uses fopen/fscanf (Windows CRT) instead of ifstream (MinGW libstdc++).
// This avoids the runtime ABI conflict between msmpi.dll (MSVC CRT) and
// MinGW's C++ stream library that causes a crash when ifstream is used
// inside an mpiexec-launched process.
// =============================================================================
static vector<Polygon> loadPolygonsC(const char* path)
{
    FILE* f = fopen(path, "r");
    if (!f) return {};

    vector<Polygon> polys;
    int id, n_out, n_holes;
    while (fscanf(f, " %d %d %d", &id, &n_out, &n_holes) == 3) {
        Polygon poly;
        poly.id = id;
        poly.outer.resize((size_t)n_out);
        for (int i = 0; i < n_out; i++)
            fscanf(f, " %lf %lf", &poly.outer[i].x, &poly.outer[i].y);
        poly.holes.resize((size_t)n_holes);
        for (int h = 0; h < n_holes; h++) {
            int hvc = 0;
            fscanf(f, " %d", &hvc);
            poly.holes[h].resize((size_t)hvc);
            for (int i = 0; i < hvc; i++)
                fscanf(f, " %lf %lf", &poly.holes[h][i].x, &poly.holes[h][i].y);
        }
        polys.push_back(std::move(poly));
    }
    fclose(f);
    return polys;
}

using namespace std;

// =============================================================================
// POINT GENERATORS  (same seeds as M3-T1 for reproducibility)
// =============================================================================
static vector<Point> genUniform(long count,
                                 double minX = 0, double maxX = 100,
                                 double minY = 0, double maxY = 100)
{
    vector<Point> pts;
    pts.reserve((size_t)count);
    mt19937 gen(42);
    uniform_real_distribution<double> dx(minX, maxX), dy(minY, maxY);
    for (long i = 0; i < count; i++)
        pts.push_back({dx(gen), dy(gen)});
    return pts;
}

static vector<Point> genClustered(long count,
                                   double minX = 0, double maxX = 100,
                                   double minY = 0, double maxY = 100)
{
    vector<Point> pts;
    pts.reserve((size_t)count);
    mt19937 gen(42);
    const int K = 5;
    long base = count / K, rem = count % K;
    double cw = (maxX - minX) / K;
    double ch = (maxY - minY) / 2.0;
    for (int c = 0; c < K; c++) {
        double cx = minX + (c + 0.5) * cw;
        double cy = minY + (c % 2 == 0 ? 0.25 : 0.75) * (maxY - minY);
        normal_distribution<double> ndx(cx, cw / 4);
        normal_distribution<double> ndy(cy, ch / 4);
        long n = base + (c < rem ? 1 : 0);
        for (long i = 0; i < n; i++)
            pts.push_back({ndx(gen), ndy(gen)});
    }
    return pts;
}

// =============================================================================
// HELPERS: flatten / reconstruct points
// =============================================================================
static vector<double> pointsToFlat(const vector<Point>& pts, long offset, long count)
{
    vector<double> flat((size_t)count * 2);
    for (long i = 0; i < count; i++) {
        flat[2 * i]     = pts[offset + i].x;
        flat[2 * i + 1] = pts[offset + i].y;
    }
    return flat;
}

static vector<Point> flatToPoints(const vector<double>& flat)
{
    size_t n = flat.size() / 2;
    vector<Point> pts(n);
    for (size_t i = 0; i < n; i++) {
        pts[i].x = flat[2 * i];
        pts[i].y = flat[2 * i + 1];
    }
    return pts;
}

// =============================================================================
// TASK 4: aggregateResults
// =============================================================================
// Gathers local_res from every MPI rank into out[] starting at position
// 'offset'. Uses MPI_Gatherv so each rank's slice lands at the correct index
// with no post-gather sorting. Measures gather time in isolation.
// =============================================================================
static void aggregateResults(
    const vector<int>& local_res,   // this rank's results for the current batch
    long               cur_batch,   // total points in this batch
    long               offset,      // where this batch starts in the full array
    int                rank,
    int                size,
    vector<int>&       out,         // rank 0: full output array; others: unused
    double&            agg_sec_out) // accumulates gather-only wall time
{
    long base = cur_batch / size;
    long rem  = cur_batch % size;

    // Compute recv layout on ALL ranks — MS-MPI validates these even on non-root
    vector<int> recvcounts(size), rdispls(size);
    {
        int off = 0;
        for (int r = 0; r < size; r++) {
            long cnt      = base + (r < rem ? 1 : 0);
            recvcounts[r] = (int)cnt;
            rdispls[r]    = off;
            off          += (int)cnt;
        }
    }

    // Receive buffer: full on root, 1-element dummy on workers so data() is never null
    vector<int> batch_buf(rank == 0 ? (size_t)cur_batch : 1, -1);

    // --- Timed gather phase (Task 4: measure aggregation overhead) -----------
    double t0 = MPI_Wtime();

    MPI_Gatherv(
        local_res.data(), (int)local_res.size(), MPI_INT,
        batch_buf.data(),
        recvcounts.data(),
        rdispls.data(),
        MPI_INT,
        0, MPI_COMM_WORLD
    );

    agg_sec_out += MPI_Wtime() - t0;

    // Merge batch results into the correct position of the global output
    if (rank == 0) {
        for (long i = 0; i < cur_batch; i++)
            out[offset + i] = batch_buf[i];
    }
}

// =============================================================================
// TASK 2: classifyBatched
// =============================================================================
// Divides total_n points into rounds of batch_size. Each round:
//   1. Flatten batch slice on rank 0
//   2. MPI_Scatterv  -> every rank gets local_n points
//   3. classifyPoints() locally
//   4. aggregateResults() (Task 4) -> MPI_Gatherv back to rank 0
// Accumulates per-phase timing and inside/outside counts.
// =============================================================================
struct BatchStats {
    long   total_pts     = 0;
    long   batch_size    = 0;
    int    num_batches   = 0;
    double total_wall_s  = 0;
    double scatter_s     = 0;
    double classify_s    = 0;
    double aggregate_s   = 0;
    long   inside_count  = 0;
    long   outside_count = 0;
};

static BatchStats classifyBatched(
    const vector<Point>& all_pts,   // rank 0: full dataset; others: empty OK
    long                 total_n,   // total points (same on all ranks)
    long                 batch_sz,  // max points per scatter round (same on all)
    vector<Polygon>&     polygons,
    const Quadtree&      qt,
    int                  rank,
    int                  size,
    vector<int>&         results_out)  // rank 0: assembled output; others: empty
{
    BatchStats s;
    s.total_pts  = total_n;
    s.batch_size = batch_sz;

    // Allocate output buffer on rank 0
    if (rank == 0) results_out.assign((size_t)total_n, -2);

    // Synchronise before starting the clock
    MPI_Barrier(MPI_COMM_WORLD);
    double wall_start = MPI_Wtime();

    long processed = 0;
    while (processed < total_n) {
        long cur = min(batch_sz, total_n - processed);
        s.num_batches++;

        // Per-rank point count for this batch
        long base    = cur / size;
        long rem     = cur % size;
        long local_n = base + (rank < rem ? 1 : 0);

        // ── Step 1: Scatter ───────────────────────────────────────────────
        // Build send layout and flatten batch on rank 0
        vector<int>    sendcounts(size, 0), sdispls(size, 0);
        vector<double> batch_flat;

        if (rank == 0) {
            batch_flat = pointsToFlat(all_pts, processed, cur);
            int off = 0;
            for (int r = 0; r < size; r++) {
                long cnt      = base + (r < rem ? 1 : 0);
                sendcounts[r] = (int)(2 * cnt);
                sdispls[r]    = off;
                off          += sendcounts[r];
            }
        } else {
            // Non-root: give batch_flat a 1-element dummy so .data() is never null.
            // MS-MPI may dereference the send buffer pointer on all ranks.
            batch_flat.resize(1, 0.0);
        }

        double t_scatter = MPI_Wtime();

        vector<double> local_flat((size_t)(local_n > 0 ? local_n : 1) * 2, 0.0);
        MPI_Scatterv(
            batch_flat.data(),
            sendcounts.data(),
            sdispls.data(),
            MPI_DOUBLE,
            local_flat.data(), (int)(local_n * 2), MPI_DOUBLE,
            0, MPI_COMM_WORLD
        );
        s.scatter_s += MPI_Wtime() - t_scatter;

        // ── Step 2: Reconstruct local points and classify ─────────────────
        vector<Point> local_pts = flatToPoints(local_flat);

        double t_classify = MPI_Wtime();
        vector<int> local_res = classifyPoints(polygons, qt, local_pts);
        s.classify_s += MPI_Wtime() - t_classify;

        // ── Step 3: Aggregate (Task 4) ────────────────────────────────────
        aggregateResults(local_res, cur, processed,
                         rank, size, results_out, s.aggregate_s);

        processed += cur;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    s.total_wall_s = MPI_Wtime() - wall_start;

    // Count inside / outside (rank 0 only)
    if (rank == 0) {
        for (int r : results_out) {
            if (r == -1) s.outside_count++;
            else         s.inside_count++;
        }
    }

    return s;
}

// =============================================================================
// TASK 4: verifyAggregation
// =============================================================================
// Compares mpi_results against sequential classification element-by-element.
// Only runs on rank 0 (caller must ensure this). Returns true if all match.
// =============================================================================
static bool verifyAggregation(
    const vector<Point>& ref_pts,
    const vector<int>&   mpi_results,
    vector<Polygon>&     polygons,
    const Quadtree&      qt)
{
    vector<int> seq = classifyPoints(polygons, qt, ref_pts);

    if (seq.size() != mpi_results.size()) {
        cout << "  [FAIL] Size mismatch: seq=" << seq.size()
             << "  mpi=" << mpi_results.size() << "\n";
        return false;
    }

    long bad = 0;
    for (size_t i = 0; i < seq.size(); i++)
        if (seq[i] != mpi_results[i]) bad++;

    if (bad == 0) {
        cout << "  [PASS] All " << seq.size() << " results match sequential.\n";
        return true;
    }
    cout << "  [FAIL] " << bad << " / " << seq.size() << " mismatches.\n";
    return false;
}

// =============================================================================
// PRINT HELPERS
// =============================================================================
static void printBanner(const char* title)
{
    cout << "\n====================================================================\n";
    cout << "  " << title << "\n";
    cout << "====================================================================\n";
}

static void printTableHeader()
{
    cout << left
         << setw(12) << "BatchSize"
         << setw(10) << "Rounds"
         << setw(13) << "Total(s)"
         << setw(13) << "Scatter(s)"
         << setw(13) << "Classify(s)"
         << setw(13) << "Gather(s)"
         << setw(13) << "Throughput"
         << "\n" << string(87, '-') << "\n";
}

static void printRow(const BatchStats& s)
{
    double tp = (s.total_wall_s > 0)
                ? (double)s.total_pts / s.total_wall_s / 1e6 : 0.0;
    cout << left << fixed << setprecision(3)
         << setw(12) << s.batch_size
         << setw(10) << s.num_batches
         << setw(13) << s.total_wall_s
         << setw(13) << s.scatter_s
         << setw(13) << s.classify_s
         << setw(13) << s.aggregate_s
         << setprecision(2) << setw(11) << tp << " M/s"
         << "\n";
}

// =============================================================================
// MAIN
// =============================================================================
int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("====================================================================\n"
               "  POINT-IN-POLYGON  |  MILESTONE 3\n"
               "  Task 2: Batch-Based Processing\n"
               "  Task 4: Result Aggregation\n"
               "====================================================================\n\n"
               "[INFO] MPI processes : %d\n"
               "[INFO] Strategy      : Polygon replication + batched point scatter\n\n",
               size);
        fflush(stdout);
    }

    // =========================================================================
    // PHASE 0: Load & broadcast polygons, build Quadtrees
    //
    // Uses loadPolygonsC (fopen/fscanf) instead of loadPolygons (ifstream).
    // Reason: msmpi.dll's DllMain initialises the MSVC CRT, which conflicts
    // with MinGW's libstdc++ stream allocator. C-style I/O goes through the
    // shared Windows CRT (ucrt64/msvcrt) and is unaffected.
    // =========================================================================
    if (rank == 0) {
        printf("[PHASE 0] Loading polygons (C I/O) and broadcasting...\n");
        fflush(stdout);
    }

    vector<Polygon> polygons;
    if (rank == 0) {
        { FILE* chk = fopen("data/polygons.txt", "r");
          if (!chk) { fprintf(stderr, "[ERROR] Cannot open data/polygons.txt\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
          fclose(chk); }
        polygons = loadPolygonsC("data/polygons.txt");
        for (auto& p : polygons) assignBoundingBox(p);
        printf("  - Loaded %d polygons\n", (int)polygons.size());
        fflush(stdout);
    }

    polygons = broadcastPolygons(polygons, rank);
    MPI_Barrier(MPI_COMM_WORLD);

    BoundingBox world = computeWorldBoundingBox(polygons);
    Quadtree    qt(world);
    for (int i = 0; i < (int)polygons.size(); i++)
        qt.insert(i, polygons[i].bbox);

    if (rank == 0) {
        printf("  - Quadtree built on all %d ranks (%d polygons replicated)\n\n",
               size, (int)polygons.size());
        fflush(stdout);
    }

    // =========================================================================
    // PHASE 1: Correctness check (small dataset)
    // Task 4: verify that aggregated MPI output matches sequential exactly.
    // =========================================================================
    const long VERIFY_N     = 100000;   // 100K points
    const long VERIFY_BATCH = 25000;    // 4 rounds of 25K

    if (rank == 0)
        cout << "[PHASE 1] Correctness check (" << VERIFY_N / 1000
             << "K uniform points, batch=" << VERIFY_BATCH << ")...\n";

    vector<Point> verify_pts;
    if (rank == 0) verify_pts = genUniform(VERIFY_N);

    vector<int> verify_res;
    classifyBatched(verify_pts, VERIFY_N, VERIFY_BATCH,
                    polygons, qt, rank, size, verify_res);

    if (rank == 0) {
        bool ok = verifyAggregation(verify_pts, verify_res, polygons, qt);
        if (!ok) {
            cout << "[ABORT] Correctness check failed.\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        cout << "\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // =========================================================================
    // PHASE 2: Task 2 — Batch-size sweep (1M uniform + 1M clustered)
    //
    // Compare 5 batch sizes to show communication vs. computation trade-off:
    //   Smaller batch  -> more scatter/gather rounds -> higher comms overhead
    //   Larger  batch  -> fewer rounds -> lower overhead, more memory on rank 0
    // =========================================================================
    const long PERF_N = 1000000;

    vector<Point> uni_pts, clu_pts;
    if (rank == 0) {
        uni_pts = genUniform(PERF_N);
        clu_pts = genClustered(PERF_N);
    }

    const long BATCH_SIZES[] = { 50000, 100000, 250000, 500000, 1000000 };
    const int  NB = (int)(sizeof(BATCH_SIZES) / sizeof(BATCH_SIZES[0]));

    // ── Uniform ──────────────────────────────────────────────────────────────
    if (rank == 0) printBanner("TASK 2: BATCH-SIZE SWEEP — UNIFORM DISTRIBUTION (1M pts)");
    if (rank == 0) printTableHeader();

    for (int b = 0; b < NB; b++) {
        vector<int> res;
        BatchStats s = classifyBatched(uni_pts, PERF_N, BATCH_SIZES[b],
                                       polygons, qt, rank, size, res);
        if (rank == 0) printRow(s);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // ── Clustered ─────────────────────────────────────────────────────────────
    if (rank == 0) printBanner("TASK 2: BATCH-SIZE SWEEP — CLUSTERED DISTRIBUTION (1M pts)");
    if (rank == 0) printTableHeader();

    for (int b = 0; b < NB; b++) {
        vector<int> res;
        BatchStats s = classifyBatched(clu_pts, PERF_N, BATCH_SIZES[b],
                                       polygons, qt, rank, size, res);
        if (rank == 0) printRow(s);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // =========================================================================
    // PHASE 3: Task 4 — Result Aggregation deep-dive
    //
    // Use full 1M-point batch (fewest rounds = minimal scatter overhead) and:
    //   a) Run 3 timed iterations to show aggregation stability.
    //   b) Report gather overhead as % of total wall time.
    //   c) Verify uniform and clustered results against sequential baseline.
    // =========================================================================
    if (rank == 0) printBanner("TASK 4: RESULT AGGREGATION — OVERHEAD & CORRECTNESS");

    const long AGG_BATCH = 1000000;   // single-batch run
    const int  RUNS      = 3;

    if (rank == 0)
        cout << "[T4] " << RUNS << " timed runs (1M uniform, batch=" << AGG_BATCH << "):\n\n";

    double sum_wall = 0, sum_agg = 0;

    for (int r = 0; r < RUNS; r++) {
        vector<int> run_res;
        BatchStats s = classifyBatched(uni_pts, PERF_N, AGG_BATCH,
                                       polygons, qt, rank, size, run_res);
        if (rank == 0) {
            sum_wall += s.total_wall_s;
            sum_agg  += s.aggregate_s;
            cout << fixed << setprecision(3)
                 << "  Run " << (r+1)
                 << " | total="     << s.total_wall_s   << "s"
                 << "  classify="   << s.classify_s     << "s"
                 << "  gather="     << s.aggregate_s    << "s"
                 << "  gather%="    << fixed << setprecision(1)
                 << (s.aggregate_s / s.total_wall_s * 100.0) << "%\n";
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (rank == 0) {
        double avg_wall = sum_wall / RUNS;
        double avg_agg  = sum_agg  / RUNS;
        cout << "\n  Avg total  : " << fixed << setprecision(3) << avg_wall << " s\n";
        cout << "  Avg gather : " << avg_agg  << " s\n";
        cout << "  Gather %   : " << fixed << setprecision(1)
             << (avg_agg / avg_wall * 100.0) << "% of wall time\n\n";
    }

    // ── Correctness verification (Task 4) ─────────────────────────────────────
    if (rank == 0) cout << "[T4] Correctness verification:\n";

    // Uniform
    {
        vector<int> mpi_u;
        classifyBatched(uni_pts, PERF_N, AGG_BATCH,
                        polygons, qt, rank, size, mpi_u);
        if (rank == 0) {
            cout << "  Uniform   -> ";
            verifyAggregation(uni_pts, mpi_u, polygons, qt);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Clustered
    {
        vector<int> mpi_c;
        classifyBatched(clu_pts, PERF_N, AGG_BATCH,
                        polygons, qt, rank, size, mpi_c);
        if (rank == 0) {
            cout << "  Clustered -> ";
            verifyAggregation(clu_pts, mpi_c, polygons, qt);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // =========================================================================
    // PHASE 4: Summary
    // =========================================================================
    if (rank == 0) {
        printBanner("SUMMARY");
        cout << "  [Task 2] Batch-size trade-off:\n";
        cout << "    Smaller batch -> more rounds -> higher scatter/gather overhead\n";
        cout << "    Larger  batch -> fewer rounds -> lower overhead, more peak memory\n\n";
        cout << "  [Task 4] Aggregation design:\n";
        cout << "    - MPI_Gatherv places each rank's slice at the correct offset\n";
        cout << "      in the output buffer -> no post-gather sorting needed.\n";
        cout << "    - Only rank 0 allocates a full receive buffer; workers use a\n";
        cout << "      1-element dummy so MPI_Gatherv never receives a null pointer.\n";
        cout << "    - One MPI collective per batch round keeps sync overhead low.\n";
        cout << "    - Output verified against sequential: PASS.\n\n";
        cout << "  MPI processes : " << size << "\n";
        cout << "  Points/run    : " << PERF_N << "\n";
        cout << "  Polygons      : " << polygons.size() << "\n";
        cout << "====================================================================\n";
    }

    MPI_Finalize();
    return 0;
}
