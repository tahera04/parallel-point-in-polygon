// =============================================================================
// MILESTONE 3 -- TASK 3: DISTRIBUTED EXECUTION MODEL
// Tahera Abidi -- 29280
// =============================================================================
// Strategy A: Point Partitioning + Polygon Replication
//   - All polygons broadcast to every process
//   - Points divided equally among processes
//   - Each process classifies its chunk independently
//
// Strategy B: Spatial Sharding + Point Routing
//   - Polygons divided spatially among processes
//   - Master routes each query point to correct process
//   - Lower memory per process, more communication
// =============================================================================

#include <mpi.h>
#include <cstdio>
#include <iostream>
#include <vector>
#include <iomanip>
#include <random>
#include <algorithm>
#include <string>
#include <cmath>
#include <sstream>

#include "../include/structures.h"
#include "../include/bounding-box.h"
#include "../include/spatial-index.h"
#include "../include/integration.h"
#include "../include/mpi_utils.h"

using namespace std;

// =============================================================================
// POLYGON LOADER (C-style to avoid MinGW/MSMPI CRT conflict)
// =============================================================================
static vector<Polygon> loadPolygonsC(const char* path)
{
    FILE* f = fopen(path, "r");
    if (!f) { printf("ERROR: cannot open %s\n", path); return {}; }
    vector<Polygon> polys;
    int id, n_out, n_holes;
    while (fscanf(f, " %d %d %d", &id, &n_out, &n_holes) == 3) {
        Polygon poly;
        poly.id = id;
        poly.outer.resize(n_out);
        for (int i = 0; i < n_out; i++)
            fscanf(f, " %lf %lf", &poly.outer[i].x, &poly.outer[i].y);
        poly.holes.resize(n_holes);
        for (int h = 0; h < n_holes; h++) {
            int hvc = 0; fscanf(f, " %d", &hvc);
            poly.holes[h].resize(hvc);
            for (int i = 0; i < hvc; i++)
                fscanf(f, " %lf %lf", &poly.holes[h][i].x, &poly.holes[h][i].y);
        }
        polys.push_back(poly);
    }
    fclose(f);
    return polys;
}

// =============================================================================
// POINT GENERATORS
// =============================================================================
static vector<Point> genUniform(long n)
{
    vector<Point> pts; pts.reserve(n);
    mt19937 gen(42);
    uniform_real_distribution<double> dx(0,100), dy(0,100);
    for (long i = 0; i < n; i++) pts.push_back({dx(gen), dy(gen)});
    return pts;
}

static vector<Point> genClustered(long n)
{
    vector<Point> pts; pts.reserve(n);
    mt19937 gen(42);
    const int K = 5;
    long base = n/K, rem = n%K;
    double cw = 20.0, ch = 25.0;
    for (int c = 0; c < K; c++) {
        double cx = (c+0.5)*cw, cy = (c%2==0?25.0:75.0);
        normal_distribution<double> ndx(cx, cw/4), ndy(cy, ch/4);
        long cnt = base + (c < rem ? 1 : 0);
        for (long i = 0; i < cnt; i++) pts.push_back({ndx(gen), ndy(gen)});
    }
    return pts;
}

// =============================================================================
// HELPERS
// =============================================================================
static vector<double> flatten(const vector<Point>& pts)
{
    vector<double> f(pts.size()*2);
    for (size_t i=0;i<pts.size();i++){f[2*i]=pts[i].x;f[2*i+1]=pts[i].y;}
    return f;
}
static vector<Point> unflatten(const vector<double>& f, long n)
{
    vector<Point> pts(n);
    for (long i=0;i<n;i++){pts[i].x=f[2*i];pts[i].y=f[2*i+1];}
    return pts;
}

// =============================================================================
// STRATEGY A -- Point Partitioning + Polygon Replication
// All ranks already have full polygon set (broadcast done in main).
// Points are scattered equally. Results gathered to rank 0.
// =============================================================================
struct Result {
    double wall=0, comm=0, cls=0;
    long inside=0, outside=0;
    long poly_per_proc=0;
};

static Result strategyA(
    const vector<Point>& all_pts, long total_n,
    vector<Polygon>& polygons, const Quadtree& qt,
    int rank, int size, vector<int>& out)
{
    Result res;
    res.poly_per_proc = (long)polygons.size();

    long base = total_n / size, rem = total_n % size;
    long local_n = base + (rank < rem ? 1 : 0);

    // Build scatter counts
    vector<int> scounts(size), sdispls(size);
    {int off=0; for(int r=0;r<size;r++){
        long c=base+(r<rem?1:0);
        scounts[r]=(int)(c*2); sdispls[r]=off; off+=scounts[r];
    }}

    // FIX: non-rank-0 processes must not pass a dangling pointer to MPI_Scatterv.
    // Pass nullptr for sendbuf on non-root; local_flat is properly sized for receive.
    vector<double> all_flat;
    if (rank==0) all_flat = flatten(all_pts);

    vector<double> local_flat((size_t)(local_n > 0 ? local_n : 1)*2, 0.0);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    double tc0 = MPI_Wtime();
    MPI_Scatterv(
        rank==0 ? all_flat.data() : nullptr,
        scounts.data(), sdispls.data(), MPI_DOUBLE,
        local_flat.data(), (int)(local_n*2), MPI_DOUBLE,
        0, MPI_COMM_WORLD);
    res.comm += MPI_Wtime()-tc0;

    vector<Point> local_pts = unflatten(local_flat, local_n);
    double tc1 = MPI_Wtime();
    vector<int> local_res = classifyPoints(polygons, qt, local_pts);
    res.cls = MPI_Wtime()-tc1;

    vector<int> rcounts(size), rdispls(size);
    {int off=0; for(int r=0;r<size;r++){
        long c=base+(r<rem?1:0);
        rcounts[r]=(int)c; rdispls[r]=off; off+=(int)c;
    }}
    if (rank==0) out.resize(total_n,-2);

    double tc2 = MPI_Wtime();
    MPI_Gatherv(local_res.data(), (int)local_res.size(), MPI_INT,
                rank==0?out.data():nullptr,
                rank==0?rcounts.data():nullptr,
                rank==0?rdispls.data():nullptr,
                MPI_INT, 0, MPI_COMM_WORLD);
    res.comm += MPI_Wtime()-tc2;

    MPI_Barrier(MPI_COMM_WORLD);
    res.wall = MPI_Wtime()-t0;

    if (rank==0)
        for (int r:out){ if(r==-1) res.outside++; else res.inside++; }

    return res;
}

// =============================================================================
// STRATEGY B -- Spatial Sharding + Point Routing
// =============================================================================

static void gridDims(int size, int& rows, int& cols)
{
    rows = (int)round(sqrt((double)size));
    while (size%rows!=0) rows--;
    cols = size/rows;
}

static int getOwner(double x, double y, int rows, int cols)
{
    double w=100.0/cols, h=100.0/rows;
    int col=(int)(x/w), row=(int)(y/h);
    col=max(0,min(cols-1,col));
    row=max(0,min(rows-1,row));
    return row*cols+col;
}

// FIX: The original code filtered polygons by their *center point* falling
// inside the shard region. This silently dropped polygons whose center was
// outside the shard but whose body extended into it. A point near the shard
// boundary could land inside one of those dropped polygons, getting -1 from
// Strategy B but the correct polygon ID from Strategy A, triggering the
// correctness check abort.
//
// Fix: keep a polygon on a shard if its *bounding box* overlaps the shard's
// region (expanded by OVERLAP). This is a conservative superset -- it may
// include extra polygons, but it never misses a relevant one.
static bool bboxOverlapsRegion(
    const BoundingBox& pb,
    double rx0, double rx1, double ry0, double ry1,
    double overlap)
{
    return (pb.max_x >= rx0 - overlap &&
            pb.min_x <  rx1 + overlap &&
            pb.max_y >= ry0 - overlap &&
            pb.min_y <  ry1 + overlap);
}

static void serPoly(const vector<Polygon>& ps, vector<int>& ib, vector<double>& db)
{
    ib.clear(); db.clear();
    ib.push_back((int)ps.size());
    for (const auto& p : ps) {
        ib.push_back(p.id);
        ib.push_back((int)p.outer.size());
        ib.push_back((int)p.holes.size());
        for (const auto& h : p.holes) ib.push_back((int)h.size());
        db.push_back(p.bbox.min_x); db.push_back(p.bbox.max_x);
        db.push_back(p.bbox.min_y); db.push_back(p.bbox.max_y);
        for (const auto& v : p.outer){db.push_back(v.x);db.push_back(v.y);}
        for (const auto& hole : p.holes)
            for (const auto& v : hole){db.push_back(v.x);db.push_back(v.y);}
    }
}

static vector<Polygon> deserPoly(const vector<int>& ib, const vector<double>& db)
{
    int ii=0,di=0;
    int np=ib[ii++];
    vector<Polygon> ps(np);
    for (int p=0;p<np;p++) {
        ps[p].id=ib[ii++];
        int no=ib[ii++], nh=ib[ii++];
        vector<int> hsz(nh); for(int h=0;h<nh;h++) hsz[h]=ib[ii++];
        ps[p].bbox={db[di],db[di+1],db[di+2],db[di+3]}; di+=4;
        ps[p].outer.resize(no);
        for(int i=0;i<no;i++){ps[p].outer[i].x=db[di++];ps[p].outer[i].y=db[di++];}
        ps[p].holes.resize(nh);
        for(int h=0;h<nh;h++){
            ps[p].holes[h].resize(hsz[h]);
            for(int i=0;i<hsz[h];i++){
                ps[p].holes[h][i].x=db[di++];
                ps[p].holes[h][i].y=db[di++];
            }
        }
    }
    return ps;
}

static Result strategyB(
    const vector<Point>& all_pts, long total_n,
    const vector<Polygon>& all_polygons,
    int rank, int size, vector<int>& out)
{
    Result res;
    const double OVERLAP = 5.0;
    int rows, cols;
    gridDims(size, rows, cols);
    double cell_w = 100.0 / cols;
    double cell_h = 100.0 / rows;

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    double comm_t = 0;

    // -- STEP 1: Distribute polygon subsets ----------------------------------
    vector<Polygon> my_polys;

    if (rank == 0) {
        for (int r = 1; r < size; r++) {
            int rc = r % cols, rr = r / cols;
            double rx0 = rc * cell_w, rx1 = (rc+1) * cell_w;
            double ry0 = rr * cell_h, ry1 = (rr+1) * cell_h;

            vector<Polygon> sub;
            for (const auto& p : all_polygons)
                if (bboxOverlapsRegion(p.bbox, rx0, rx1, ry0, ry1, OVERLAP))
                    sub.push_back(p);

            vector<int> ib; vector<double> db;
            serPoly(sub, ib, db);
            int isz=(int)ib.size(), dsz=(int)db.size();

            double tc=MPI_Wtime();
            MPI_Send(&isz,1,MPI_INT,   r,0,MPI_COMM_WORLD);
            MPI_Send(&dsz,1,MPI_INT,   r,1,MPI_COMM_WORLD);
            MPI_Send(ib.data(),isz,MPI_INT,   r,2,MPI_COMM_WORLD);
            MPI_Send(db.data(),dsz,MPI_DOUBLE,r,3,MPI_COMM_WORLD);
            comm_t += MPI_Wtime()-tc;
        }

        // Rank 0 keeps its own region
        double rx0=0, rx1=cell_w, ry0=0, ry1=cell_h;
        for (const auto& p : all_polygons)
            if (bboxOverlapsRegion(p.bbox, rx0, rx1, ry0, ry1, OVERLAP))
                my_polys.push_back(p);

    } else {
        int isz=0, dsz=0;
        double tc=MPI_Wtime();
        MPI_Recv(&isz,1,MPI_INT,   0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Recv(&dsz,1,MPI_INT,   0,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        vector<int> ib(isz); vector<double> db(dsz);
        MPI_Recv(ib.data(),isz,MPI_INT,   0,2,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Recv(db.data(),dsz,MPI_DOUBLE,0,3,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        comm_t += MPI_Wtime()-tc;
        my_polys = deserPoly(ib, db);
    }

    res.poly_per_proc = (long)my_polys.size();

    BoundingBox fixed_world{0,100,0,100};
    Quadtree local_qt(fixed_world);
    for (int i=0;i<(int)my_polys.size();i++)
        local_qt.insert(i, my_polys[i].bbox);

    // -- STEP 2: Route points and classify -----------------------------------
    if (rank == 0) out.assign(total_n, -1);

    if (rank == 0) {
        vector<vector<long>>  ridx(size);
        vector<vector<Point>> rpts(size);
        for (long i=0;i<total_n;i++) {
            int owner = getOwner(all_pts[i].x, all_pts[i].y, rows, cols);
            ridx[owner].push_back(i);
            rpts[owner].push_back(all_pts[i]);
        }

        for (int r=1;r<size;r++) {
            long npts=(long)rpts[r].size();
            vector<double> flat=flatten(rpts[r]);
            int fsz=(int)flat.size();
            double tc=MPI_Wtime();
            MPI_Send(&npts,1,MPI_LONG,     r,10,MPI_COMM_WORLD);
            MPI_Send(flat.data(),fsz,MPI_DOUBLE,r,11,MPI_COMM_WORLD);
            comm_t += MPI_Wtime()-tc;
        }

        double tc=MPI_Wtime();
        vector<int> r0res = classifyPoints(my_polys, local_qt, rpts[0]);
        res.cls += MPI_Wtime()-tc;
        for (size_t i=0;i<ridx[0].size();i++)
            out[ridx[0][i]] = r0res[i];

        for (int r=1;r<size;r++) {
            long npts=(long)rpts[r].size();
            vector<int> wres(npts,-1);
            double tc2=MPI_Wtime();
            MPI_Recv(wres.data(),(int)npts,MPI_INT,r,20,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            comm_t += MPI_Wtime()-tc2;
            for (size_t i=0;i<ridx[r].size();i++)
                out[ridx[r][i]] = wres[i];
        }

        for (int r:out){ if(r==-1) res.outside++; else res.inside++; }

    } else {
        long npts=0;
        double tc=MPI_Wtime();
        MPI_Recv(&npts,1,MPI_LONG,0,10,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        vector<double> flat((size_t)(max(npts,1L))*2,0);
        MPI_Recv(flat.data(),(int)(npts*2),MPI_DOUBLE,0,11,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        comm_t += MPI_Wtime()-tc;

        vector<Point> my_pts = unflatten(flat, npts);

        double tc2=MPI_Wtime();
        vector<int> local_res = classifyPoints(my_polys, local_qt, my_pts);
        res.cls = MPI_Wtime()-tc2;

        double tc3=MPI_Wtime();
        MPI_Send(local_res.data(),(int)local_res.size(),MPI_INT,0,20,MPI_COMM_WORLD);
        comm_t += MPI_Wtime()-tc3;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    res.wall = MPI_Wtime()-t0;
    res.comm = comm_t;
    return res;
}

// =============================================================================
// PRINT
// =============================================================================
static void printResult(const char* name, const Result& r, long n, int size)
{
    cout << "  " << name << "\n";
    cout << "    Wall time       : " << fixed << setprecision(3) << r.wall << " sec\n";
    cout << "    Classify time   : " << r.cls  << " sec\n";
    cout << "    Comm time       : " << r.comm << " sec\n";
    cout << "    Comm overhead   : " << fixed << setprecision(1)
         << (r.wall>0 ? r.comm/r.wall*100 : 0) << "%\n";
    cout << "    Throughput      : " << fixed << setprecision(0)
         << (r.wall>0 ? n/r.wall : 0) << " pts/sec\n";
    cout << "    Inside          : " << r.inside  << "\n";
    cout << "    Outside         : " << r.outside << "\n";
    cout << "    Polygons/process: ~" << r.poly_per_proc << "\n";
    cout << "    Points/process  : ~" << n/size << "\n\n";
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

    int rows, cols;
    gridDims(size, rows, cols);

    if (rank==0) {
        // FIX: em-dashes replaced with "--" -- Windows CP1252 console cannot
        // display UTF-8 multi-byte characters, producing garbled output (e.g. ΓÇö).
        cout << "====================================================================\n";
        cout << "  POINT-IN-POLYGON  |  MILESTONE 3 -- TASK 3\n";
        cout << "  Strategy A: Point Partitioning + Polygon Replication\n";
        cout << "  Strategy B: Spatial Sharding + Point Routing\n";
        cout << "====================================================================\n\n";
        cout << "[INFO] MPI processes : " << size << "\n";
        cout << "[INFO] Grid layout   : " << rows << "x" << cols << "\n";
        cout << "[INFO] Overlap buffer: 5.0 units\n\n";
        fflush(stdout);
    }

    if (rank==0) cout << "[PHASE 0] Loading polygons...\n";

    vector<Polygon> all_polygons;
    if (rank==0) {
        all_polygons = loadPolygonsC("data/polygons.txt");
        for (auto& p : all_polygons) assignBoundingBox(p);
        cout << "  - Loaded " << all_polygons.size() << " polygons\n";
    }

    vector<Polygon> poly_A = broadcastPolygons(all_polygons, rank);
    MPI_Barrier(MPI_COMM_WORLD);

    BoundingBox world{0,100,0,100};
    Quadtree qt_A(world);
    for (int i=0;i<(int)poly_A.size();i++)
        qt_A.insert(i, poly_A[i].bbox);

    if (rank==0) cout << "  - Quadtree built on all " << size << " ranks\n\n";

    // -- Correctness check: 100K points --------------------------------------
    const long VN = 100000;
    if (rank==0) cout << "[PHASE 1] Correctness check (" << VN/1000 << "K points)...\n";

    vector<Point> vpts;
    if (rank==0) vpts = genUniform(VN);

    vector<int> vA, vB;
    strategyA(vpts, VN, poly_A, qt_A, rank, size, vA);
    strategyB(vpts, VN, all_polygons, rank, size, vB);

    if (rank==0) {
        vector<int> vSeq = classifyPoints(poly_A, qt_A, vpts);
        long abMM=0, asMM=0;
        for (size_t i=0;i<vA.size();i++) {
            if(vA[i]!=vB[i])   abMM++;
            if(vA[i]!=vSeq[i]) asMM++;
        }
        cout << "  Strategy A vs B  : " << (abMM==0?"[PASS]":"[FAIL]")
             << " (" << abMM << " mismatches)\n";
        cout << "  Strategy A vs Seq: " << (asMM==0?"[PASS]":"[FAIL]")
             << " (" << asMM << " mismatches)\n\n";

        if (abMM > 0 || asMM > 0) {
            cout << "[ABORT] Correctness failed.\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // -- Performance: 1M uniform ---------------------------------------------
    const long PN = 1000000;
    vector<Point> upts, cpts;
    if (rank==0) { upts=genUniform(PN); cpts=genClustered(PN); }

    if (rank==0) cout << "[PHASE 2] Performance: 1M UNIFORM points\n";
    vector<int> uA_res, uB_res;
    Result uA = strategyA(upts, PN, poly_A, qt_A, rank, size, uA_res);
    Result uB = strategyB(upts, PN, all_polygons, rank, size, uB_res);
    if (rank==0) {
        printResult("Strategy A (Point Partition + Polygon Replication)", uA, PN, size);
        printResult("Strategy B (Spatial Sharding + Point Routing)",      uB, PN, size);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // -- Performance: 1M clustered -------------------------------------------
    if (rank==0) cout << "[PHASE 3] Performance: 1M CLUSTERED points\n";
    vector<int> cA_res, cB_res;
    Result cA = strategyA(cpts, PN, poly_A, qt_A, rank, size, cA_res);
    Result cB = strategyB(cpts, PN, all_polygons, rank, size, cB_res);
    if (rank==0) {
        printResult("Strategy A (Point Partition + Polygon Replication)", cA, PN, size);
        printResult("Strategy B (Spatial Sharding + Point Routing)",      cB, PN, size);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // -- Correctness on 1M ---------------------------------------------------
    if (rank==0) {
        cout << "[PHASE 4] Correctness on 1M points\n";
        long umm=0, cmm=0;
        for(size_t i=0;i<uA_res.size();i++) if(uA_res[i]!=uB_res[i]) umm++;
        for(size_t i=0;i<cA_res.size();i++) if(cA_res[i]!=cB_res[i]) cmm++;
        cout << "  Uniform  A vs B: " << (umm==0?"[PASS]":"[FAIL]")
             << " (" << umm << " mismatches)\n";
        cout << "  Clustered A vs B: " << (cmm==0?"[PASS]":"[FAIL]")
             << " (" << cmm << " mismatches)\n\n";
    }

    // -- Summary -------------------------------------------------------------
    if (rank==0) {
        cout << "====================================================================\n";
        cout << "  COMPARISON SUMMARY (" << size << " processes)\n";
        cout << "====================================================================\n";
        cout << left << setw(30) << "Metric"
             << setw(20) << "Strategy A"
             << setw(20) << "Strategy B" << "\n";
        cout << string(70,'-') << "\n";

        auto fmt  = [](double v){ ostringstream s; s<<fixed<<setprecision(3)<<v<<"s"; return s.str(); };
        auto fmtp = [](double v){ ostringstream s; s<<fixed<<setprecision(0)<<v<<"pts/s"; return s.str(); };

        cout << left << setw(30) << "Uniform wall time"
             << setw(20) << fmt(uA.wall) << setw(20) << fmt(uB.wall) << "\n";
        cout << left << setw(30) << "Clustered wall time"
             << setw(20) << fmt(cA.wall) << setw(20) << fmt(cB.wall) << "\n";
        cout << left << setw(30) << "Uniform throughput"
             << setw(20) << fmtp(PN/uA.wall) << setw(20) << fmtp(PN/uB.wall) << "\n";
        cout << left << setw(30) << "Clustered throughput"
             << setw(20) << fmtp(PN/cA.wall) << setw(20) << fmtp(PN/cB.wall) << "\n";
        cout << left << setw(30) << "Polygons per process"
             << setw(20) << (to_string(uA.poly_per_proc)+" (all)")
             << setw(20) << ("~"+to_string(uB.poly_per_proc)+" (subset)") << "\n";
        cout << left << setw(30) << "Uniform comm overhead"
             << setw(20) << (to_string((int)(uA.comm/uA.wall*100))+"%")
             << setw(20) << (to_string((int)(uB.comm/uB.wall*100))+"%") << "\n";
        cout << string(70,'-') << "\n\n";

        cout << "  Strategy A: simpler, predictable, equal load per process\n";
        cout << "  Strategy B: lower memory per process (~1/" << size
             << " polygons), higher comm cost\n";
        cout << "  Faster strategy (uniform)  : "
             << (uA.wall < uB.wall ? "Strategy A" : "Strategy B") << "\n";
        cout << "  Faster strategy (clustered): "
             << (cA.wall < cB.wall ? "Strategy A" : "Strategy B") << "\n\n";
    }

    MPI_Finalize();
    return 0;
}