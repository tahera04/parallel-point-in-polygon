#include "../include/mpi_utils.h"
#include <mpi.h>

using namespace std;

// ---------------------------------------------------------------------------
// serializePolygons
// ---------------------------------------------------------------------------
// int_buf  : [num_polygons | for each poly: id, outer_count, hole_count,
//                                           hole_0_size, hole_1_size, ...]
// double_buf: [for each poly: min_x, max_x, min_y, max_y,
//                             outer_x0, outer_y0, ...,
//                             hole0_x0, hole0_y0, ..., hole1_x0, ...]
void serializePolygons(const vector<Polygon>& polygons,
                       vector<int>&    int_buf,
                       vector<double>& double_buf)
{
    int_buf.clear();
    double_buf.clear();

    int_buf.push_back((int)polygons.size());

    for (const auto& poly : polygons) {
        int_buf.push_back(poly.id);
        int_buf.push_back((int)poly.outer.size());
        int_buf.push_back((int)poly.holes.size());
        for (const auto& hole : poly.holes)
            int_buf.push_back((int)hole.size());

        // bbox
        double_buf.push_back(poly.bbox.min_x);
        double_buf.push_back(poly.bbox.max_x);
        double_buf.push_back(poly.bbox.min_y);
        double_buf.push_back(poly.bbox.max_y);

        // outer boundary
        for (const auto& pt : poly.outer) {
            double_buf.push_back(pt.x);
            double_buf.push_back(pt.y);
        }

        // holes
        for (const auto& hole : poly.holes)
            for (const auto& pt : hole) {
                double_buf.push_back(pt.x);
                double_buf.push_back(pt.y);
            }
    }
}

// ---------------------------------------------------------------------------
// deserializePolygons
// ---------------------------------------------------------------------------
vector<Polygon> deserializePolygons(const vector<int>&    int_buf,
                                    const vector<double>& double_buf)
{
    int ii = 0;   // index into int_buf
    int di = 0;   // index into double_buf

    int num_polygons = int_buf[ii++];
    vector<Polygon> polygons(num_polygons);

    for (int p = 0; p < num_polygons; p++) {
        Polygon& poly = polygons[p];

        poly.id              = int_buf[ii++];
        int outer_count      = int_buf[ii++];
        int hole_count       = int_buf[ii++];

        vector<int> hole_sizes(hole_count);
        for (int h = 0; h < hole_count; h++)
            hole_sizes[h] = int_buf[ii++];

        // bbox
        poly.bbox.min_x = double_buf[di++];
        poly.bbox.max_x = double_buf[di++];
        poly.bbox.min_y = double_buf[di++];
        poly.bbox.max_y = double_buf[di++];

        // outer boundary
        poly.outer.resize(outer_count);
        for (int i = 0; i < outer_count; i++) {
            poly.outer[i].x = double_buf[di++];
            poly.outer[i].y = double_buf[di++];
        }

        // holes
        poly.holes.resize(hole_count);
        for (int h = 0; h < hole_count; h++) {
            poly.holes[h].resize(hole_sizes[h]);
            for (int i = 0; i < hole_sizes[h]; i++) {
                poly.holes[h][i].x = double_buf[di++];
                poly.holes[h][i].y = double_buf[di++];
            }
        }
    }

    return polygons;
}

// ---------------------------------------------------------------------------
// broadcastPolygons
// ---------------------------------------------------------------------------
vector<Polygon> broadcastPolygons(vector<Polygon>& polygons, int rank)
{
    vector<int>    int_buf;
    vector<double> double_buf;

    if (rank == 0)
        serializePolygons(polygons, int_buf, double_buf);

    // Exchange buffer sizes so every rank can allocate before the data broadcast.
    int int_sz = (rank == 0) ? (int)int_buf.size()    : 0;
    int dbl_sz = (rank == 0) ? (int)double_buf.size() : 0;
    MPI_Bcast(&int_sz, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dbl_sz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        int_buf.resize(int_sz);
        double_buf.resize(dbl_sz);
    }

    // Broadcast the actual data.
    MPI_Bcast(int_buf.data(),    int_sz, MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Bcast(double_buf.data(), dbl_sz, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0)
        return polygons;   // already have them
    return deserializePolygons(int_buf, double_buf);
}
