#ifndef MPI_UTILS_H
#define MPI_UTILS_H

#include <vector>
#include <mpi.h>
#include "structures.h"

using namespace std;

// Flatten a polygon vector into two contiguous buffers suitable for MPI_Bcast.
//
// int_buf layout  : [num_polygons, (id, outer_count, hole_count, hole_0_size, ...) ...]
// double_buf layout: [(bbox.min_x, max_x, min_y, max_y, outer pts x/y ..., hole pts x/y ...) ...]
void serializePolygons(const vector<Polygon>& polygons,
                       vector<int>&    int_buf,
                       vector<double>& double_buf);

// Reconstruct a polygon vector from the two flat buffers produced by serializePolygons.
vector<Polygon> deserializePolygons(const vector<int>&    int_buf,
                                    const vector<double>& double_buf);

// Broadcast the full polygon dataset from rank 0 to every rank.
// rank 0: serializes polygons, broadcasts sizes, broadcasts data, returns polygons unchanged.
// other: receives sizes, allocates buffers, receives data, deserializes, returns result.
vector<Polygon> broadcastPolygons(vector<Polygon>& polygons, int rank);

#endif
