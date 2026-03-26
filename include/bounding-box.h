#ifndef BOUNDING_BOX_H
#define BOUNDING_BOX_H

#include "structures.h"

BoundingBox computeBoundingBox(const vector<Point>& vertices);

void assignBoundingBox(Polygon& poly);

bool pointInsideBoundingBox(const Point& pt, const BoundingBox& box);

#endif