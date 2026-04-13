#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include <string>
#include "structures.h"

using namespace std;

vector<Polygon> loadPolygons(const string& filename);
vector<Point> loadPoints(const string& filename);
void generateUniformPoints(int count, double minX, double maxX, double minY, double maxY, const string& filename);
void generateClusteredPoints(int count, const string& filename);
void generateMultipleSizes(const string& folder);
void generateTestCases(const string& filename);

#endif