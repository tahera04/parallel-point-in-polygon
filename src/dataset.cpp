#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>
#include "../include/structures.h"

using namespace std;

// ==================== FILE INPUT ====================

vector<Polygon> loadPolygons(const string& filename) {
    vector<Polygon> polygons;
    ifstream file(filename);

    if (!file.is_open()) {
        cout << "Error: Could not open file " << filename << endl;
        return polygons;
    }

    int id, outerCount, holeCount;
    while (file >> id >> outerCount >> holeCount) {
        Polygon poly;
        poly.id = id;

        for (int i = 0; i < outerCount; i++) {
            Point p;
            file >> p.x >> p.y;
            poly.outer.push_back(p);
        }

        for (int h = 0; h < holeCount; h++) {
            int holeVertexCount;
            file >> holeVertexCount;
            vector<Point> hole;
            for (int i = 0; i < holeVertexCount; i++) {
                Point p;
                file >> p.x >> p.y;
                hole.push_back(p);
            }
            poly.holes.push_back(hole);
        }

        polygons.push_back(poly);
    }

    file.close();
    return polygons;
}

vector<Point> loadPoints(const string& filename) {
    vector<Point> points;
    ifstream file(filename);

    if (!file.is_open()) {
        cout << "Error: Could not open file " << filename << endl;
        return points;
    }

    Point p;
    while (file >> p.x >> p.y) {
        points.push_back(p);
    }

    file.close();
    return points;
}

// ==================== HELPERS ====================

double randFloat(double min, double max) {
    return min + (rand() / (double)RAND_MAX) * (max - min);
}

// Generate a complex polygon ring (circle-like with noise)
vector<Point> generateComplexRing(double centerX, double centerY, 
                                   double radius, int numVertices, 
                                   double noiseFactor) {
    vector<Point> ring;
    for (int i = 0; i < numVertices; i++) {
        double angle = 2.0 * M_PI * i / numVertices;
        double noise = 1.0 + randFloat(-noiseFactor, noiseFactor);
        double r = radius * noise;
        double x = centerX + r * cos(angle) + randFloat(-radius*0.05, radius*0.05);
        double y = centerY + r * sin(angle) + randFloat(-radius*0.05, radius*0.05);
        ring.push_back({x, y});
    }
    return ring;
}

// ==================== POLYGON GENERATION ====================

void generateComplexPolygons(int numPolygons, const string& filename) {
    ofstream file(filename);

    if (!file.is_open()) {
        cout << "Error: Could not create file " << filename << endl;
        return;
    }

    cout << "Generating " << numPolygons << " complex polygons..." << endl;

    double mapSize = 1000000.0;
    int id = 1;

    for (int i = 0; i < numPolygons; i++) {
        // Random center
        double cx = randFloat(10000, mapSize - 10000);
        double cy = randFloat(10000, mapSize - 10000);

        // Random radius between 500 and 5000
        double radius = randFloat(500.0, 5000.0);

        // Complex outer boundary: 100 to 300 vertices with high noise
        int outerVertices = 100 + rand() % 200;
        vector<Point> outer = generateComplexRing(cx, cy, radius, outerVertices, 0.35);

        // 3 to 7 holes per polygon
        int numHoles = 3 + rand() % 5;
        vector<vector<Point>> holes;

        for (int h = 0; h < numHoles; h++) {
            // Hole center offset from polygon center
            double hcx = cx + randFloat(-radius * 0.4, radius * 0.4);
            double hcy = cy + randFloat(-radius * 0.4, radius * 0.4);
            double hRadius = radius * randFloat(0.05, 0.12);

            // Each hole has 50 to 150 vertices
            int holeVertices = 50 + rand() % 100;
            vector<Point> hole = generateComplexRing(hcx, hcy, hRadius, holeVertices, 0.25);
            holes.push_back(hole);
        }

        // Write polygon to file
        file << id << " " << outerVertices << " " << numHoles << "\n";
        for (auto& p : outer) {
            file << p.x << " " << p.y << "\n";
        }
        for (auto& hole : holes) {
            file << hole.size() << "\n";
            for (auto& p : hole) {
                file << p.x << " " << p.y << "\n";
            }
        }

        id++;

        if (i % 1000 == 0) {
            cout << "Generated " << i << " / " << numPolygons << " polygons\r" << flush;
        }
    }

    file.close();
    cout << "\nDone! Written to " << filename << endl;
}

// ==================== MULTI-POLYGON GENERATION ====================

void generateMultiPolygons(int numMultiPolygons, const string& filename) {
    ofstream file(filename);

    if (!file.is_open()) {
        cout << "Error: Could not create file " << filename << endl;
        return;
    }

    cout << "Generating " << numMultiPolygons << " multi-polygons..." << endl;

    double mapSize = 1000000.0;
    int id = 1;

    for (int i = 0; i < numMultiPolygons; i++) {
        // Each multi-polygon has 3 to 6 parts
        int numParts = 3 + rand() % 4;

        file << "MULTI " << id << " " << numParts << "\n";

        for (int part = 0; part < numParts; part++) {
            double cx = randFloat(10000, mapSize - 10000);
            double cy = randFloat(10000, mapSize - 10000);
            double radius = randFloat(300.0, 3000.0);

            // 80 to 200 vertices per part
            int outerVertices = 80 + rand() % 120;
            vector<Point> outer = generateComplexRing(cx, cy, radius, outerVertices, 0.3);

            // 2 to 4 holes per part
            int numHoles = 2 + rand() % 3;
            vector<vector<Point>> holes;

            for (int h = 0; h < numHoles; h++) {
                double hcx = cx + randFloat(-radius * 0.35, radius * 0.35);
                double hcy = cy + randFloat(-radius * 0.35, radius * 0.35);
                double hRadius = radius * randFloat(0.04, 0.10);
                int holeVertices = 40 + rand() % 80;
                vector<Point> hole = generateComplexRing(hcx, hcy, hRadius, holeVertices, 0.2);
                holes.push_back(hole);
            }

            file << outerVertices << " " << numHoles << "\n";
            for (auto& p : outer) {
                file << p.x << " " << p.y << "\n";
            }
            for (auto& hole : holes) {
                file << hole.size() << "\n";
                for (auto& p : hole) {
                    file << p.x << " " << p.y << "\n";
                }
            }
        }

        id++;

        if (i % 500 == 0) {
            cout << "Generated " << i << " / " << numMultiPolygons << " multi-polygons\r" << flush;
        }
    }

    file.close();
    cout << "\nDone! Written to " << filename << endl;
}

// ==================== POINT GENERATION ====================

void generateUniformPoints(int count, double minX, double maxX, 
                            double minY, double maxY, const string& filename) {
    ofstream file(filename);

    if (!file.is_open()) {
        cout << "Error: Could not create file " << filename << endl;
        return;
    }

    for (int i = 0; i < count; i++) {
        double x = randFloat(minX, maxX);
        double y = randFloat(minY, maxY);
        file << x << " " << y << "\n";
    }

    file.close();
    cout << "Generated " << count << " uniform points to " << filename << endl;
}

void generateClusteredPoints(int count, const string& filename) {
    ofstream file(filename);

    if (!file.is_open()) {
        cout << "Error: Could not create file " << filename << endl;
        return;
    }

    // 10 cluster centers spread across map
    double centersX[] = {100000, 300000, 500000, 700000, 900000,
                         200000, 400000, 600000, 800000, 150000};
    double centersY[] = {100000, 400000, 200000, 500000, 300000,
                         700000, 900000, 600000, 800000, 500000};
    int numClusters = 10;

    for (int i = 0; i < count; i++) {
        int c = rand() % numClusters;
        double x = centersX[c] + randFloat(-20000, 20000);
        double y = centersY[c] + randFloat(-20000, 20000);
        file << x << " " << y << "\n";
    }

    file.close();
    cout << "Generated " << count << " clustered points to " << filename << endl;
}

void generateMultipleSizes(const string& folder) {
    int sizes[] = {1000, 10000, 100000, 1000000};
    int numSizes = 4;

    for (int i = 0; i < numSizes; i++) {
        string uniformFile = folder + "/uniform_" + to_string(sizes[i]) + ".txt";
        string clusteredFile = folder + "/clustered_" + to_string(sizes[i]) + ".txt";
        generateUniformPoints(sizes[i], 0, 1000000, 0, 1000000, uniformFile);
        generateClusteredPoints(sizes[i], clusteredFile);
    }
}

// ==================== TEST CASES ====================

void generateTestCases(const string& filename) {
    ofstream file(filename);

    if (!file.is_open()) {
        cout << "Error: Could not create file " << filename << endl;
        return;
    }

    // Clearly inside a polygon
    file << "5.0 5.0\n";
    // Clearly outside all polygons
    file << "999999.9 999999.9\n";
    // Point on edge
    file << "5.0 0.0\n";
    // Point on vertex
    file << "0.0 0.0\n";
    // Inside outer boundary but inside hole (outside)
    file << "35.123456 35.123456\n";
    // Inside outer boundary outside hole (inside)
    file << "22.654321 22.654321\n";
    // Floating point precision edge case
    file << "10.0000001 10.0000001\n";
    // Negative-adjacent boundary
    file << "0.0000001 0.0000001\n";

    file.close();
    cout << "Generated test cases to " << filename << endl;
}