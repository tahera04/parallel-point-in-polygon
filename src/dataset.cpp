#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>
#include "../include/structures.h"
#include <iomanip>

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
        double cx = randFloat(450000, 550000);
        double cy = randFloat(450000, 550000);
        double radius = randFloat(15000.0, 40000.0);

        // 500 to 1000 vertices with high noise - very complex boundary
        int outerVertices = 1500 + rand() % 1500;
        vector<Point> outer = generateComplexRing(cx, cy, radius, outerVertices, 0.45);

        // 8 to 15 holes per polygon
        int numHoles = 8 + rand() % 8;
        vector<vector<Point>> holes;

        for (int h = 0; h < numHoles; h++) {
            double hcx = cx + randFloat(-radius * 0.5, radius * 0.5);
            double hcy = cy + randFloat(-radius * 0.5, radius * 0.5);
            double hRadius = radius * randFloat(0.03, 0.08);

            // 200 to 400 vertices per hole
            int holeVertices = 500 + rand() % 500;
            vector<Point> hole = generateComplexRing(hcx, hcy, hRadius, holeVertices, 0.3);
            holes.push_back(hole);
        }

        file << id << " " << outerVertices << " " << numHoles << "\n";
        file << fixed << setprecision(8);
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

    cout << "Generating " << numMultiPolygons << " complex multi-polygons..." << endl;

    double mapSize = 1000000.0;
    int id = 1;

    for (int i = 0; i < numMultiPolygons; i++) {
        // 5 to 8 parts per multi-polygon
        int numParts = 5 + rand() % 4;

        file << "MULTI " << id << " " << numParts << "\n";

        for (int part = 0; part < numParts; part++) {
            double cx = randFloat(10000, mapSize - 10000);
            double cy = randFloat(10000, mapSize - 10000);
            double radius = randFloat(1000.0, 5000.0);

            // 400 to 800 vertices per part
            int outerVertices = 400 + rand() % 400;
            vector<Point> outer = generateComplexRing(cx, cy, radius, outerVertices, 0.4);

            // 6 to 10 holes per part
            int numHoles = 6 + rand() % 5;
            vector<vector<Point>> holes;

            for (int h = 0; h < numHoles; h++) {
                double hcx = cx + randFloat(-radius * 0.4, radius * 0.4);
                double hcy = cy + randFloat(-radius * 0.4, radius * 0.4);
                double hRadius = radius * randFloat(0.03, 0.07);
                int holeVertices = 150 + rand() % 150;
                vector<Point> hole = generateComplexRing(hcx, hcy, hRadius, holeVertices, 0.25);
                holes.push_back(hole);
            }

            file << outerVertices << " " << numHoles << "\n";
            file << fixed << setprecision(8);
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

    // cluster points
    double centerX = 500000;
    double centerY = 500000;

    for (int i = 0; i < count; i++) {
        double x = centerX + randFloat(-30000, 30000);
        double y = centerY + randFloat(-30000, 30000);
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