#include <iostream>
#include <fstream>
#include <vector>
#include "../include/structures.h"

using namespace std;

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

        // Read outer boundary vertices
        for (int i = 0; i < outerCount; i++) {
            Point p;
            file >> p.x >> p.y;
            poly.outer.push_back(p);
        }

        // Read holes
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

void generateUniformPoints(int count, double minX, double maxX, double minY, double maxY, const string& filename) {
    ofstream file(filename);

    if (!file.is_open()) {
        cout << "Error: Could not create file " << filename << endl;
        return;
    }

    for (int i = 0; i < count; i++) {
        double x = minX + (rand() / (double)RAND_MAX) * (maxX - minX);
        double y = minY + (rand() / (double)RAND_MAX) * (maxY - minY);
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

    double centersX[] = {100, 300, 500, 700};
    double centersY[] = {100, 400, 200, 500};
    int numClusters = 4;

    for (int i = 0; i < count; i++) {
        int c = rand() % numClusters;
        double x = centersX[c] + (rand() / (double)RAND_MAX) * 80 - 40;
        double y = centersY[c] + (rand() / (double)RAND_MAX) * 80 - 40;
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
        generateUniformPoints(sizes[i], 0, 800, 0, 600, uniformFile);
        generateClusteredPoints(sizes[i], clusteredFile);
    }
}

void generateTestCases(const string& filename) {
    ofstream file(filename);

    if (!file.is_open()) {
        cout << "Error: Could not create file " << filename << endl;
        return;
    }

    // Clearly inside polygon 1 (simple square 0,0 to 10,10)
    file << "5 5\n";

    // Clearly outside all polygons
    file << "500 500\n";

    // Point on edge of polygon 1
    file << "5 0\n";

    // Point on vertex of polygon 1
    file << "0 0\n";

    // Inside polygon 2 outer boundary but inside hole (should be outside)
    file << "35 35\n";

    // Inside polygon 2 outer boundary and outside hole (should be inside)
    file << "22 22\n";

    // Inside complex polygon 3
    file << "150 60\n";

    // Outside complex polygon 3
    file << "50 50\n";

    file.close();
    cout << "Generated test cases to " << filename << endl;
}