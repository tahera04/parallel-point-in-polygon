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

    // Define 4 cluster centers (like city centers)
    double centersX[] = {100, 300, 500, 700};
    double centersY[] = {100, 400, 200, 500};
    int numClusters = 4;

    for (int i = 0; i < count; i++) {
        // Pick a random cluster center
        int c = rand() % numClusters;

        // Generate a point near that center with small random offset
        double x = centersX[c] + (rand() / (double)RAND_MAX) * 80 - 40;
        double y = centersY[c] + (rand() / (double)RAND_MAX) * 80 - 40;

        file << x << " " << y << "\n";
    }

    file.close();
    cout << "Generated " << count << " clustered points to " << filename << endl;
}