#include <iostream>
#include <cstdlib>
#include "../include/dataset.h"

using namespace std;

int main() {
    srand(42);

    cout << "=== Generating Complex Polygons ===" << endl;
    generateComplexPolygons(200000, "data/polygons.txt");

    cout << "=== Generating Multi-Polygons ===" << endl;
    generateMultiPolygons(20000, "data/multipolygons.txt");

    cout << "=== Generating Point Datasets ===" << endl;
    generateUniformPoints(50000000, 0, 1000000, 0, 1000000, "data/points_uniform.txt");
    generateClusteredPoints(50000000, "data/points_clustered.txt");

    cout << "=== Generating Test Cases ===" << endl;
    generateTestCases("data/testcases.txt");

    cout << "=== All Done! ===" << endl;
    return 0;
}