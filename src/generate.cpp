#include <iostream>
#include <cstdlib>
#include "../include/dataset.h"

using namespace std;

int main() {
    srand(42);

    cout << "=== Generating Complex Polygons ===" << endl;
    generateComplexPolygons(50000, "data/polygons.txt");

    cout << "=== Generating Multi-Polygons ===" << endl;
    generateMultiPolygons(10000, "data/multipolygons.txt");

    cout << "=== Generating Point Datasets ===" << endl;
    generateMultipleSizes("data");

    cout << "=== Generating Test Cases ===" << endl;
    generateTestCases("data/testcases.txt");

    cout << "=== All Done! ===" << endl;
    return 0;
}