#include "../include/structures.h"
#include "../include/spatial-index.h"
#include <vector>
#include <queue>
#include <mutex>
#include <chrono>
#include <omp.h>
#include <iostream>

using namespace std;

bool isPointInsidePolygon(Point p, Polygon& poly);

using Batch = vector<int>;

vector<int> classifyWithDynamicQueue(
    vector<Polygon>& polygons,
    const Quadtree& spatialIndex,
    const vector<Point>& queryPoints,
    int batchSize,
    int numThreads)
{
    queue<Batch> taskQueue;
    mutex queueMutex;

    int n = (int)queryPoints.size();

    // -------------------------------
    // Build task queue
    // -------------------------------
    for (int i = 0; i < n; i += batchSize) {
        Batch batch;
        int end = min(i + batchSize, n);

        for (int j = i; j < end; j++) {
            batch.push_back(j);
        }

        taskQueue.push(batch);
    }

    vector<int> results(n, -1);

    cout << "\n[Load Balancer] Dynamic Task Queue Started\n";
    cout << "[Load Balancer] Points  : " << n << "\n";
    cout << "[Load Balancer] Batch   : " << batchSize << "\n";
    cout << "[Load Balancer] Threads : " << numThreads << "\n";
    cout << "[Load Balancer] Batches : " << taskQueue.size() << "\n\n";

    auto start = chrono::high_resolution_clock::now();

    #pragma omp parallel num_threads(numThreads)
    {
        while (true) {

            Batch batch;

            // ---- critical section: queue access only ----
            {
                lock_guard<mutex> lock(queueMutex);

                if (taskQueue.empty()) {
                    break;
                }

                batch = taskQueue.front();
                taskQueue.pop();
            }

            for (int idx : batch) {
                const Point& p = queryPoints[idx];

                vector<int> candidates = spatialIndex.query(p);

                for (int c : candidates) {
                    if (isPointInsidePolygon(p, polygons[c])) {
                        results[idx] = polygons[c].id;
                        break;
                    }
                }
            }
        }
    }

    auto end = chrono::high_resolution_clock::now();
    double timeTaken = chrono::duration<double>(end - start).count();

    int inside = 0;
    for (int r : results)
        if (r != -1) inside++;

    cout << "[Load Balancer] Done\n";
    cout << "[Load Balancer] Time    : " << timeTaken << " sec\n";
    cout << "[Load Balancer] Inside  : " << inside << "\n";
    cout << "[Load Balancer] Outside : " << n - inside << "\n\n";

    return results;
}