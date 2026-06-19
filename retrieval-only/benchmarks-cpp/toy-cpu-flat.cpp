#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include <faiss/IndexFlat.h>

int main() {
    // -------------------------------
    // Configuration
    // -------------------------------
    constexpr faiss::idx_t nb = 100000;   // database size
    constexpr faiss::idx_t nq = 1000;     // number of queries
    constexpr int d = 768;                // embedding dimension
    constexpr int k = 10;                 // top-k

    // -------------------------------
    // Generate random data
    // -------------------------------
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> xb(nb * d); // row-major flatenned array of database vectors
    std::vector<float> xq(nq * d); // row-major flatenned array of query vectors

    for (auto& x : xb) x = dist(rng); // initializing xb with random values in U[0,1]
    for (auto& x : xq) x = dist(rng); // initializing xq with random values in U[0,1]


    // -------------------------------
    // Build index
    // -------------------------------
    faiss::IndexFlatL2 index(d); // Initialize db with 768 as no. of embedding dimensions

    auto t0 = std::chrono::high_resolution_clock::now();
    index.add(nb, xb.data()); // database initialization by adding vectors
    auto t1 = std::chrono::high_resolution_clock::now();

    std::cout << "Index size: " << index.ntotal << " vectors\n";
    std::cout << "Add time: "
              << std::chrono::duration<double>(t1 - t0).count()
              << " s\n";

    // -------------------------------
    // Search
    // -------------------------------
    std::vector<faiss::idx_t> labels(nq * k); // top-k neighbor IDs for each query
    std::vector<float> distances(nq * k); // corresponding distances

    auto t2 = std::chrono::high_resolution_clock::now();
    index.search( // search for all query vectors in xq, find the IDs and distances of top-K nearest neighbours
        nq,
        xq.data(),
        k,
        distances.data(),
        labels.data()
    );
    auto t3 = std::chrono::high_resolution_clock::now();

    double search_time =
        std::chrono::duration<double>(t3 - t2).count();

    std::cout << "Search time: " << search_time << " s\n";
    std::cout << "Queries/sec: " << nq / search_time << "\n";
    std::cout << "Latency/query: "
              << (search_time / nq) * 1e6
              << " us\n";

    // Prevent compiler from optimizing away results, compiler may entire skip index.search() if it sees unused result arrays
    std::cout << "First neighbor ID: " << labels[0]
              << ", distance: " << distances[0] << "\n";

    return 0;
}