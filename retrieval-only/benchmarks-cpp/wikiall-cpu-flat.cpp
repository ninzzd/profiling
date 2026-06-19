#include <fstream>
#include <vector>
#include <cstdint>
#include <random>
#include <faiss/IndexFlat.h>
#include <cstring>
#include <chrono>
#include <iostream>

int main(int argc, char** argv) {
    int32_t n, d;
    constexpr int nq = 100;
    constexpr int k = 10;

    std::ifstream in("./../datasets/wikiall/base.1M.fbin", std::ios::binary);

    std::cout << "Reading dataset from disk..." << std::endl;
    auto t0 = std::chrono::high_resolution_clock::now();
    in.read(reinterpret_cast<char*>(&n), sizeof(int32_t));
    in.read(reinterpret_cast<char*>(&d), sizeof(int32_t));
    std::vector<float> xb(static_cast<size_t>(n) * d);
    in.read(reinterpret_cast<char*>(xb.data()),
            xb.size() * sizeof(float));
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Read complete!" << std::endl;
    std::cout << "Read time: "
              << std::chrono::duration<double>(t1 - t0).count()
              << " s\n";

    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<int> dist(0,(int)n-1);  // integers in [0, 99]

    std::vector<float> xq(static_cast<size_t>(nq) * d);
    for (int i = 0; i < nq; ++i) {
        memcpy(xq.data() + i * d, xb.data() + dist(gen) * d, d * sizeof(float));
    }

    faiss::IndexFlatL2 index(d);
    auto t2 = std::chrono::high_resolution_clock::now();
    index.add(n, xb.data());
    auto t3 = std::chrono::high_resolution_clock::now();
    std::cout << "Index creation complete!" << std::endl;
    std::cout << "Index size: " << index.ntotal << " vectors\n";
    std::cout << "Index dimensions:" << d << std::endl;
    std::cout << "Index creation latency: "
              << std::chrono::duration<double>(t3 - t2).count()
              << " s\n";

    std::vector<faiss::idx_t> labels(static_cast<size_t>(nq) * k);
    std::vector<float> distances(static_cast<size_t>(nq) * k);

    auto t4 = std::chrono::high_resolution_clock::now();
    index.search(nq, xq.data(), k, distances.data(), labels.data());
    auto t5 = std::chrono::high_resolution_clock::now();

    std::cout << "Search complete!" << std::endl;
    std::cout << "Latency: "
              << std::chrono::duration<double>(t5 - t4).count()
              << " s\n";
    std::cout << "Latency/query: "
              << std::chrono::duration<double>(t5 - t4).count() / nq
              << " s\n";
    std::cout << "Throughput: "
              << nq / std::chrono::duration<double>(t5 - t4).count()
              << " queries/s\n";
    return 0;
}