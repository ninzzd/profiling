#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cstring>
// Objective: To geerate a common query batch for all benchmarks

int main(int argc, char** argv){
    int nq = 1;
    int32_t n, d;

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

    if (argc < 2)
        std::cout << "No. of queries not specified. Defaulting to single query." << std::endl;
    else
        nq = std::stoi(argv[1]);
    
    std::vector<float> xq(static_cast<size_t>(nq * d));
    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<int> dist(0,(int)n-1);
    
    for (int i = 0; i < nq; i++) {
        memcpy(xq.data() + i * d, xb.data() + dist(gen) * d, d * sizeof(float));
    }
    
    std::ofstream out("query.fbin",std::ios::binary);
    out.write(reinterpret_cast<char*>(n), sizeof(int32_t));
    out.write(reinterpret_cast<char*>(d), sizeof(int32_t));
    return 0;
}