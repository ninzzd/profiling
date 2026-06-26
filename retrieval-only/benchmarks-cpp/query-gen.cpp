#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cstring>
#include <assert.h>
// Objective: To geerate a common query batch for all benchmarks

int main(int argc, char** argv){
    int nq, nb;
    int n, d, k;

    if(argc != 4){
        std::cerr << "Incorrect syntax: query_gen <nq> <k> <nb>\n";
        return -1;
    }
    nq = std::stoi(argv[1]);
    k = std::stoi(argv[2]);
    nb = std::stoi(argv[3]);

    std::ifstream qryin("./../datasets/wikiall/queries.fbin", std::ios::binary);
    std::ifstream gtin("./../datasets/wikiall/groundtruth.1M.neighbours.fbin", std::ios::binary);

    std::cout << "Reading dataset from disk..." << std::endl;
    auto t0 = std::chrono::high_resolution_clock::now();
    qryin.read(reinterpret_cast<char*>(&n), sizeof(int));
    qryin.read(reinterpret_cast<char*>(&d), sizeof(int));
    std::vector<float> xb(static_cast<size_t>(n) * d);
    qryin.read(reinterpret_cast<char*>(xb.data()),
            xb.size() * sizeof(float));
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Read complete!" << std::endl;
    std::cout << "Read time: "
              << std::chrono::duration<double>(t1 - t0).count()
              << " s\n";
    assert(nq <= n);
    std::vector<float> xq(nq * d);
    std::vector<float> gt(nq * k * d);
    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<int> dist(0,(int)n-1);
    
    std::cout << "Creating query vectors..." << std::endl;
    auto t2 = std::chrono::high_resolution_clock::now();
    // additional code required for multiple batch sampling
    for (int i = 0; i < nq; i++) {
        memcpy(xq.data() + i * d, xb.data() + dist(gen) * d, d * sizeof(float));
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    std::cout << "Finished query vector selection and batch creation" << std::endl;
    std::cout << "Query batch creation time: "
              <<  std::chrono::duration<double>(t3 - t2).count()
              << "s\n";
    
    std::cout << "Writing query batch to disk.." << std::endl;
    auto t4 = std::chrono::high_resolution_clock::now();
    std::ofstream out("query.fbin",std::ios::binary);
    out.write(reinterpret_cast<char*>(&nq), sizeof(int));
    out.write(reinterpret_cast<char*>(&d), sizeof(int));
    out.write(reinterpret_cast<char*>(xq.data()), static_cast<size_t>(nq*d*sizeof(float)));
    auto t5 = std::chrono::high_resolution_clock::now();
    std::cout << "Finished writing query batch" << std::endl;
    std::cout << "Query batch write time: "
              <<  std::chrono::duration<double>(t5 - t4).count()
              << "s\n";

    return 0;
}