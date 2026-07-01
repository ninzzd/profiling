#include <fstream>
#include <sstream>
#include <filesystem>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cstring>
#include <assert.h>
#include <faiss/Index.h>
// Objective: To geerate a common query batch for all benchmarks

int main(int argc, char** argv){
    uint32_t nq, nb; // batch size, no. of distinct batches
    uint32_t n, d, k; // total number of query vectors, dimensions and top-k
    uint32_t ngt, kgt; // no. of groundtruth entries, no. of labels per entry

    if(argc != 4){
        std::cerr << "Incorrect syntax: query_gen <nq> <k> <nb>\n";
        return -1;
    }
    nq = std::stoi(argv[1]);
    k = std::stoi(argv[2]);
    nb = std::stoi(argv[3]);

    std::ifstream qryin("./../datasets/wikiall/queries.fbin", std::ios::binary);
    std::ifstream gtin("./../datasets/wikiall/groundtruth.1M.neighbors.ibin", std::ios::binary);

    

    std::cout << "Reading query dataset and groundtruth labels from disk..." << std::endl;
    auto t0 = std::chrono::high_resolution_clock::now();
    qryin.read(reinterpret_cast<char*>(&n), sizeof(int));
    qryin.read(reinterpret_cast<char*>(&d), sizeof(int));
    assert(nq <= n);
    std::vector<float> xb(static_cast<size_t>(n) * d);
    qryin.read(reinterpret_cast<char*>(xb.data()), xb.size() * sizeof(float));
    gtin.read(reinterpret_cast<char*>(&ngt), sizeof(uint32_t));
    gtin.read(reinterpret_cast<char*>(&kgt), sizeof(uint32_t));
    assert(ngt == n);
    assert(k <= kgt);
    std::vector<uint32_t> xgt(static_cast<size_t>(ngt) * kgt);
    gtin.read(reinterpret_cast<char*>(xgt.data()), xgt.size() * sizeof(uint32_t));
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Read complete!" << std::endl;
    std::cout << "Read time: "
              << std::chrono::duration<double>(t1 - t0).count()
              << " s\n";

    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<int> dist(0,(int)n-1);

    for(int i = 1;i <= nb;i++){
        std::cout << "Batch " << i << ": ";
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<float> xq(nq * d);
        std::vector<uint32_t> gt(nq * k);
        
        // std::cout << "Creating query vectors..." << std::endl;
        // auto t2 = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < nq; j++) {
            int temp = dist(gen);
            memcpy(xq.data() + j * d, xb.data() + temp * d, d * sizeof(float));
            memcpy(gt.data() + j * k, xgt.data() + temp * kgt, k*sizeof(uint32_t));
        }
        // auto t3 = std::chrono::high_resolution_clock::now();
        // std::cout << "Finished query vector selection and batch creation" << std::endl;
        // std::cout << "Query batch creation time: "
        //         <<  std::chrono::duration<double>(t3 - t2).count()
        //         << "s\n";
        
        // std::cout << "Writing query batch to disk.." << std::endl;
        // auto t4 = std::chrono::high_resolution_clock::now();
        std::stringstream qryfname;
        std::stringstream gtfname;


        std::filesystem::create_directories("./queries");
        std::filesystem::create_directories("./gt");

        qryfname << "./queries/query" << i << ".fbin";
        gtfname << "./gt/gt" << i << ".fbin";

        std::ofstream qryout(qryfname.str(),std::ios::binary);
        std::ofstream gtout(gtfname.str(),std::ios::binary);
        qryout.write(reinterpret_cast<char*>(&nq), sizeof(uint32_t));
        qryout.write(reinterpret_cast<char*>(&d), sizeof(uint32_t));
        qryout.write(reinterpret_cast<char*>(xq.data()), static_cast<size_t>(nq*d*sizeof(float)));
        gtout.write(reinterpret_cast<char*>(&nq), sizeof(uint32_t));
        gtout.write(reinterpret_cast<char*>(&k), sizeof(uint32_t));
        gtout.write(reinterpret_cast<char*>(gt.data()), static_cast<size_t>(nq*k*sizeof(uint32_t)));
        // auto t5 = std::chrono::high_resolution_clock::now();
        // std::cout << "Finished writing query batch" << std::endl;
        // std::cout << "Query batch write time: "
        //         <<  std::chrono::duration<double>(t5 - t4).count()
        //         << "s\n";
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Complete (" << std::chrono::duration<double>(end - start).count() << "s)\n";
    }
    return 0;
}