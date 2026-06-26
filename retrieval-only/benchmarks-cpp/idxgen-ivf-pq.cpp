#include <fstream>
#include <vector>
#include <cstdint>
#include <chrono>
#include <iostream>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/index_io.h>

int main(int argc, char** argv) {
    int n, d;
    int nlist, nprobe, nbits, m;

    if(argc != 5){
        std::cerr << "Incorrect syntax: idxgen_ivf_pq <nlist> <nprobe> <nbits> <m>\n";
        return -1;
    }
    nlist = std::stoi(argv[1]);
    nprobe = std::stoi(argv[2]);
    nbits = std::stoi(argv[3]);
    m = std::stoi(argv[4]);

    // Reading dataset .fbin file
    std::ifstream in("./../datasets/wikiall/base.1M.fbin", std::ios::binary);
    if (!in) {
        std::cerr << "Could not open dataset.\n";
        return -1;
    }

    std::cout << "Reading dataset...\n";
    auto start = std::chrono::high_resolution_clock::now();
    in.read(reinterpret_cast<char*>(&n), sizeof(int32_t));
    in.read(reinterpret_cast<char*>(&d), sizeof(int32_t));
    std::vector<float> xb(static_cast<size_t>(n) * d);
    in.read(reinterpret_cast<char*>(xb.data()),
            xb.size() * sizeof(float));
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Read complete!\n";
    std::cout << "Read time: "
              << std::chrono::duration<double>(end - start).count()
              << " s\n";
    
    // Building faiss index object
    std::cout << "Building IVF-PQ index...\n";
    start = std::chrono::high_resolution_clock::now();
    faiss::IndexFlatIP cq(d);
    faiss::IndexIVFPQ index(&cq,d,nlist,m,nbits,faiss::METRIC_INNER_PRODUCT);
    index.nprobe = nprobe;
    index.train(n, xb.data());
    index.add(n, xb.data());
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Build complete!\n";
    std::cout << "Build time: "
              << std::chrono::duration<double>(end - start).count()
              << " s\n";

    // Writing index to .fbin file
    std::cout << "Writing index to disk...\n";
    start = std::chrono::high_resolution_clock::now();
    faiss::write_index(&index, "./cpu-ivf-pq.index");
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Write time: "
              << std::chrono::duration<double>(end - start).count()
              << " s\n";

    return 0;
}