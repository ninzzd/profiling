#include <fstream>
#include <vector>
#include <cstdint>
#include <chrono>
#include <iostream>

#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>

int main() {
    int n, d;

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
    std::cout << "Building FlatIP index...\n";
    start = std::chrono::high_resolution_clock::now();
    faiss::IndexFlatL2 index(d);
    index.add(n, xb.data());
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Build complete!\n";
    std::cout << "Build time: "
              << std::chrono::duration<double>(end - start).count()
              << " s\n";

    // Writing index to .fbin file
    std::cout << "Writing index to disk...\n";
    start = std::chrono::high_resolution_clock::now();
    faiss::write_index(&index, "./cpu-baseline.index");
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Write time: "
              << std::chrono::duration<double>(end - start).count()
              << " s\n";

    return 0;
}