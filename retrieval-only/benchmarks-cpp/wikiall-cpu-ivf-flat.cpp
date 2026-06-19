#include <fstream>
#include <vector>
#include <cstdint>
#include <random>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <cstring>
#include <chrono>
#include <iostream>

int main(int argc, char** argv) {
    int32_t n, d;
    constexpr int nq = 100;     // number of query vectors
    constexpr int k = 10;       // no. of nearest neighbors to retrieve    
    constexpr int nlist = 100;  // number of Voronoi cells (clusters) for IVF
    constexpr int nprobe = 10; // number of Voronoi cells to probe at query time0
    std::cout << "IVF Properties:" << std::endl;
    std::cout << "nlist: " << nlist << std::endl;
    std::cout << "nprobe: " << nprobe << std::endl;
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
    std::cout << "Index size: " << n << " \n";
    std::cout << "Index dimensions:" << d << std::endl;
    std::cout << "Read time: "
              << std::chrono::duration<double>(t1 - t0).count()
              << " s\n";

    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<int> dist(0,(int)n-1);  // integers in [0, 99]

    std::vector<float> xq(static_cast<size_t>(nq) * d);
    std::cout << "Generating query vectors..." << std::endl;
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < nq; ++i) {
        memcpy(xq.data() + i * d, xb.data() + dist(gen) * d, d * sizeof(float));
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    std::cout << "Query generation complete!" << std::endl;
    std::cout << "Query generation time: "
              << std::chrono::duration<double>(t3 - t2).count()
              << " s\n";

    faiss::IndexFlatL2 centroids(d); // index that stores centroid vectors in flat contiguous array
    faiss::IndexIVFFlat index(&centroids, d, nlist, faiss::METRIC_L2); // IVF index with flat quantization
    
    std::cout << "Training the IVF index..." << std::endl;
    auto t4 = std::chrono::high_resolution_clock::now();
    index.train(n, xb.data()); // train the IVF index with the dataset
    auto t5 = std::chrono::high_resolution_clock::now();
    std::cout << "Training complete!" << std::endl;
    std::cout << "Training time: "
              << std::chrono::duration<double>(t5 - t4).count()
              << " s\n";

    std::cout << "Adding vectors to the IVF index..." << std::endl;
    auto t6 = std::chrono::high_resolution_clock::now();
    index.add(n, xb.data());
    auto t7 = std::chrono::high_resolution_clock::now();
    std::cout << "Index created!" << std::endl;
    std::cout << "Index creation latency: "
              << std::chrono::duration<double>(t7 - t6).count()
              << " s\n";

    index.nprobe = nprobe;
    
    std::vector<faiss::idx_t> labels(static_cast<size_t>(nq) * k);
    std::vector<float> distances(static_cast<size_t>(nq) * k);

    auto t8 = std::chrono::high_resolution_clock::now();
    index.search(nq, xq.data(), k, distances.data(), labels.data());
    auto t9 = std::chrono::high_resolution_clock::now();

    std::cout << "Search complete!" << std::endl;
    std::cout << "Latency: "
              << std::chrono::duration<double>(t9 - t8).count()
              << " s\n";
    std::cout << "Latency/query: "
              << std::chrono::duration<double>(t9 - t8).count() / nq
              << " s\n";
    std::cout << "Throughput: "
              << nq / std::chrono::duration<double>(t9 - t8).count()
              << " queries/s\n";
    return 0;
}
