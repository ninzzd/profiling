#include <fstream>
#include <vector>
#include <cstdint>
#include <random>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <cstring>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <cmath>
// Usage: ./wikiall-cpu-ivf-flat <mode> <num_iterations>
int main(int argc, char** argv) {
    int32_t n, d;
    int iter = 1;
    int mode = 0;
    constexpr int nq = 100;     // number of query vectors
    constexpr int k = 10;       // no. of nearest neighbors to retrieve    
    constexpr int nlist = 100;  // number of Voronoi cells (clusters) for IVF
    constexpr int nprobe = 10;  // number of Voronoi cells to probe at query time
    constexpr int nbits = 8;    // number of bits per subspace
    constexpr int m = 64;       // number of subspaces

    if (argc > 2) {
        iter = std::max(std::stoi(argv[2]),1);
        mode = !(!std::stoi(argv[1])); // 0 for silent, 1 for verbose
    }

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

    faiss::IndexFlatL2 centroids(d); // index that stores centroid vectors in flat contiguous array
    faiss::IndexIVFPQ index(&centroids, d, nlist, m, nbits, faiss::METRIC_L2); // IVF index with flat quantization
    
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
    std::vector<double> latency_vec;
    std::vector<double> throughput_vec;

    for(int i = 1;i <= iter;i++){
        std::cout << "Iteration: " << i << std::endl;
        std::vector<faiss::idx_t> labels(static_cast<size_t>(nq) * k);
        std::vector<float> distances(static_cast<size_t>(nq) * k);

        std::vector<float> xq(static_cast<size_t>(nq) * d);
        if(mode){
            std::cout << "Generating query vectors..." << std::endl;
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < nq; ++i) {
            memcpy(xq.data() + i * d, xb.data() + dist(gen) * d, d * sizeof(float));
        }
        auto t3 = std::chrono::high_resolution_clock::now();

        if(mode){
            std::cout << "Query generation complete!" << std::endl;
            std::cout << "Query generation time: "
                    << std::chrono::duration<double>(t3 - t2).count()
                    << " s\n";
        }

        auto t8 = std::chrono::high_resolution_clock::now();
        index.search(nq, xq.data(), k, distances.data(), labels.data());
        auto t9 = std::chrono::high_resolution_clock::now();

        double latency = std::chrono::duration<double>(t9 - t8).count();
        double throughput = nq/latency;
        if(mode){
            std::cout << "Search complete!" << std::endl;
            std::cout << "Latency: "
                    <<  latency
                    << " s\n";
            std::cout << "Latency/query: "
                    << std::chrono::duration<double>(t9 - t8).count() / nq
                    << " s\n";
            std::cout << "Throughput: "
                    <<  throughput
                    << " queries/s\n";
        }
        latency_vec.push_back(latency);
        throughput_vec.push_back(throughput);
    }
    std::sort(latency_vec.begin(),latency_vec.end());
    double p50lat = latency_vec[(int)floor(iter/2.0)];
    double p90lat = latency_vec[(int)floor(9.0*iter/10.0)];
    double p95lat = latency_vec[(int)floor(9.5*iter/10.0)];
    double p99lat = latency_vec[(int)floor(9.9*iter/10.0)];
    double avglat = std::accumulate(latency_vec.begin(),latency_vec.end(),0.0)/iter;
    double peakthr = *std::max_element(throughput_vec.begin(),throughput_vec.end());
    double avgthr = std::accumulate(throughput_vec.begin(),throughput_vec.end(),0.0)/iter;

    // std::cout << std::fixed << std::setprecision(9);
    std::cout << "P50 Latency: " << p50lat << "s" << std::endl;
    std::cout << "P90 Latency: " << p90lat << "s" << std::endl;
    std::cout << "P95 Latency: " << p95lat << "s" << std::endl;
    std::cout << "P99 Latency: " << p99lat << "s" << std::endl;
    std::cout << "Mean Latency: " << avglat << "s" << std::endl;
    std::cout << "Peak Throughput: " << peakthr << "qps" << std::endl;
    std::cout << "Mean Throughput: " << avgthr << "qps" << std::endl;
    return 0;
}
