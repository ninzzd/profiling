// NOTE: requires FAISS built with FAISS_ENABLE_CUVS=ON (the "cuvs" build
// variant). GpuIndexCagra is only compiled in when cuVS is linked.
#include <fstream>
#include <vector>
#include <cstdint>
#include <chrono>
#include <iostream>

#include <faiss/IndexHNSW.h>
#include <faiss/gpu/GpuIndexCagra.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/index_io.h>

int main(int argc, char** argv) {
    int n, d;
    int intermediate_graph_degree, graph_degree;

    if(argc != 3){
        std::cerr << "Incorrect syntax: idxgen_gpu_cagra <intermediate_graph_degree> <graph_degree>\n";
        return -1;
    }
    intermediate_graph_degree = std::stoi(argv[1]);
    graph_degree = std::stoi(argv[2]);

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

    // Building CAGRA graph (the GPU, graph-based equivalent of HNSW)
    std::cout << "Building GPU CAGRA index...\n";
    start = std::chrono::high_resolution_clock::now();
    faiss::gpu::StandardGpuResources res;
    faiss::gpu::GpuIndexCagraConfig config;
    config.intermediate_graph_degree = intermediate_graph_degree;
    config.graph_degree = graph_degree;
    faiss::gpu::GpuIndexCagra index(&res, d, faiss::METRIC_L2, config);
    // train() builds the graph over the full base set in one call (no
    // separate add() needed unless custom ids are required)
    index.train(n, xb.data());
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Build complete!\n";
    std::cout << "Build time: "
              << std::chrono::duration<double>(end - start).count()
              << " s\n";

    // Copying to a CPU faiss::IndexHNSWCagra (the base-layer HNSW graph
    // built from CAGRA's kNN graph) for serialization
    std::cout << "Writing index to disk...\n";
    start = std::chrono::high_resolution_clock::now();
    faiss::IndexHNSWCagra cpu_index;
    index.copyTo(&cpu_index);
    faiss::write_index(&cpu_index, "./gpu-cagra.index");
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Write time: "
              << std::chrono::duration<double>(end - start).count()
              << " s\n";

    return 0;
}
