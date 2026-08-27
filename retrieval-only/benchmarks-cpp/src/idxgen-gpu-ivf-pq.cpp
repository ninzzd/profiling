#include <fstream>
#include <vector>
#include <cstdint>
#include <chrono>
#include <iostream>

#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/GpuIndexIVFPQ.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/index_io.h>

int main(int argc, char** argv) {
    int n, d;
    int nlist, nprobe, nbits, m;

    if(argc != 5){
        std::cerr << "Incorrect syntax: idxgen_gpu_ivf_pq <nlist> <nprobe> <nbits> <m>\n";
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

    // Building faiss GPU index object
    std::cout << "Building GPU IVF-PQ index...\n";
    start = std::chrono::high_resolution_clock::now();
    faiss::gpu::StandardGpuResources res;
    faiss::gpu::GpuIndexIVFPQ index(&res, d, nlist, m, nbits, faiss::METRIC_L2);
    index.nprobe = nprobe;
    index.train(n, xb.data());
    index.add(n, xb.data());
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Build complete!\n";
    std::cout << "Build time: "
              << std::chrono::duration<double>(end - start).count()
              << " s\n";

    // Copying to CPU and writing index to disk (GPU indexes are not
    // directly serializable, faiss::write_index only accepts CPU indexes)
    std::cout << "Writing index to disk...\n";
    start = std::chrono::high_resolution_clock::now();
    faiss::Index* cpu_index = faiss::gpu::index_gpu_to_cpu(&index);
    faiss::write_index(cpu_index, "./gpu-ivf-pq.index");
    delete cpu_index;
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Write time: "
              << std::chrono::duration<double>(end - start).count()
              << " s\n";

    return 0;
}
