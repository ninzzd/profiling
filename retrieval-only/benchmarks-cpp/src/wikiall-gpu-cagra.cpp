// NOTE: requires FAISS built with FAISS_ENABLE_CUVS=ON (the "cuvs" build
// variant). GpuIndexCagra is only compiled in when cuVS is linked.
#include <fstream>
#include <vector>
#include <cstdint>
#include <sstream>
#include <cstring>
#include <chrono>
#include <iostream>
#include <filesystem>
#include <numeric>

#include <faiss/IndexHNSW.h>
#include <faiss/gpu/GpuIndexCagra.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/index_io.h>

#include <utils.hpp>

int main(int argc, char** argv) {
    int nq;
    int k;
    int nb;
    if(argc != 3){
        std::cerr << "Incorrect syntax: wikiall_gpu_cagra <nb> <stats-path>\n";
        return -1;
    }

    nb = std::stoi(argv[1]);
    std::string stats_path = argv[2];

    faiss::Index* base = faiss::read_index("./gpu-cagra.index");
    auto* cpu_index = dynamic_cast<faiss::IndexHNSWCagra*>(base);
    if (!cpu_index) {
        std::cerr << "Failed to load baseline index.\n";
        return -1;
    }

    faiss::gpu::StandardGpuResources res;
    faiss::gpu::GpuIndexCagraConfig config;
    faiss::gpu::GpuIndexCagra index(&res, cpu_index->d, faiss::METRIC_L2, config);
    index.copyFrom(cpu_index);
    delete base;

    faiss::gpu::SearchParametersCagra search_params;
    // Analogous to HNSW's efSearch: bounds the intermediate candidate list
    // during graph traversal, trading recall for latency.
    search_params.itopk_size = 64;

    std::vector<double> latvec;
    std::vector<double> thpvec;
    std::vector<double> rclvec;

    for(int i = 1;i <= nb;i++){
        std::stringstream qryfbin;
        std::stringstream gtfbin;
        std::vector<float> xq;
        std::vector<uint32_t> xgt;
        qryfbin << "./queries/query" << i << ".fbin";
        gtfbin << "./gt/gt" << i << ".fbin";
        nq = readQuery(index.ntotal,index.d,qryfbin.str(),gtfbin.str(),k,xq,xgt);

        std::vector<faiss::idx_t> labels(static_cast<size_t>(nq) * k);
        std::vector<float> distances(static_cast<size_t>(nq) * k);

        auto start = std::chrono::high_resolution_clock::now();
        index.search(nq, xq.data(), k, distances.data(), labels.data(), &search_params);
        auto end = std::chrono::high_resolution_clock::now();
        double lat = std::chrono::duration<double>(end - start).count();
        double thp = nq/lat;
        double rcl = 0.0;
        for (int q = 0; q < nq; q++) {
            double qrcl = 0.0;
            for (int j = 0; j < k; j++) {
                auto begin = xgt.begin() + q * k;
                auto end   = begin + k;
                if (std::find(begin, end,
                            static_cast<uint32_t>(labels[q * k + j])) != end)
                    qrcl++;
            }
            rcl += qrcl / k;
        }
        rcl /= nq;
        latvec.push_back(lat);
        thpvec.push_back(thp);
        rclvec.push_back(rcl);
    }
    std::sort(latvec.begin(),latvec.end());
    // latency
    double minlat = *std::min_element(latvec.begin(),latvec.end());
    double p50lat = latvec[(int)floor(nb/2.0)];
    double p90lat = latvec[(int)floor(9.0*nb/10.0)];
    double avglat = std::accumulate(latvec.begin(),latvec.end(),0.0)/nb;
    double maxlat = *std::max_element(latvec.begin(),latvec.end());
    double stdlat = 0.0;
    for (int i = 0; i < nb; i++) {
        stdlat += (latvec[i] - avglat) * (latvec[i] - avglat);
    }
    stdlat = std::sqrt(stdlat/nb);

    // throughput
    double avgthr = std::accumulate(thpvec.begin(),thpvec.end(),0.0)/nb;

    // recall
    double minrcl = *std::min_element(rclvec.begin(),rclvec.end());
    double avgrcl = std::accumulate(rclvec.begin(),rclvec.end(),0.0)/nb;
    double maxrcl = *std::max_element(rclvec.begin(),rclvec.end());
    double stdrcl = 0.0;
    for (int i = 0; i < nb; i++) {
        stdrcl += (rclvec[i] - avgrcl) * (rclvec[i] - avgrcl);
    }
    stdrcl = std::sqrt(stdrcl/nb);

    std::cout << "Min Latency: " << minlat << "s" << std::endl;
    std::cout << "P50 Latency: " << p50lat << "s" << std::endl;
    std::cout << "P90 Latency: " << p90lat << "s" << std::endl;
    std::cout << "Mean Latency: " << avglat << "s" << std::endl;
    std::cout << "Max Latency: " << maxlat << "s" << std::endl;
    std::cout << "Latency Standard Deviation: " << stdlat << "s" << std::endl;

    std::cout << "Mean Throughput: " << avgthr << "qps" << std::endl;

    std::cout << "Min Recall: " << maxrcl*100.0 << "%" << std::endl;
    std::cout << "Mean Recall: " << avgrcl*100.0 << "%" << std::endl;
    std::cout << "Max Recall: " << maxrcl*100.0 << "%" << std::endl;
    std::cout << "Recall Standard Deviation: " << stdrcl*100 << "%" << std::endl;

    bool new_file = !std::filesystem::exists(stats_path);
    std::ofstream stats(stats_path,std::ios::app);
    if (new_file)
        stats << "index,nq,k,nlist,nprobe,nbits,m,M,efConstruction,efSearch,"
                "minlat,p50lat,p90lat,avglat,maxlat,stdlat,"
                "avgQPS,"
                "minrecall,avgrecall,maxrecall,stdrcl\n";

    stats << "gpu-cagra,";      // index
    stats << nq << ",";         // nq
    stats << k << ",";          // k
    stats << ",";               // nlist
    stats << ",";               // nprobe
    stats << ",";               // nbits
    stats << ",";               // m
    stats << ",";               // M
    stats << ",";               // efConstruction
    stats << search_params.itopk_size << ","; // efSearch (itopk_size)

    // latency
    stats << minlat << ",";     // minimum latency
    stats << p50lat << ",";     // 50th percentile latency
    stats << p90lat << ",";     // 90th percentile latency
    stats << avglat << ",";     // average latency
    stats << maxlat << ",";     // maximum latency
    stats << stdlat << ",";     // standard deviation of latency

    // throughput
    stats << avgthr << ",";     // average throughput

    // recall
    stats << minrcl << ",";     // minimum recall
    stats << avgrcl << ",";     // average recall
    stats << maxrcl << ",";     // maximum recall
    stats << stdrcl << "\n";    // standard deviation of recall
    return 0;
}
