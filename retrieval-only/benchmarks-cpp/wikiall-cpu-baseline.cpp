#include <fstream>
#include <vector>
#include <cstdint>
#include <random>
#include <faiss/IndexFlat.h>
#include <cstring>
#include <chrono>
#include <iostream>
#include <filesystem>

// Syntax: wikiall_cpu_flat <k> <iter>
int main(int argc, char** argv) {
    int n, d;
    int nq, dq;
    int k;
    int iter;
    if(argc < 3){
        std::cerr << "Incorrect syntax: wikiall_cpu_flat <k> <iter>\n";
        return -1;
    }
    else{
        k = std::stoi(argv[1]);
        iter = std::stoi(argv[2]);
    }

    std::ifstream in("./../datasets/wikiall/base.1M.fbin", std::ios::binary);

    std::cout << "Reading dataset from disk..." << std::endl;
    auto t0 = std::chrono::high_resolution_clock::now();
    in.read(reinterpret_cast<char*>(&n), sizeof(int32_t));
    in.read(reinterpret_cast<char*>(&d), sizeof(int32_t));
    std::vector<float> xb(static_cast<size_t>(n) * d);
    in.read(reinterpret_cast<char*>(xb.data()), xb.size() * sizeof(float));
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Index read complete!" << std::endl;
    std::cout << "Index read time: "
              << std::chrono::duration<double>(t1 - t0).count()
              << " s\n";

    std::ifstream qrystrm("./query.fbin",std::ios::binary);
    auto t2 = std::chrono::high_resolution_clock::now();
    if(!qrystrm){
        std::cerr << "Query file not found" << std::endl;
        return -1;
    }
    qrystrm.read(reinterpret_cast<char*>(&nq), sizeof(int32_t));
    qrystrm.read(reinterpret_cast<char*>(&dq), sizeof(int32_t));
    if(dq != d){
        std::cerr << "Incompatable query: dimension mismatch" << std::endl;
        return -1;
    }
    std::vector<float> xq(static_cast<size_t>(nq*d));
    qrystrm.read(reinterpret_cast<char*>(xq.data()),xq.size()*sizeof(float));
    auto t3 = std::chrono::high_resolution_clock::now();
    std::cout << "Query read complete!" << std::endl;
    std::cout << "Query read time: "
              << std::chrono::duration<double>(t3 - t2).count()
              << " s\n";
    std::cout << "Query batch size: " << nq << std::endl;
    std::cout << "Search size (k): " << k << std::endl;
    std::cout << "Search iteration count: " << iter << std::endl;

    std::vector<double> latvec;
    std::vector<double> thpvec;
    faiss::IndexFlatIP index(d); // cosine similarity - most commonly used metric
    
    t2 = std::chrono::high_resolution_clock::now();
    index.add(n, xb.data());
    t3 = std::chrono::high_resolution_clock::now();
    std::cout << "Index creation complete!" << std::endl;
    std::cout << "Index size: " << index.ntotal << " vectors\n";
    std::cout << "Index dimensions:" << d << std::endl;
    std::cout << "Index creation latency: "
              << std::chrono::duration<double>(t3 - t2).count()
              << " s\n";

    std::vector<faiss::idx_t> labels(static_cast<size_t>(nq) * k);
    std::vector<float> distances(static_cast<size_t>(nq) * k);

    for(int i = 1;i <= iter;i++){
        auto t4 = std::chrono::high_resolution_clock::now();
        index.search(nq, xq.data(), k, distances.data(), labels.data());
        auto t5 = std::chrono::high_resolution_clock::now();
        double lat = std::chrono::duration<double>(t5 - t4).count();
        double thp = nq/lat;
        latvec.push_back(lat);
        thpvec.push_back(thp);
    }
    std::sort(latvec.begin(),latvec.end());
    double p50lat = latvec[(int)floor(iter/2.0)];
    double p90lat = latvec[(int)floor(9.0*iter/10.0)];
    double p95lat = latvec[(int)floor(9.5*iter/10.0)];
    double p99lat = latvec[(int)floor(9.9*iter/10.0)];
    double avglat = std::accumulate(latvec.begin(),latvec.end(),0.0)/iter;
    double peakthr = *std::max_element(thpvec.begin(),thpvec.end());
    double avgthr = std::accumulate(thpvec.begin(),thpvec.end(),0.0)/iter;

    // std::cout << std::fixed << std::setprecision(9);
    std::cout << "P50 Latency: " << p50lat << "s" << std::endl;
    std::cout << "P90 Latency: " << p90lat << "s" << std::endl;
    std::cout << "P95 Latency: " << p95lat << "s" << std::endl;
    std::cout << "P99 Latency: " << p99lat << "s" << std::endl;
    std::cout << "Mean Latency: " << avglat << "s" << std::endl;
    std::cout << "Peak Throughput: " << peakthr << "qps" << std::endl;
    std::cout << "Mean Throughput: " << avgthr << "qps" << std::endl;

    std::ofstream baseres("./baseline-res.fbin",std::ios::binary);
    bool new_file = !std::filesystem::exists("./stats.csv");
    std::ofstream stats("./stats.csv",std::ios::app);
    if(new_file)
        stats << "index,nq,k,nlist,nprobe,nbits,m,M,efConstruction,efSearch,p50lat,p90lat,p95lat,p99lat,avglat,peakQPS,avgQPS,peakrecall,avgrecall\n";

    stats << "baseline,";       // index   
    stats << nq << ",";         // nq
    stats << k << ",";          // k
    stats << ",";               // nlist
    stats << ",";               // nprobe
    stats << ",";               // nbits
    stats << ",";               // m
    stats << ",";               // M
    stats << ",";               // efConstruction
    stats << ",";               // efSearch
    stats << p50lat << ",";     // p50lat  
    stats << p90lat << ",";     // p90lat
    stats << p95lat << ",";     // p95lat
    stats << p99lat << ",";     // p99lat
    stats << avglat << ",";     // avglat
    stats << peakthr << ",";    // peakqps
    stats << avgthr << ",";     // avgqps
    stats << 1.0 << ",";        // peakrecall
    stats << 1.0 << "\n";        // avgrecall

    /*
    Results format:
    <iter>
    <k>
    <labels>
    <stats>
    */
    std::cout << "Writing results to disk..." << std::endl;
    auto t4 = std::chrono::high_resolution_clock::now();
    baseres.write(reinterpret_cast<char*>(&k), sizeof(int));
    baseres.write(reinterpret_cast<char*>(&iter), sizeof(int));
    baseres.write(reinterpret_cast<char*>(labels.data()), sizeof(faiss::idx_t)*labels.size());
    
    auto t5 = std::chrono::high_resolution_clock::now();
    std::cout << "Results writing complete!\n";
    std::cout << "Write time: " << std::chrono::duration<double>(t5 - t4).count() << " s\n";

    return 0;
}