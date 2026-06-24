#include <fstream>
#include <vector>
#include <cstdint>
#include <random>
#include <faiss/IndexHNSW.h>
#include <cstring>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <cmath>
// Usage: ./wikiall-cpu-ivf-flat <mode> <num_iterations>
int main(int argc, char** argv) {
    int n, d;
    int nq, dq;
    int iter;
    int k;

    // Tunable HNSW specific params (will enable sweep later)
    int efConstruction = 200;
    int efSearch = 64;
    int M = 16;       // number of neighbours per node at layer 0 (bottom layer)

    std::cout << "HNSW Properties:" << std::endl;
    std::cout << "efConstruction: " << efConstruction << std::endl;
    std::cout << "efSearch: " << efSearch << std::endl;
    std::cout << "M: " << M << std::endl;

    std::ifstream in("./../datasets/wikiall/base.1M.fbin", std::ios::binary); 
    std::cout << "Reading dataset from disk..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    in.read(reinterpret_cast<char*>(&n), sizeof(int32_t));
    in.read(reinterpret_cast<char*>(&d), sizeof(int32_t));
    std::vector<float> xb(static_cast<size_t>(n) * d);
    in.read(reinterpret_cast<char*>(xb.data()),
            xb.size() * sizeof(float));
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Read complete!" << std::endl;
    std::cout << "Index size: " << n << " \n";
    std::cout << "Index dimensions:" << d << std::endl;
    std::cout << "Read time: "
              << std::chrono::duration<double>(end - start).count()
              << " s\n";

    // Index creation
    faiss::IndexHNSWFlat index(d,M); // IVF index with flat quantization
    index.hnsw.efConstruction = efConstruction;
    index.hnsw.efSearch = efSearch;
    std::cout << "Adding vectors to the HNSW index..." << std::endl;
    start = std::chrono::high_resolution_clock::now();
    index.add(n, xb.data());
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Index created!" << std::endl;
    std::cout << "Index creation latency: "
              << std::chrono::duration<double>(end - start).count()
              << " s\n";
    
    // Query read
    std::ifstream qrystrm("./query.fbin",std::ios::binary);
    start = std::chrono::high_resolution_clock::now();
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
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Query read complete!" << std::endl;
    std::cout << "Query read time: "
              << std::chrono::duration<double>(end - start).count()
              << " s\n";
    std::cout << "Query batch size: " << nq << std::endl;

    // Baseline results read
    std::cout << "Reading baseline results...\n";
    start = std::chrono::high_resolution_clock::now();
    std::ifstream baseres("./baseline-res.fbin",std::ios::binary);
    baseres.read(reinterpret_cast<char*>(&k), sizeof(int));
    baseres.read(reinterpret_cast<char*>(&iter), sizeof(int));
    std::vector<faiss::idx_t> base_labels(nq*k);
    baseres.read(reinterpret_cast<char*>(base_labels.data()), sizeof(faiss::idx_t)*nq*k);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Baseline results read complete!\n";
    std::cout << "Read time: " << std::chrono::duration<double>(end - start).count() << " s\n";

    std::vector<double> latvec;
    std::vector<double> thrvec;
    std::vector<faiss::idx_t> labels(static_cast<size_t>(nq) * k);
    std::vector<float> distances(static_cast<size_t>(nq) * k);

    for(int i = 1;i <= iter;i++){
        auto t8 = std::chrono::high_resolution_clock::now();
        index.search(nq, xq.data(), k, distances.data(), labels.data());
        auto t9 = std::chrono::high_resolution_clock::now();
        double lat = std::chrono::duration<double>(t9 - t8).count();
        double thr = nq/lat;
        latvec.push_back(lat);
        thrvec.push_back(thr);
    }
    

    // Latency and throughput calculation
    std::sort(latvec.begin(),latvec.end());
    double p50lat = latvec[(int)floor(iter/2.0)];
    double p90lat = latvec[(int)floor(9.0*iter/10.0)];
    double p95lat = latvec[(int)floor(9.5*iter/10.0)];
    double p99lat = latvec[(int)floor(9.9*iter/10.0)];
    double avglat = std::accumulate(latvec.begin(),latvec.end(),0.0)/iter;
    double peakthr = *std::max_element(thrvec.begin(),thrvec.end());
    double avgthr = std::accumulate(thrvec.begin(),thrvec.end(),0.0)/iter;

    // Recall calculation
    std::vector<double> recall(nq);
    for(int i = 0;i < nq;i++){
        std::vector<faiss::idx_t> baseqres(k);
        std::vector<faiss::idx_t> qres(k);
        memcpy(baseqres.data(),base_labels.data()+i*k,sizeof(faiss::idx_t)*k);
        memcpy(qres.data(),labels.data()+i*k,sizeof(faiss::idx_t)*k);
        recall[i] = 0.0f;
        for(int j = 0;j < k;j++){
            auto it = std::find(baseqres.begin(),baseqres.end(),qres[j]);
            if(it != baseqres.end()){
                recall[i] += 1.0f;
            }
        }
        recall[i] /= (double)k;
    }
    double peakrcl = *std::max_element(recall.begin(),recall.end());
    double avgrcl = std::accumulate(recall.begin(),recall.end(),0.0)/(double)nq;

    
    std::cout << "P50 Latency: " << p50lat << "s" << std::endl;
    std::cout << "P90 Latency: " << p90lat << "s" << std::endl;
    std::cout << "P95 Latency: " << p95lat << "s" << std::endl;
    std::cout << "P99 Latency: " << p99lat << "s" << std::endl;
    std::cout << "Mean Latency: " << avglat << "s" << std::endl;
    std::cout << "Peak Throughput: " << peakthr << "qps" << std::endl;
    std::cout << "Mean Throughput: " << avgthr << "qps" << std::endl;
    std::cout << "Peak recall: " << peakrcl*100.0 << "%" << std::endl;
    std::cout << "Mean recall: " << avgrcl*100.0 << "%" << std::endl;

    return 0;
}
