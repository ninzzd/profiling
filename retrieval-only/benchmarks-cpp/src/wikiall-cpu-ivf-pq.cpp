#include <fstream>
#include <vector>
#include <cstdint>
#include <random>
#include <faiss/IndexIVFPQ.h>
#include <faiss/index_io.h>
#include <sstream>
#include <cstring>
#include <chrono>
#include <iostream>
#include <filesystem>

// Syntax: wikiall_cpu_flat <k> <iter>
int readQuery(int n, int d, std::string qryfbin, std::string gtfbin, int &k, std::vector<float> &xq, std::vector<uint32_t> &xgt){
    int nq, dq;
    int ngt;
    std::ifstream qrystrm(qryfbin,std::ios::binary);
    std::ifstream gtstrm(gtfbin,std::ios::binary);
    
    if(!qrystrm){
        std::cerr << "Query file not found" << std::endl;
        exit(-1);
    }
    if(!gtstrm){
        std::cerr << "Groundtruth file not found" << std::endl;
        exit(-1);
    }
    auto start = std::chrono::high_resolution_clock::now();
    qrystrm.read(reinterpret_cast<char*>(&nq), sizeof(int32_t));
    qrystrm.read(reinterpret_cast<char*>(&dq), sizeof(int32_t));
    gtstrm.read(reinterpret_cast<char*>(&ngt), sizeof(uint32_t));
    gtstrm.read(reinterpret_cast<char*>(&k), sizeof(uint32_t));
    assert(nq <= n);
    assert(ngt == nq);
    assert(d == dq);
    xq = std::vector<float>(static_cast<size_t>(nq*d));
    xgt = std::vector<uint32_t>(static_cast<size_t>(nq*k));
    qrystrm.read(reinterpret_cast<char*>(xq.data()),xq.size()*sizeof(float));
    gtstrm.read(reinterpret_cast<char*>(xgt.data()),xgt.size()*sizeof(uint32_t));
    auto end = std::chrono::high_resolution_clock::now();
    return nq;
}
int main(int argc, char** argv) {
    int nq;
    int k;
    int nb;
    if(argc != 3){
        std::cerr << "Incorrect syntax: wikiall_cpu_ivf_pq <nb> <stats-path>\n";
        return -1;
    }

    nb = std::stoi(argv[1]);
    std::string stats_path = argv[2];

    faiss::Index* base = faiss::read_index("./cpu-ivf-pq.index");
    auto* index = dynamic_cast<faiss::IndexIVFPQ*>(base);
    if (!index) {
        std::cerr << "Failed to load baseline index.\n";
        return -1;
    }
    
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
        nq = readQuery(index->ntotal,index->d,qryfbin.str(),gtfbin.str(),k,xq,xgt);

        std::vector<faiss::idx_t> labels(static_cast<size_t>(nq) * k);
        std::vector<float> distances(static_cast<size_t>(nq) * k);

        auto start = std::chrono::high_resolution_clock::now();
        index->search(nq, xq.data(), k, distances.data(), labels.data());
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

    stats << "cpu-ivf-pq,";   // index
    stats << nq << ",";         // nq
    stats << k << ",";          // k
    stats << index->nlist << ",";               // nlist
    stats << index->nprobe << ",";               // nprobe
    stats << index->pq.nbits << ",";               // nbits
    stats << index->pq.M << ",";               // m
    stats << ",";               // M
    stats << ",";               // efConstruction
    stats << ",";               // efSearch
    
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