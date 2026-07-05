#include <utils.hpp>

#include <cassert>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int readQuery(
    int n,
    int d,
    const std::string& qryfbin,
    const std::string& gtfbin,
    int& k,
    std::vector<float>& xq,
    std::vector<uint32_t>& xgt) {
    int nq, dq;
    int ngt;
    std::ifstream qrystrm(qryfbin, std::ios::binary);
    std::ifstream gtstrm(gtfbin, std::ios::binary);

    if (!qrystrm) {
        std::cerr << "Query file not found" << std::endl;
        exit(-1);
    }
    if (!gtstrm) {
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
    xq = std::vector<float>(static_cast<size_t>(nq * d));
    xgt = std::vector<uint32_t>(static_cast<size_t>(nq * k));
    qrystrm.read(
        reinterpret_cast<char*>(xq.data()),
        xq.size() * sizeof(float));
    gtstrm.read(
        reinterpret_cast<char*>(xgt.data()),
        xgt.size() * sizeof(uint32_t));
    auto end = std::chrono::high_resolution_clock::now();
    return nq;
}
