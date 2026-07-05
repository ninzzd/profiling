#pragma once
#include <fstream>
#include <string>
#include <vector>
#include <cstdint>

void writeStats(
    std::vector<double> latvec,
    std::vector<double> thrvec,
    std::vector<double> rclvec,
    const std::string& filename);

int readQuery(
    int n,
    int d,
    const std::string& qryfbin,
    const std::string& gtfbin,
    int& k,
    std::vector<float>& xq,
    std::vector<uint32_t>& xgt);
