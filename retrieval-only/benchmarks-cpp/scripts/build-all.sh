#!/bin/bash

cmake_dir=~/Git/profiling/retrieval-only/benchmarks-cpp
# CMake config commands
cmake -B builds/build-generic -S $cmake_dir -DFAISS_BUILD_VARIANT=generic
cmake -B builds/build-avx2 -S $cmake_dir -DFAISS_BUILD_VARIANT=avx2
cmake -B builds/build-avx512 -S $cmake_dir -DFAISS_BUILD_VARIANT=avx512
cmake -B builds/build-dd -S $cmake_dir -DFAISS_BUILD_VARIANT=dd
cmake -B builds/build-cuda -S $cmake_dir -DFAISS_BUILD_VARIANT=cuda
cmake -B builds/build-cuvs -S $cmake_dir -DFAISS_BUILD_VARIANT=cuvs

# Build commands
make -C builds/build-generic -j$(nproc)
make -C builds/build-dd -j$(nproc)
make -C builds/build-avx2 -j$(nproc)
make -C builds/build-avx512 -j$(nproc)
make -C builds/build-cuda -j$(nproc)
make -C builds/build-cuvs -j$(nproc)