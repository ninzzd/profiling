# Retrieval Profiling for FAISS

## Build Configurations

To support debugging, detailed profiling, GPU/SIMD acceleration and other options, it is better to build from source. Hence, I have added the [FAISS](https://github.com/facebookresearch/faiss.git) repository as a submodule in the directory `retireval-only/faiss`

### Dependencies

FAISS builds for SIMD and other configurations, require: 
    
    - OpenBlas

### CPU-Only
This configuration includes:

    - C++ library (w/o python)
    - AVX2 SIMD utilization
    - Debugging with info

Here's the cmake command to create the build configuration:
```bash
cmake -B ../faiss-builds/cpu-debug-cpp \
    -DBUILD_SHARED_LIBS=ON \
    -DFAISS_ENABLE_GPU=OFF \
    -DFAISS_ENABLE_PYTHON=OFF \
    -DBUILD_TESTING=OFF \
    -DFAISS_ENABLE_MKL=OFF \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

Build the given configuration:
```bash
# keep it safe, don't use too many cores
cmake --build ../faiss-builds/cpu-debug-cpp -j8 
```
## Benchmarks

### Simple CPU-Only Benchmark

Refer to [`simple-cpu-debug.cpp`](./../retrieval-only/benchmarks-cpp/simple-cpu-debug.cpp) source file. An IVF database of with 100,000 vectors was initialized, with an embedding dimensionality of 768. A standard uniform random distribution $U[0,1]$ was sampled to initialize the components of all the vectors. The same sampling was done for obtaining 1000 query vectors. After creating the database vectors, adding them to the IVF index (flat L2, i.e. row-major order of vector components, L2-normed distance) and creating the query vectors, the queries were searched. The IVF initialization time and search time were also calculated.

**No-Tool Profiling (std::chrono)**

```bash
Index size: 100000 vectors
Add time: 0.0623132 s
Search time: 1.76192 s
Queries/sec: 567.563
Latency/query: 1761.92 us
First neighbor ID: 76510, distance: 106.56
```