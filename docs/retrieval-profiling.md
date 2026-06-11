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
cmake -B builds/cpu-debug-cpp \
  -DFAISS_ENABLE_GPU=OFF \
  -DFAISS_ENABLE_PYTHON=OFF \
  -DBUILD_TESTING=OFF \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

Build the given configuration:
```bash
# keep it safe, don't use too many cores
cmake --build builds/cpu-debug-cpp -j8 
```
