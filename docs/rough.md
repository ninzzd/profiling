# Retrieval Benchmarking

## FAISS Build Configurations

All builds assume you're in the `retrieval-only/faiss/` directory and output to `retrieval-only/faiss-builds/<variant>/`.

### Generic CPU Build
No SIMD optimizations. The baseline for comparison.

```bash
cmake -B ../faiss-builds/generic -S . \
    -DFAISS_OPT_LEVEL=generic \
    -DFAISS_ENABLE_GPU=OFF \
    -DFAISS_ENABLE_PYTHON=OFF \
    -DFAISS_ENABLE_EXTRAS=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -DFAISS_ENABLE_MKL=OFF
make -C ../faiss-builds/generic -j$(nproc) faiss
```

### AVX2 CPU Build
Compiled with `-mavx2 -mfma -mf16c -mpopcnt`. The `faiss_avx2` target is the primary library; the generic `faiss` target is excluded from `all` but still exists.

```bash
cmake -B ../faiss-builds/avx2 -S . \
    -DFAISS_OPT_LEVEL=avx2 \
    -DFAISS_ENABLE_GPU=OFF \
    -DFAISS_ENABLE_PYTHON=OFF \
    -DFAISS_ENABLE_EXTRAS=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -DFAISS_ENABLE_MKL=OFF
make -C ../faiss-builds/avx2 -j$(nproc) faiss_avx2
```

### AVX512 CPU Build
Compiled with AVX2 + AVX-512F/CD/VL/DQ/BW. The `faiss_avx512` target is the primary library; `faiss_avx2` is also built as a fallback.

```bash
cmake -B ../faiss-builds/avx512 -S . \
    -DFAISS_OPT_LEVEL=avx512 \
    -DFAISS_ENABLE_GPU=OFF \
    -DFAISS_ENABLE_PYTHON=OFF \
    -DFAISS_ENABLE_EXTRAS=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -DFAISS_ENABLE_MKL=OFF
make -C ../faiss-builds/avx512 -j$(nproc) faiss_avx512
```

### Dynamic Dispatch (DD) CPU Build
Single library that detects CPU features at runtime and dispatches to the best SIMD code path. No need for separate builds — useful for deployment, less useful for benchmarking individual SIMD levels.

```bash
cmake -B ../faiss-builds/dd -S . \
    -DFAISS_OPT_LEVEL=dd \
    -DFAISS_ENABLE_GPU=OFF \
    -DFAISS_ENABLE_PYTHON=OFF \
    -DFAISS_ENABLE_EXTRAS=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -DFAISS_ENABLE_MKL=OFF
make -C ../faiss-builds/dd -j$(nproc) faiss
```

### CUDA GPU Build
Enables GPU index support via CUDA. Requires a CUDA toolkit installation.

```bash
cmake -B ../faiss-builds/cuda -S . \
    -DFAISS_ENABLE_GPU=ON \
    -DFAISS_ENABLE_PYTHON=OFF \
    -DFAISS_ENABLE_EXTRAS=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -DFAISS_ENABLE_MKL=OFF \
    -DCMAKE_CUDA_ARCHITECTURES=120
make -C ../faiss-builds/cuda -j$(nproc) faiss
```

### CUDA GPU Build with cuVS
Enables GPU indexes with cuVS (NVIDIA's GPU-accelerated vector search library). Requires CUDA toolkit and internet access (fetches RAPIDS cmake infrastructure).

```bash
cmake -B ../faiss-builds/cuvs -S . \
    -DFAISS_ENABLE_GPU=ON \
    -DFAISS_ENABLE_CUVS=ON \
    -DFAISS_ENABLE_PYTHON=OFF \
    -DFAISS_ENABLE_EXTRAS=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -DFAISS_ENABLE_MKL=OFF
make -C ../faiss-builds/cuvs -j$(nproc) faiss
```

## Linking Benchmarks Against a Specific Build

From `retrieval-only/benchmarks-cpp/`, configure with the desired variant:

```bash
cmake -B build-<variant> -S . \
    -DFAISS_BUILD_VARIANT=<variant>
make -C build-<variant> -j$(nproc)
```

Where `<variant>` is one of: `generic`, `avx2`, `avx512`, `dd`, `cuda`, `cuvs`

