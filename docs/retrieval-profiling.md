# Retrieval Profiling and Benchmarking with FAISS

FAISS is built from source (submodule at `retrieval-only/faiss`) to support SIMD variant
comparisons, GPU builds, and profiling with debug info.

## Dataset

1M-vector subset of the Wiki-All 88M dataset, dim 768.

```bash
curl -O https://data.rapids.ai/raft/datasets/wiki_all_1M/wiki_all_1M.tar
tar -xvf wiki_all_1M.tar -C retrieval-only/datasets/wikiall
```

## FAISS Build Variants

All commands run from `retrieval-only/faiss/`, output to `retrieval-only/faiss-builds/<variant>/`.
Dependency: OpenBLAS. The `cuda`/`cuvs` variants additionally need the CUDA toolkit
(`/usr/local/cuda`, currently 13.3); `cuvs` needs the cuVS C++ library on top of that —
see Prerequisites below.

| Variant | Notes |
|---|---|
| `generic` | No SIMD. Baseline. |
| `avx2` | `-mavx2 -mfma -mf16c -mpopcnt`. Builds `faiss_avx2`. |
| `avx512` | AVX2 + AVX-512F/CD/VL/DQ/BW. Builds `faiss_avx512` (+ `faiss_avx2` fallback). |
| `dd` | Dynamic dispatch — one library, picks best SIMD level at runtime via `cpuid`. Bundles all tiers, so it's the largest build. |
| `cuda` | GPU indexes, requires CUDA toolkit. |
| `cuvs` | GPU indexes via NVIDIA cuVS. Requires CUDA toolkit + a C++ cuVS install; rapids-cmake is fetched at configure time but cuVS itself is **not** — it must already be present. Adds CAGRA. |

```bash
# generic
cmake -B ../faiss-builds/generic -S . -DFAISS_OPT_LEVEL=generic \
    -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DFAISS_ENABLE_EXTRAS=OFF \
    -DBUILD_SHARED_LIBS=OFF -DFAISS_ENABLE_MKL=OFF
make -C ../faiss-builds/generic -j$(nproc) faiss

# avx2
cmake -B ../faiss-builds/avx2 -S . -DFAISS_OPT_LEVEL=avx2 \
    -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DFAISS_ENABLE_EXTRAS=OFF \
    -DBUILD_SHARED_LIBS=OFF -DFAISS_ENABLE_MKL=OFF
make -C ../faiss-builds/avx2 -j$(nproc) faiss_avx2

# cuda
cmake -B ../faiss-builds/cuda -S . -DFAISS_ENABLE_GPU=ON \
    -DFAISS_ENABLE_PYTHON=OFF -DFAISS_ENABLE_EXTRAS=OFF -DBUILD_SHARED_LIBS=OFF \
    -DFAISS_ENABLE_MKL=OFF -DCMAKE_CUDA_ARCHITECTURES=120
make -C ../faiss-builds/cuda -j$(nproc) faiss

# avx512
cmake -B ../faiss-builds/avx512 -S . -DFAISS_OPT_LEVEL=avx512 \
    -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DFAISS_ENABLE_EXTRAS=OFF \
    -DBUILD_SHARED_LIBS=OFF -DFAISS_ENABLE_MKL=OFF
make -C ../faiss-builds/avx512 -j$(nproc) faiss_avx512

# dd (dynamic dispatch -- single library, runtime SIMD selection)
cmake -B ../faiss-builds/dd -S . -DFAISS_OPT_LEVEL=dd \
    -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DFAISS_ENABLE_EXTRAS=OFF \
    -DBUILD_SHARED_LIBS=OFF -DFAISS_ENABLE_MKL=OFF
make -C ../faiss-builds/dd -j$(nproc) faiss

# cuvs -- cuda flags plus FAISS_ENABLE_CUVS; cuvs_ROOT points at the C++ cuVS install
cmake -B ../faiss-builds/cuvs -S . -DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_CUVS=ON \
    -DFAISS_ENABLE_PYTHON=OFF -DFAISS_ENABLE_EXTRAS=OFF -DBUILD_SHARED_LIBS=OFF \
    -DFAISS_ENABLE_MKL=OFF -DCMAKE_CUDA_ARCHITECTURES=120 \
    -Dcuvs_ROOT=$CONDA_PREFIX
make -C ../faiss-builds/cuvs -j$(nproc) faiss
```

`CMAKE_CUDA_ARCHITECTURES=120` targets the local RTX 5070 (Blackwell, sm_120). Widening the
list multiplies GPU compile time by roughly one pass per architecture.

### Prerequisites for the `cuvs` variant

FAISS links `cuvs::cuvs` and includes RAFT headers, i.e. the **C++** cuVS library. Note that
`CMakeLists.txt:124` calls a bare `find_package(cuvs)` with no `REQUIRED`, so a missing cuVS
does **not** fail at configure time — it fails much later during compilation with
`fatal error: raft/core/device_resources.hpp: No such file or directory`. Check
`cuvs_DIR` in `faiss-builds/cuvs/CMakeCache.txt`: `cuvs_DIR-NOTFOUND` means cuVS was not
picked up, whatever configure printed.

The apt packages (`libcuvs1-cuda-13`, `libcuvs1-dev-cuda-13`) are **not** sufficient — they
ship only the C API (`libcuvs_c.so`, target `cuvs::c_api`), with no `libcuvs.so`, no RAFT
headers, and no `cuvs::cuvs` target. Install the C++ build from the RAPIDS conda channel
instead, matching the RAPIDS version FAISS pins in `cmake/thirdparty/fetch_rapids.cmake`
(currently 26.06) and the local CUDA major version:

```bash
conda create -n cuvs -c rapidsai -c conda-forge -c nvidia libcuvs=26.06 cuda-version=13.0
conda activate cuvs   # sets CONDA_PREFIX, consumed by -Dcuvs_ROOT above
```

**Use a CMake >= 3.30.4 for anything cuvs.** Both rapids-cmake and cuVS's own
`cuvs-config.cmake` require it, and the system `/usr/bin/cmake` is 3.28.3 — it fails the
configure outright. The project venv already ships 3.30.9, so prefix the cuvs `cmake`
invocations (both the FAISS one above and the benchmark one below) with it:

```bash
alias cmake=~/Git/profiling/.venv/bin/cmake   # or call the full path
```

Verified working: `libcuvs 26.06.00 cuda13` in `~/anaconda3/envs/cuvs`, FAISS `libfaiss.a`
(190 MB) and all 18 benchmark targets including CAGRA.

## Benchmarks (`retrieval-only/benchmarks-cpp/`)

Links against one FAISS variant via `-DFAISS_BUILD_VARIANT=<generic|avx2|avx512|dd|cuda|cuvs>`:

```bash
cmake -B builds/build-<variant> -S . -DFAISS_BUILD_VARIANT=<variant>
make -C builds/build-<variant> -j$(nproc)

# cuvs: libfaiss.a leaves cuvs::/rmm:: symbols unresolved, so the executables must
# link the cuVS C++ library too -- and this needs the >=3.30.4 cmake as well
~/Git/profiling/.venv/bin/cmake -B builds/build-cuvs -S . \
    -DFAISS_BUILD_VARIANT=cuvs -Dcuvs_ROOT=$CONDA_PREFIX
make -C builds/build-cuvs -j$(nproc)
# or build all variants at once:
./scripts/build-all.sh
```

The `*_gpu_*` executables are only emitted for the `cuda` and `cuvs` variants, and the CAGRA
pair (`idxgen_gpu_cagra`, `wikiall_gpu_cagra`) only for `cuvs` — a `cuvs` benchmark build
therefore requires `faiss-builds/cuvs/faiss/libfaiss.a` to exist first.

Executables read `../datasets/wikiall/*` and write index/query/gt files relative to cwd —
**always run from `retrieval-only/benchmarks-cpp/`**.

The `cuvs` binaries carry the conda prefix in their RUNPATH, so they resolve `libcuvs`,
`librmm`, `libstdc++`, `libgomp` and `libopenblas` from the conda env. No `LD_LIBRARY_PATH`
needed; worth knowing if a `cuvs` binary ever misbehaves at runtime while the same code works
under another variant.

## Sweep Scripts (`scripts/`)

- `run-all.sh <none>` — runs all four sweeps below for every build variant.
- `bs-sweep.sh <variant>` — batch-size sweep (baseline/HNSW/IVF-Flat/IVF-PQ together).
- `hnsw-param-sweep.sh <variant>` — sweeps M, efConstruction, efSearch.
- `ivf-flat-param-sweep.sh <variant>` — sweeps nlist, nprobe.
- `ivf-pq-param-sweep.sh <variant>` — sweeps nlist, nprobe, m, nbits.
- `gpu-ivf-flat-param-sweep.sh <cuda|cuvs>` — sweeps nlist, nprobe.
- `gpu-ivf-pq-param-sweep.sh <cuda|cuvs>` — sweeps nlist, nprobe, m; nbits on `cuvs` only.
- `gpu-cagra-param-sweep.sh cuvs` — sweeps graph_degree, intermediate_graph_degree.

The GPU scripts reject a variant that cannot supply their binaries, and write no PNGs: the
`*-graphing.py` scripts hardwire the CPU result directories. GPU IVF-PQ parameter ranges are
narrower than the CPU ones because `GpuIndexIVFPQ::verifyPQSettings_` rejects much of the CPU
sweep — on the non-cuVS path `nbits` must be exactly 8, and the 48 KiB shared-memory budget
(`4 * m * 2^nbits`) caps `m` at 48.

Each writes a CSV and PNGs to `results/{workload-sweep,param-sweep/hnsw,param-sweep/ivf-flat,param-sweep/ivf-pq}-<variant>/`.
All scripts `cd` to `benchmarks-cpp/` internally, so they can be invoked from any directory.

Before running sweeps, cap BLAS threading to avoid oversubscription (see below):

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=$(nproc)
```

## Known Issues

- **Thread oversubscription**: FAISS parallelizes via OpenMP; each of those threads can
  independently spawn its own OpenBLAS thread pool, causing up to `nproc²` threads to
  contend for `nproc` cores (seen: 57M involuntary context switches, 73% time in kernel
  instead of userspace). Fix: `OPENBLAS_NUM_THREADS=1` so only the outer OpenMP layer
  parallelizes.
- **SIMD build variant doesn't uniformly speed things up**: `IndexFlatL2`/`IndexIVFFlat`
  route their hot path through BLAS (identical across all variants — `FAISS_BUILD_VARIANT`
  doesn't touch it), so flat search shows ~0% difference between builds. HNSW is
  traversal/heap-bound, not FLOP-bound, so it's similarly flat. Only IVF-PQ's scanner is
  routed through FAISS's own SIMD dispatch (`with_simd_level` in `IndexIVFPQ.cpp`), and even
  there gains are modest (~25–30%, not 4–16x) because PQ's asymmetric distance computation
  is table-lookup/gather-based, which vectorizes less cleanly than dense arithmetic.
  `generic`/`avx2`/`avx512` builds fix a `SINGLE_SIMD_LEVEL` at compile time; `dd` detects
  it at runtime via `cpuid` (`faiss/impl/simd_dispatch.h`).
- **Hyperthreading is unrelated to `FAISS_BUILD_VARIANT`**: the variant only picks a SIMD
  instruction set; thread count follows `nproc` (which already includes SMT siblings)
  regardless of variant.

## Baseline Reference Numbers (1M Wiki-All, dim 768)

Early exploratory runs (pre-sweep-script era), for rough context — superseded by the CSV
sweeps in `results/` for anything rigorous.

| Benchmark | Index creation | Mean latency | Mean throughput | Recall |
|---|---|---|---|---|
| Flat (100k random vectors, toy) | 0.06 s | 1.76 ms/query | ~568 qps | 100% (exact) |
| Flat (1M Wiki-All, k=10) | 0.60 s | 29.1 ms/query | 34.3 qps | 100% (exact) |
| IVF-Flat (nlist=100, nprobe=10) | 0.89 s | 6.6 ms/query | 152 qps | — |
| IVF-PQ (nlist=100, nprobe=10) | 7.5 s (+81.6 s training) | 24.8 ms/query | 4065 qps | — |
| HNSW (M=16, efConstruction=200, efSearch=64) | 171 s | 8.8 ms/query | 11992 qps | — |

IVF-PQ training (subspace k-means) is the dominant cost at index build time; HNSW graph
construction is the slowest to build but fastest to query.

Inter-index comparison (nq=10, k=100, common query batch):

| Index | Mean latency | Mean throughput | Mean recall@100 |
|---|---|---|---|
| Baseline (Flat, exact) | 235 ms | 43.1 qps | 100% |
| HNSW | 0.60 ms | 19449 qps | 90.8% |
| IVF-Flat (nlist=2048, nprobe=36) | 13.4 ms | 757 qps | 98.8% |
| IVF-PQ (nlist=2048, nprobe=36, nbits=8, m=48) | 1.36 ms | 8340 qps | 58.7% |
