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
Dependency: OpenBLAS.

| Variant | Notes |
|---|---|
| `generic` | No SIMD. Baseline. |
| `avx2` | `-mavx2 -mfma -mf16c -mpopcnt`. Builds `faiss_avx2`. |
| `avx512` | AVX2 + AVX-512F/CD/VL/DQ/BW. Builds `faiss_avx512` (+ `faiss_avx2` fallback). |
| `dd` | Dynamic dispatch — one library, picks best SIMD level at runtime via `cpuid`. Bundles all tiers, so it's the largest build. |
| `cuda` | GPU indexes, requires CUDA toolkit. |
| `cuvs` | GPU indexes via NVIDIA cuVS, requires CUDA toolkit + internet (RAPIDS cmake fetch). |

```bash
# generic
cmake -B ../faiss-builds/generic -S . -DFAISS_OPT_LEVEL=generic \
    -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DFAISS_ENABLE_EXTRAS=OFF \
    -DBUILD_SHARED_LIBS=OFF -DFAISS_ENABLE_MKL=OFF
make -C ../faiss-builds/generic -j$(nproc) faiss

# avx2 (swap target/opt-level for avx512, dd)
cmake -B ../faiss-builds/avx2 -S . -DFAISS_OPT_LEVEL=avx2 \
    -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DFAISS_ENABLE_EXTRAS=OFF \
    -DBUILD_SHARED_LIBS=OFF -DFAISS_ENABLE_MKL=OFF
make -C ../faiss-builds/avx2 -j$(nproc) faiss_avx2

# cuda
cmake -B ../faiss-builds/cuda -S . -DFAISS_ENABLE_GPU=ON \
    -DFAISS_ENABLE_PYTHON=OFF -DFAISS_ENABLE_EXTRAS=OFF -DBUILD_SHARED_LIBS=OFF \
    -DFAISS_ENABLE_MKL=OFF -DCMAKE_CUDA_ARCHITECTURES=120
make -C ../faiss-builds/cuda -j$(nproc) faiss

# cuvs: same as cuda plus -DFAISS_ENABLE_CUVS=ON
```

## Benchmarks (`retrieval-only/benchmarks-cpp/`)

Links against one FAISS variant via `-DFAISS_BUILD_VARIANT=<generic|avx2|avx512|dd|cuda|cuvs>`:

```bash
cmake -B builds/build-<variant> -S . -DFAISS_BUILD_VARIANT=<variant>
make -C builds/build-<variant> -j$(nproc)
# or build all variants at once:
./scripts/build-all.sh
```

Executables read `../datasets/wikiall/*` and write index/query/gt files relative to cwd —
**always run from `retrieval-only/benchmarks-cpp/`**.

## Sweep Scripts (`scripts/`)

- `run-all.sh <none>` — runs all four sweeps below for every build variant.
- `bs-sweep.sh <variant>` — batch-size sweep (baseline/HNSW/IVF-Flat/IVF-PQ together).
- `hnsw-param-sweep.sh <variant>` — sweeps M, efConstruction, efSearch.
- `ivf-flat-param-sweep.sh <variant>` — sweeps nlist, nprobe.
- `ivf-pq-param-sweep.sh <variant>` — sweeps nlist, nprobe, m, nbits.

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
