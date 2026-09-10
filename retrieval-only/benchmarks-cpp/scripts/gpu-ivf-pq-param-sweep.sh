#!/bin/bash
cd "$(dirname "$0")/.." || exit 1

cleanup() {
    rm -rf ./gpu-ivf-pq.index
    rm -rf ./queries
    rm -rf ./gt
}

interrupt() {
    echo "Interrupt received. Cleaning up..."
    cleanup
    exit 130
}

trap interrupt SIGINT SIGTERM

# GPU executables are only produced for the cuda and cuvs variants
# (see FAISS_GPU_VARIANT in CMakeLists.txt)
case "$1" in
    cuda|cuvs) ;;
    *) echo "Usage: $0 <cuda|cuvs>"; exit 1 ;;
esac

# GpuIndexIVFPQ is far more constrained than the CPU IndexIVFPQ, and the two
# build variants take different code paths (use_cuvs defaults to true whenever
# FAISS is compiled with cuVS). Constraints checked in
# GpuIndexIVFPQ::verifyPQSettings_():
#
#   cuda (non-cuVS path)
#     - nbits must be exactly 8 (interleavedLayout defaults to false)
#     - m must be a supported PQ code length AND divide d (=768)
#     - shared memory: 4 bytes * m * 2^nbits <= sharedMemPerBlock (48 KiB),
#       which caps m at 48 unless useFloat16LookupTables is enabled
#   cuvs
#     - nbits in [4,8], and (nbits * m) % 8 == 0
#
# The m values below are the intersection valid on both, so cuda and cuvs
# results stay directly comparable.

workspace=~/Git/profiling
build_dir=$workspace/retrieval-only/benchmarks-cpp/builds
querygen=$build_dir/build-$1/query_gen
idxgen_gpu_ivf_pq=$build_dir/build-$1/idxgen_gpu_ivf_pq
gpu_ivf_pq=$build_dir/build-$1/wikiall_gpu_ivf_pq
stats_dir=./results/param-sweep/gpu-ivf-pq-$1
stats_path=$stats_dir/gpu-ivf-pq-param-sweep.csv

mkdir -p "$stats_dir"
rm -f "$stats_path" # remove old stats file
cleanup             # remove stale indexes and queries

# logging: stderr from every binary invocation is appended here
log_dir=./logs
mkdir -p "$log_dir"
log_path="$log_dir/gpu-ivf-pq-param-sweep-$1-$(date +%Y%m%d-%H%M%S).log"
echo "=== gpu-ivf-pq-param-sweep | variant=$1 | started $(date -Is) ===" > "$log_path"
echo "stderr log: $log_path"

# runtime env params (not swept)
# Cap the BLAS pool so it cannot oversubscribe cores against FAISS's
# own OpenMP loops. FAISS parallelism (OMP_NUM_THREADS) is left alone.
openblas_threads=1
export OPENBLAS_NUM_THREADS=$openblas_threads

# workload params
k=10
nq=32
nb=100

$querygen $nq $k $nb > /dev/null 2>>"$log_path" # generate fixed batches of queries and groundtruths

# nlist sweep
nlist=(64 128 256 512 1024 2048 4096 8192)

nprobe=32

m=32

nbits=8

echo "nlist sweep..." | tee -a "$log_path"
start=$(date +%s%N)
for n in "${nlist[@]}"
do
    echo "Running nlist=$n" | tee -a "$log_path"

    $idxgen_gpu_ivf_pq $n $nprobe $nbits $m > /dev/null 2>>"$log_path" || { echo "  index build failed, skipping"; continue; }

    $gpu_ivf_pq $nb $stats_path > /dev/null 2>>"$log_path"

done
end=$(date +%s%N)
elapsed_s=$(awk "BEGIN {print ($end-$start)/1000000000}")
echo "$elapsed_s s"

# nprobe sweep
nlist=2048

nprobe=(1 2 4 8 16 32 64 128 256)

m=32

nbits=8

echo "nprobe sweep..." | tee -a "$log_path"
start=$(date +%s%N)
for n in "${nprobe[@]}"
do
    echo "Running nprobe=$n" | tee -a "$log_path"

    $idxgen_gpu_ivf_pq $nlist $n $nbits $m > /dev/null 2>>"$log_path" || { echo "  index build failed, skipping"; continue; }

    $gpu_ivf_pq $nb $stats_path > /dev/null 2>>"$log_path"

done
end=$(date +%s%N)
elapsed_s=$(awk "BEGIN {print ($end-$start)/1000000000}")
echo "$elapsed_s s"

# m sweep -- capped at 48 by the 48 KiB shared-memory limit on the cuda path
nlist=2048

nprobe=32

m=(8 12 16 24 32 48)

nbits=8

echo "m sweep..." | tee -a "$log_path"
start=$(date +%s%N)
for n in "${m[@]}"
do
    echo "Running m=$n" | tee -a "$log_path"

    $idxgen_gpu_ivf_pq $nlist $nprobe $nbits $n > /dev/null 2>>"$log_path" || { echo "  index build failed, skipping"; continue; }

    $gpu_ivf_pq $nb $stats_path > /dev/null 2>>"$log_path"

done
end=$(date +%s%N)
elapsed_s=$(awk "BEGIN {print ($end-$start)/1000000000}")
echo "$elapsed_s s"

# nbits sweep -- cuvs only; the non-cuVS path hard-requires nbits == 8
nlist=2048

nprobe=32

m=32

if [ "$1" = "cuvs" ]; then
    nbits=(4 5 6 7 8)

    echo "nbits sweep..." | tee -a "$log_path"
    start=$(date +%s%N)
    for n in "${nbits[@]}"
    do
        echo "Running nbits=$n" | tee -a "$log_path"

        $idxgen_gpu_ivf_pq $nlist $nprobe $n $m > /dev/null 2>>"$log_path" || { echo "  index build failed, skipping"; continue; }

        $gpu_ivf_pq $nb $stats_path > /dev/null 2>>"$log_path"

    done
    end=$(date +%s%N)
    elapsed_s=$(awk "BEGIN {print ($end-$start)/1000000000}")
    echo "$elapsed_s s"
else
    echo "nbits sweep skipped: the cuda (non-cuVS) path requires nbits == 8."
fi

echo "GPU IVF-PQ parameter sweep completed successfully."
cleanup

echo "Plotting results..."
source ${workspace}/.venv/bin/activate
python3 ./scripts/gpu-ivf-pq-param-sweep-graphing.py --build-type $1 --save
