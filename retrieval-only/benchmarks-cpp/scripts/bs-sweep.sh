#!/bin/bash
cd "$(dirname "$0")/.." || exit 1

cleanup() {
    rm -rf ./*.index
    rm -rf ./queries
    rm -rf ./gt
}

interrupt() {
    echo "Interrupt received. Cleaning up..."
    cleanup
    exit 130
}

trap interrupt SIGINT SIGTERM

variant=$1
mode=${2:-all}

usage() {
    echo "Usage: $0 <generic|dd|avx2|avx512|cuda|cuvs> [all|cpu-only|gpu-only]"
    exit 1
}

case "$variant" in
    generic|dd|avx2|avx512|cuda|cuvs) ;;
    *) usage ;;
esac

case "$mode" in
    all|cpu-only|gpu-only) ;;
    *) usage ;;
esac

# GPU executables are only produced for the cuda and cuvs variants, and CAGRA
# additionally requires cuVS (see FAISS_GPU_VARIANT in CMakeLists.txt).
run_gpu=0
run_cagra=0
case "$variant" in
    cuda) run_gpu=1 ;;
    cuvs) run_gpu=1; run_cagra=1 ;;
esac

if [ "$mode" = "gpu-only" ] && [ $run_gpu -eq 0 ]; then
    echo "gpu-only requires a GPU build variant (cuda or cuvs)."
    exit 1
fi

run_cpu=1
[ "$mode" = "gpu-only" ] && run_cpu=0
[ "$mode" = "cpu-only" ] && { run_gpu=0; run_cagra=0; }

workspace=~/Git/profiling
build_dir=$workspace/retrieval-only/benchmarks-cpp/builds
querygen=$build_dir/build-$variant/query_gen
idxgen_base=$build_dir/build-$variant/idxgen_baseline
idxgen_hnsw=$build_dir/build-$variant/idxgen_hnsw
idxgen_ivf_flat=$build_dir/build-$variant/idxgen_ivf_flat
idxgen_ivf_pq=$build_dir/build-$variant/idxgen_ivf_pq
idxgen_gpu_ivf_flat=$build_dir/build-$variant/idxgen_gpu_ivf_flat
idxgen_gpu_ivf_pq=$build_dir/build-$variant/idxgen_gpu_ivf_pq
idxgen_gpu_cagra=$build_dir/build-$variant/idxgen_gpu_cagra
base=$build_dir/build-$variant/wikiall_cpu_baseline
hnsw=$build_dir/build-$variant/wikiall_cpu_hnsw
ivf_flat=$build_dir/build-$variant/wikiall_cpu_ivf_flat
ivf_pq=$build_dir/build-$variant/wikiall_cpu_ivf_pq
gpu_ivf_flat=$build_dir/build-$variant/wikiall_gpu_ivf_flat
gpu_ivf_pq=$build_dir/build-$variant/wikiall_gpu_ivf_pq
gpu_cagra=$build_dir/build-$variant/wikiall_gpu_cagra
stats_dir=./results/workload-sweep-$variant
stats_path=$stats_dir/bs-sweep-stats.csv

mkdir -p "$stats_dir"
# Each invocation writes a fresh CSV holding exactly the indexes it ran, so a
# cpu-only/gpu-only run does not leave stale rows from a previous sweep behind.
rm -f "$stats_path" # remove old stats file
cleanup # remove old indexes and queries

# logging: stderr from every binary invocation is appended here
log_dir=./logs
mkdir -p "$log_dir"
log_path="$log_dir/bs-sweep-$variant-$(date +%Y%m%d-%H%M%S).log"
echo "=== bs-sweep | variant=$variant | started $(date -Is) ===" > "$log_path"
echo "stderr log: $log_path"

# runtime env params (not swept)
# Cap the BLAS pool so it cannot oversubscribe cores against FAISS's
# own OpenMP loops. FAISS parallelism (OMP_NUM_THREADS) is left alone.
openblas_threads=1
export OPENBLAS_NUM_THREADS=$openblas_threads

# fixed workload params
k=10
nb=100

# fixed hnsw params
efconstruction=200
efsearch=32
M=16

# fixed ivf flat/pq params
nlist=2048
nprobe=32
nbits=8
m=64

# fixed gpu ivf-pq params -- GpuIndexIVFPQ is more constrained than the CPU
# IndexIVFPQ: m must divide d (=768) and the 48 KiB shared-memory limit on the
# non-cuVS path caps it at 48, so the CPU m=64 is not usable here.
gpu_m=32

# fixed cagra params (cuvs only); cuVS requires
# intermediate_graph_degree >= graph_degree
intermediate_graph_degree=128
graph_degree=64

# building indexes
echo "Building indexes..." | tee -a "$log_path"
if [ $run_cpu -eq 1 ]; then
    $idxgen_base > /dev/null 2>>"$log_path" & \
    $idxgen_hnsw $efconstruction $efsearch $M > /dev/null 2>>"$log_path" & \
    $idxgen_ivf_flat $nlist $nprobe > /dev/null 2>>"$log_path" & \
    $idxgen_ivf_pq $nlist $nprobe $nbits $m > /dev/null 2>>"$log_path"
    wait
fi

# GPU indexes are built one at a time -- concurrent builds would contend for
# the same device memory.
if [ $run_gpu -eq 1 ]; then
    $idxgen_gpu_ivf_flat $nlist $nprobe > /dev/null 2>>"$log_path" || echo "  gpu-ivf-flat index build failed, skipping"
    $idxgen_gpu_ivf_pq $nlist $nprobe $nbits $gpu_m > /dev/null 2>>"$log_path" || echo "  gpu-ivf-pq index build failed, skipping"
fi

if [ $run_cagra -eq 1 ]; then
    $idxgen_gpu_cagra $intermediate_graph_degree $graph_degree > /dev/null 2>>"$log_path" || echo "  gpu-cagra index build failed, skipping"
fi
echo "Build complete"

# sweep array
batches=(1 2 4 8 16 32 64 128 256 512 1024)

start=$(date +%s%N)
for nq in "${batches[@]}"
do
    echo "Running nq=$nq" | tee -a "$log_path"

    $querygen $nq $k $nb > /dev/null 2>>"$log_path"

    if [ $run_cpu -eq 1 ]; then
        $base $nb $stats_path> /dev/null 2>>"$log_path"

        $hnsw $nb $stats_path> /dev/null 2>>"$log_path"

        $ivf_flat $nb $stats_path> /dev/null 2>>"$log_path"

        $ivf_pq $nb $stats_path> /dev/null 2>>"$log_path"
    fi

    if [ $run_gpu -eq 1 ]; then
        [ -f ./gpu-ivf-flat.index ] && $gpu_ivf_flat $nb $stats_path > /dev/null 2>>"$log_path"

        [ -f ./gpu-ivf-pq.index ] && $gpu_ivf_pq $nb $stats_path > /dev/null 2>>"$log_path"
    fi

    if [ $run_cagra -eq 1 ]; then
        [ -f ./gpu-cagra.index ] && $gpu_cagra $nb $stats_path > /dev/null 2>>"$log_path"
    fi

    rm -rf ./queries
    rm -rf ./gt

done
end=$(date +%s%N)
elapsed_s=$(awk "BEGIN {print ($end-$start)/1000000000}")
echo "$elapsed_s s"

echo "Batch size sweep complete. Cleaning up..."
cleanup

echo "Plotting results..."
source ${workspace}/.venv/bin/activate
python3 ./scripts/bs-sweep-graphing.py --save --build-type $variant
