#!/bin/bash
cd "$(dirname "$0")/.." || exit 1

cleanup() {
    rm -rf ./gpu-cagra.index
    rm -rf ./queries
    rm -rf ./gt
}

interrupt() {
    echo "Interrupt received. Cleaning up..."
    cleanup
    exit 130
}

trap interrupt SIGINT SIGTERM

# CAGRA is the GPU counterpart of HNSW and is only compiled in when FAISS is
# built with FAISS_ENABLE_CUVS=ON, i.e. the cuvs variant alone
# (see the FAISS_BUILD_VARIANT STREQUAL "cuvs" block in CMakeLists.txt)
if [ "$1" != "cuvs" ]; then
    echo "Usage: $0 cuvs"
    echo "GpuIndexCagra requires the cuvs build variant."
    exit 1
fi

workspace=~/Git/profiling
build_dir=$workspace/retrieval-only/benchmarks-cpp/builds
querygen=$build_dir/build-$1/query_gen
idxgen_gpu_cagra=$build_dir/build-$1/idxgen_gpu_cagra
gpu_cagra=$build_dir/build-$1/wikiall_gpu_cagra
stats_dir=./results/param-sweep/gpu-cagra-$1
stats_path=$stats_dir/gpu-cagra-param-sweep.csv

mkdir -p "$stats_dir"
rm -f "$stats_path" # remove old stats file
cleanup             # remove stale indexes and queries

# logging: stderr from every binary invocation is appended here
log_dir=./logs
mkdir -p "$log_dir"
log_path="$log_dir/gpu-cagra-param-sweep-$1-$(date +%Y%m%d-%H%M%S).log"
echo "=== gpu-cagra-param-sweep | variant=$1 | started $(date -Is) ===" > "$log_path"
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

# graph_degree sweep (the CAGRA analogue of HNSW's M).
# cuVS requires intermediate_graph_degree >= graph_degree, so the fixed
# intermediate degree below is the largest value the sweep reaches.
intermediate_graph_degree=128

graph_degree=(16 32 48 64 96 128)

echo "graph_degree sweep..." | tee -a "$log_path"
start=$(date +%s%N)
for g in "${graph_degree[@]}"
do
    echo "Running graph_degree=$g" | tee -a "$log_path"

    $idxgen_gpu_cagra $intermediate_graph_degree $g > /dev/null 2>>"$log_path" || { echo "  index build failed, skipping"; continue; }

    $gpu_cagra $nb $stats_path > /dev/null 2>>"$log_path"

done
end=$(date +%s%N)
elapsed_s=$(awk "BEGIN {print ($end-$start)/1000000000}")
echo "$elapsed_s s"

# intermediate_graph_degree sweep (the pruning pool CAGRA builds before
# optimising down to graph_degree -- loosely the analogue of efConstruction)
graph_degree=64

intermediate_graph_degree=(64 96 128 192 256)

echo "intermediate_graph_degree sweep..." | tee -a "$log_path"
start=$(date +%s%N)
for i in "${intermediate_graph_degree[@]}"
do
    echo "Running intermediate_graph_degree=$i" | tee -a "$log_path"

    $idxgen_gpu_cagra $i $graph_degree > /dev/null 2>>"$log_path" || { echo "  index build failed, skipping"; continue; }

    $gpu_cagra $nb $stats_path > /dev/null 2>>"$log_path"

done
end=$(date +%s%N)
elapsed_s=$(awk "BEGIN {print ($end-$start)/1000000000}")
echo "$elapsed_s s"

echo "GPU CAGRA parameter sweep completed successfully."
cleanup

# NOTE: there is no search-side sweep here. itopk_size (CAGRA's efSearch
# analogue) is hardcoded to 64 in src/wikiall-gpu-cagra.cpp rather than taken
# from argv, so it cannot be varied from this script as things stand.
#
# NOTE: no plotting step -- the existing *-graphing.py scripts are hardwired to
# the CPU result directories (results/param-sweep/hnsw-<variant>).
