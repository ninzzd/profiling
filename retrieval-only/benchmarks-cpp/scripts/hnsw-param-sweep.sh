#!/bin/bash
cd "$(dirname "$0")/.." || exit 1

cleanup() {
    rm -rf ./cpu-hnsw.index
    rm -rf ./queries
    rm -rf ./gt
}

interrupt() {
    echo "Interrupt received. Cleaning up..."
    cleanup
    exit 130
}

trap interrupt SIGINT SIGTERM

workspace=~/Git/profiling
build_dir=$workspace/retrieval-only/benchmarks-cpp/builds
querygen=$build_dir/build-$1/query_gen
idxgen_hnsw=$build_dir/build-$1/idxgen_hnsw
hnsw=$build_dir/build-$1/wikiall_cpu_hnsw
stats_dir=./results/param-sweep/hnsw-$1
stats_path=$stats_dir/hnsw-param-sweep.csv

mkdir -p "$stats_dir"
rm -f "$stats_path" # remove old stats file

# logging: stderr from every binary invocation is appended here
log_dir=./logs
mkdir -p "$log_dir"
log_path="$log_dir/hnsw-param-sweep-$1-$(date +%Y%m%d-%H%M%S).log"
echo "=== hnsw-param-sweep | variant=$1 | started $(date -Is) ===" > "$log_path"
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

# M sweep
M=(8 16 24 32 48 64)

efconstruction=200
efsearch=32

echo "M sweep..." | tee -a "$log_path"
start=$(date +%s%N)
for m in "${M[@]}"
do
    echo "Running M=$m" | tee -a "$log_path"

    $idxgen_hnsw $efconstruction $efsearch $m > /dev/null 2>>"$log_path"

    $hnsw $nb $stats_path > /dev/null 2>>"$log_path"

done
end=$(date +%s%N)
elapsed_s=$(awk "BEGIN {print ($end-$start)/1000000000}")
echo "$elapsed_s s"

# efConstruction sweep
M=16

efconstruction=(40 80 120 160 200 300 400)

efsearch=32

echo "efConstruction sweep..." | tee -a "$log_path"
start=$(date +%s%N)
for efc in "${efconstruction[@]}"
do
    echo "Running efConstruction=$efc" | tee -a "$log_path"

    $idxgen_hnsw $efc $efsearch $M > /dev/null 2>>"$log_path"

    $hnsw $nb $stats_path > /dev/null 2>>"$log_path"

done
end=$(date +%s%N)
elapsed_s=$(awk "BEGIN {print ($end-$start)/1000000000}")
echo "$elapsed_s s"

# efSearch sweep
M=16

efconstruction=200

efsearch=(4 8 16 32 64 128 256)

echo "efSearch sweep..." | tee -a "$log_path"
start=$(date +%s%N)
for efs in "${efsearch[@]}"
do
    echo "Running efSearch=$efs" | tee -a "$log_path"

    $idxgen_hnsw $efconstruction $efs $M > /dev/null 2>>"$log_path"

    $hnsw $nb $stats_path > /dev/null 2>>"$log_path"

done
end=$(date +%s%N)
elapsed_s=$(awk "BEGIN {print ($end-$start)/1000000000}")
echo "$elapsed_s s"

echo "HNSW parameter sweep completed successfully."
cleanup

echo "Plotting results..."
source ${workspace}/.venv/bin/activate
python3 ./scripts/hnsw-param-sweep-graphing.py --save --build-type $1
