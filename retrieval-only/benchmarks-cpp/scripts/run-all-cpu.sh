#!/bin/bash
# Runs every CPU sweep across all CPU build variants.
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

# runtime env params (not swept)
# Set here too so the value is visible at the top level; each sweep
# script also sets it independently when run standalone.
openblas_threads=1
export OPENBLAS_NUM_THREADS=$openblas_threads

# logging: the driver's own log; each sweep script writes its own alongside it
log_dir=./logs
mkdir -p "$log_dir"
log_path="$log_dir/run-all-cpu-$(date +%Y%m%d-%H%M%S).log"
echo "=== run-all-cpu | started $(date -Is) ===" > "$log_path"
echo "driver log: $log_path"

build_configs=("generic" "dd" "avx2" "avx512")

total_start=$(date +%s%N)
for n in "${build_configs[@]}"
do
    echo "Running CPU sweeps for build type: $n ..." | tee -a "$log_path"
    ./scripts/bs-sweep.sh "$n" cpu-only 2>>"$log_path"
    ./scripts/hnsw-param-sweep.sh "$n" 2>>"$log_path"
    ./scripts/ivf-flat-param-sweep.sh "$n" 2>>"$log_path"
    ./scripts/ivf-pq-param-sweep.sh "$n" 2>>"$log_path"
done
total_end=$(date +%s%N)

cleanup

elapsed_s=$(awk "BEGIN {print ($total_end-$total_start)/1000000000}")
echo "All CPU sweeps completed in $elapsed_s s" | tee -a "$log_path"
