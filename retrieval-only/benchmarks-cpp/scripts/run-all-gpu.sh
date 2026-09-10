#!/bin/bash
# Runs every GPU sweep across the GPU build variants.
#
# Only cuda and cuvs produce GPU executables (see FAISS_GPU_VARIANT in
# CMakeLists.txt); CAGRA additionally requires cuVS, so it runs for cuvs only.
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
log_path="$log_dir/run-all-gpu-$(date +%Y%m%d-%H%M%S).log"
echo "=== run-all-gpu | started $(date -Is) ===" > "$log_path"
echo "driver log: $log_path"

build_configs=("cuda" "cuvs")

total_start=$(date +%s%N)
for n in "${build_configs[@]}"
do
    echo "Running GPU sweeps for build type: $n ..." | tee -a "$log_path"
    ./scripts/bs-sweep.sh "$n" gpu-only 2>>"$log_path"
    ./scripts/gpu-ivf-flat-param-sweep.sh "$n" 2>>"$log_path"
    ./scripts/gpu-ivf-pq-param-sweep.sh "$n" 2>>"$log_path"

    # CAGRA is cuvs-only
    if [ "$n" = "cuvs" ]; then
        ./scripts/gpu-cagra-param-sweep.sh "$n" 2>>"$log_path"
    fi
done
total_end=$(date +%s%N)

cleanup

elapsed_s=$(awk "BEGIN {print ($total_end-$total_start)/1000000000}")
echo "All GPU sweeps completed in $elapsed_s s" | tee -a "$log_path"
