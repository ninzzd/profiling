#!/bin/bash
cd "$(dirname "$0")/.." || exit 1

cleanup() {
    rm -rf ./cpu-ivf-pq.index
    rm -rf ./queries
    rm -rf ./gt
}

interrupt() {
    echo "Interrupt received. Cleaning up..."
    cleanup
    exit 130
}

trap interrupt SIGINT SIGTERM

build_configs=("generic" "dd" "avx2" "avx512")

for n in "${build_configs[@]}"
do
    echo "Running sweeps for build type: $n ..."
    ./scripts/bs-sweep.sh $n
    ./scripts/hnsw-param-sweep.sh $n
    ./scripts/ivf-flat-param-sweep.sh $n
    ./scripts/ivf-pq-param-sweep.sh $n
done

cleanup