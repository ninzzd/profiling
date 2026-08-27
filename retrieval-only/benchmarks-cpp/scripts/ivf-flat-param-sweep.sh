#!/bin/bash
cd "$(dirname "$0")/.." || exit 1

cleanup() {
    echo "Cleaning up..."
    rm -rf ./cpu-ivf-flat.index
    rm -rf ./queries
    rm -rf ./gt
    exit 130
}

trap cleanup SIGINT SIGTERM

workspace=~/Git/profiling
build_dir=$workspace/retrieval-only/benchmarks-cpp/builds
querygen=$build_dir/build-$1/query_gen
idxgen_ivf_flat=$build_dir/build-$1/idxgen_ivf_flat
ivf_flat=$build_dir/build-$1/wikiall_cpu_ivf_flat
stats_dir=./results/param-sweep/ivf-flat-$1
stats_path=$stats_dir/ivf-flat-param-sweep.csv

mkdir -p "$stats_dir"
rm -f "$stats_path" # remove old stats file

# nlist sweep
nlist=(64 128 256 512 1024 2048 4096 8192)

nprobe=32

# workload params
k=10
nq=32
nb=100

$querygen $nq $k $nb > /dev/null # generate fixed batches of queries and groundtruths
echo "nlist sweep..."
start=$(date +%s%N)
for n in "${nlist[@]}"
do
    echo "Running nlist=$n"

    $idxgen_ivf_flat $n $nprobe > /dev/null

    $ivf_flat $nb $stats_path > /dev/null

done
end=$(date +%s%N)
elapsed_s=$(awk "BEGIN {print ($end-$start)/1000000000}")
echo "$elapsed_s s"


nlist=2048
nprobe=(1 2 4 8 16 32 64 128 256)
echo "nrobe sweep..."
start=$(date +%s%N)
for n in "${nprobe[@]}"
do
    echo "Running nprobe=$n"

    $idxgen_ivf_flat $nlist $n > /dev/null

    $ivf_flat $nb $stats_path > /dev/null

done
end=$(date +%s%N)
elapsed_s=$(awk "BEGIN {print ($end-$start)/1000000000}")
echo "$elapsed_s s"

echo "Plotting results..."
source ${workspace}/.venv/bin/activate
python3 ./scripts/ivf-flat-param-sweep-graphing.py --build-type $1