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

workspace=~/Git/profiling
build_dir=$workspace/retrieval-only/benchmarks-cpp/builds
querygen=$build_dir/build-$1/query_gen
idxgen_ivf_pq=$build_dir/build-$1/idxgen_ivf_pq
ivf_pq=$build_dir/build-$1/wikiall_cpu_ivf_pq
stats_dir=./results/param-sweep/ivf-pq-$1
stats_path=$stats_dir/ivf-pq-param-sweep.csv

mkdir -p "$stats_dir"
rm -f "$stats_path" # remove old stats file

# workload params
k=10
nq=32
nb=100

# nlist sweep
nlist=(64 128 256 512 1024 2048 4096 8192)

nprobe=32

m=64

nbits=8

$querygen $nq $k $nb > /dev/null # generate fixed batches of queries and groundtruths
echo "nlist sweep..."
start=$(date +%s%N)
for n in "${nlist[@]}"
do
    echo "Running nlist=$n"

    $idxgen_ivf_pq $n $nprobe $nbits $m > /dev/null

    $ivf_pq $nb $stats_path > /dev/null

done
end=$(date +%s%N)
elapsed_s=$(awk "BEGIN {print ($end-$start)/1000000000}")
echo "$elapsed_s s"

# nprobe sweep
nlist=2048

nprobe=(1 2 4 8 16 32 64 128 256)

m=64

nbits=8

echo "nprobe sweep..."
start=$(date +%s%N)
for n in "${nprobe[@]}"
do
    echo "Running nprobe=$n"

    $idxgen_ivf_pq $nlist $n $nbits $m > /dev/null

    $ivf_pq $nb $stats_path > /dev/null

done
end=$(date +%s%N)
elapsed_s=$(awk "BEGIN {print ($end-$start)/1000000000}")
echo "$elapsed_s s"

# m sweep
nlist=2048

nprobe=32

m=(8 16 32 48 64 96 128 256)

nbits=8

echo "m sweep..."
start=$(date +%s%N)
for n in "${m[@]}"
do
    echo "Running m=$n"

    $idxgen_ivf_pq $nlist $nprobe $nbits $n > /dev/null

    $ivf_pq $nb $stats_path > /dev/null

done
end=$(date +%s%N)
elapsed_s=$(awk "BEGIN {print ($end-$start)/1000000000}")
echo "$elapsed_s s"

# nbits sweep
nlist=2048

nprobe=32

m=64

nbits=(4 5 6 7 8)

echo "nbits sweep..."
start=$(date +%s%N)
for n in "${nbits[@]}"
do
    echo "Running nbits=$n"

    $idxgen_ivf_pq $nlist $nprobe $n $m > /dev/null

    $ivf_pq $nb $stats_path > /dev/null

done
end=$(date +%s%N)
elapsed_s=$(awk "BEGIN {print ($end-$start)/1000000000}")
echo "$elapsed_s s"
echo "IVF-PQ parameter sweep completed successfully."
cleanup

echo "Plotting results..."
source ${workspace}/.venv/bin/activate
python3 ./scripts/ivf-pq-param-sweep-graphing.py --build-type $1