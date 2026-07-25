#!/bin/bash
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

workspace=~/Git/profiling
build_dir=$workspace/retrieval-only/benchmarks-cpp/builds
querygen=.$build_dir/build-$1/query_gen
idxgen_base=.$build_dir/build-$1/idxgen_baseline
idxgen_hnsw=.$build_dir/build-$1/idxgen_hnsw
idxgen_ivf_flat=.$build_dir/build-$1/idxgen_ivf_flat
idxgen_ivf_pq=.$build_dir/build-$1/idxgen_ivf_pq
base=.$build_dir/build-$1/wikiall_cpu_baseline
hnsw=.$build_dir/build-$1/wikiall_cpu_hnsw
ivf_flat=.$build_dir/build-$1/wikiall_cpu_ivf_flat
ivf_pq=.$build_dir/build-$1/wikiall_cpu_ivf_pq
stats_path=.results/workload-sweep/bs-sweep-stats.csv

rm -rf $stats_path # remove old stats file
cleanup # remove old indexes and queries

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

# building indexes
echo "Building indexes..."
$idxgen_base > /dev/null & \
$idxgen_hnsw $efconstruction $efsearch $M > /dev/null & \
$idxgen_ivf_flat $nlist $nprobe > /dev/null & \
$idxgen_ivf_pq $nlist $nprobe $nbits $m > /dev/null
wait
echo "Build complete"

# sweep array
batches=(1 2 4 8 16 32 64 128 256 512 1024)

start=$(date +%s%N)
for nq in "${batches[@]}"
do
    echo "Running nq=$nq"

    $querygen $nq $k $nb > /dev/null

    $base $nb $stats_path> /dev/null

    $hnsw $nb $stats_path> /dev/null

    $ivf_flat $nb $stats_path> /dev/null

    $ivf_pq $nb $stats_path> /dev/null

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
python3 ./scripts/bs-sweep-graphing.py --build-type $1