#!/bin/bash

querygen=./build/query_gen
base=./build/wikiall_cpu_baseline
hnsw=./build/wikiall_cpu_hnsw
ivf_flat=./build/wikiall_cpu_ivf_flat
ivf_pq=./build/wikiall_cpu_ivf_pq

# fixed workload params
k=10
iter=100

# fixed hnsw params
efconstruction=200
efsearch=32
M=16

# fixed ivf flat/pq params
nlist=2048
nprobe=32
nbits=8
m=64

# sweep array
batches=(1 2 4 8 16 32 64 128 256 512 1024)

start=$(date +%s%N)
for nq in "${batches[@]}"
do
    echo "Running nq=$nq"

    $querygen $nq > /dev/null

    $base $k $iter > /dev/null

    $hnsw $efconstruction $efsearch $M > /dev/null

    $ivf_flat $nlist $nprobe > /dev/null

    $ivf_pq $nlist $nprobe $nbits $m > /dev/null

done
end=$(date +%s%N)
elapsed_s=$(awk "BEGIN {print ($end-$start)/1000000000}")
echo "$elapsed_s s"