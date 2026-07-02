#!/bin/bash

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

querygen=./build/query_gen
idxgen_hnsw=./build/idxgen_hnsw
hnsw=./build/wikiall_cpu_hnsw

# workload params
k=10
nq=32
nb=100

$querygen $nq $k $nb > /dev/null # generate fixed batches of queries and groundtruths

# M sweep
M=(8 16 24 32 48 64)

efconstruction=200
efsearch=32

echo "M sweep..."
start=$(date +%s%N)
for m in "${M[@]}"
do
    echo "Running M=$m"

    $idxgen_hnsw $efconstruction $efsearch $m > /dev/null

    $hnsw $nb > /dev/null

done
end=$(date +%s%N)
elapsed_s=$(awk "BEGIN {print ($end-$start)/1000000000}")
echo "$elapsed_s s"

# efConstruction sweep
M=16

efconstruction=(40 80 120 160 200 300 400)

efsearch=32

echo "efConstruction sweep..."
start=$(date +%s%N)
for efc in "${efconstruction[@]}"
do
    echo "Running efConstruction=$efc"

    $idxgen_hnsw $efc $efsearch $M > /dev/null

    $hnsw $nb > /dev/null

done
end=$(date +%s%N)
elapsed_s=$(awk "BEGIN {print ($end-$start)/1000000000}")
echo "$elapsed_s s"

# efSearch sweep
M=16

efconstruction=200

efsearch=(4 8 16 32 64 128 256)

echo "efSearch sweep..."
start=$(date +%s%N)
for efs in "${efsearch[@]}"
do
    echo "Running efSearch=$efs"

    $idxgen_hnsw $efconstruction $efs $M > /dev/null

    $hnsw $nb > /dev/null

done
end=$(date +%s%N)
elapsed_s=$(awk "BEGIN {print ($end-$start)/1000000000}")
echo "$elapsed_s s"

echo "HNSW parameter sweep completed successfully."
cleanup