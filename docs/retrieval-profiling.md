# Retrieval Profiling and Benchmarking with FAISS

## Build Configurations

To support debugging, detailed profiling, GPU/SIMD acceleration and other options, it is better to build from source. Hence, I have added the [FAISS](https://github.com/facebookresearch/faiss.git) repository as a submodule in the directory `retireval-only/faiss`

### Dependencies

FAISS builds for SIMD and other configurations, require: 
    
    - OpenBlas

### CPU-Only
This configuration includes:

    - C++ library (w/o python)
    - AVX2 SIMD utilization
    - Debugging with info

Here's the cmake command to create the build configuration:
```bash
cd retrieval-only/faiss # Run the following cmake command from inside faiss directory
cmake -B ../faiss-builds/cpu-debug-cpp \
    -DBUILD_SHARED_LIBS=ON \
    -DFAISS_ENABLE_GPU=OFF \
    -DFAISS_ENABLE_PYTHON=OFF \
    -DBUILD_TESTING=OFF \
    -DFAISS_ENABLE_MKL=OFF \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

Build the given configuration:
```bash
# keep it safe, don't use too many cores
cmake --build ../faiss-builds/cpu-debug-cpp -j8 
```
## Simple Benchmarking

### General Build Commands
**Configuration:**
```bash
cmake -S . -B build
```

**Build:**
```bash
make -C build -j$(nproc)
```

### Simple CPU-Only Baseline (Toy)

(change documentation based on new knowledge about `IndexFlatL2`, checkout **locality-sensitive hashing**, understand its applicability for FPGA-based acceleration)

Refer to [`simple-cpu-debug.cpp`](./../retrieval-only/benchmarks-cpp/simple-cpu-debug.cpp) source file. An IVF database of with 100,000 vectors was initialized, with an embedding dimensionality of 768. A standard uniform random distribution $U[0,1]$ was sampled to initialize the components of all the vectors. The same sampling was done for obtaining 1000 query vectors. 

After creating the database vectors, adding them to the IVF index, an object of `faiss::IndexFlatL2`(flat L2, i.e. row-major order of vector components, L2-normed distance) and creating the query vectors, the queries were searched. The IVF initialization time and search time were also calculated.

\*\*Note: `IndexFlatL2` uses ENN (exact nearest-neighbour search)

**No-Tool Profiling (std::chrono)**

Run:
```bash
./build/simple_cpu_debug
```
Example output:
```bash
Index size: 100000 vectors
Add time: 0.0623132 s
Search time: 1.76192 s
Queries/sec: 567.563
Latency/query: 1761.92 us
First neighbor ID: 76510, distance: 106.56
```

**Perf Stats**

Run:
```bash
perf stats ./build/simple_cpu_debug
```
Example output:
```text
Index size: 100000 vectors
Add time: 0.0672288 s
Search time: 2.11102 s
Queries/sec: 473.705
Latency/query: 2111.02 us
First neighbor ID: 76510, distance: 106.56

 Performance counter stats for './build/simple_cpu_debug':

    38,566,040,186      task-clock                       #    3.859 CPUs utilized             
         2,598,867      context-switches                 #   67.387 K/sec                     
               173      cpu-migrations                   #    4.486 /sec                      
           156,777      page-faults                      #    4.065 K/sec                     
   224,976,150,169      instructions                     #    1.22  insn per cycle            
                                                  #    0.15  stalled cycles per insn   
   184,821,061,849      cycles                           #    4.792 GHz                       
    33,824,624,093      stalled-cycles-frontend          #   18.30% frontend cycles idle      
    37,422,130,061      branches                         #  970.339 M/sec                     
       233,682,833      branch-misses                    #    0.62% of all branches           

       9.994306125 seconds time elapsed

      22.542640000 seconds user
      16.023611000 seconds sys
```

For a more detailed output, run:

```bash
perf stat -d ./build/simple_cpu_debug
```

Example output:

```text
Index size: 100000 vectors
Add time: 0.0676504 s
Search time: 1.66923 s
Queries/sec: 599.077
Latency/query: 1669.23 us
First neighbor ID: 76510, distance: 106.56

 Performance counter stats for './build/simple_cpu_debug':

    32,421,957,859      task-clock                       #    3.389 CPUs utilized             
         1,993,540      context-switches                 #   61.487 K/sec                     
               210      cpu-migrations                   #    6.477 /sec                      
           154,209      page-faults                      #    4.756 K/sec                     
   170,500,306,145      instructions                     #    1.08  insn per cycle            
                                                  #    0.16  stalled cycles per insn     (71.37%)
   157,897,445,884      cycles                           #    4.870 GHz                         (71.37%)
    27,904,770,154      stalled-cycles-frontend          #   17.67% frontend cycles idle        (71.41%)
    27,804,037,355      branches                         #  857.568 M/sec                       (71.45%)
       171,919,845      branch-misses                    #    0.62% of all branches             (71.49%)
    97,467,547,388      L1-dcache-loads                  #    3.006 G/sec                       (71.49%)
     1,271,384,080      L1-dcache-load-misses            #    1.30% of all L1-dcache accesses   (71.44%)

       9.566664370 seconds time elapsed

      21.374901000 seconds user
      11.047600000 seconds sys
```

### Dataset
The 1M subset of the Wiki-All 88M dataset is being used for creating the index. A real dataset and vector database would provide good insights on embedding-space locality, it's effect on performance and real retrieval workload profiling

Download the tar-archived dataset:
```bash
curl -O https://data.rapids.ai/raft/datasets/wiki_all_1M/wiki_all_1M.tar
```

Extract the dataset:
```bash
tar -xvf wiki_all_1M.tar -C retrieval-only/datasets/wikiall
```

### Wiki-All CPU-Only Flat Index Baseline
- File: `wikiall-cpu-flat.cpp`
- Wikiall 1M dataset
- Created `IndexFlatL2` vector database
- Query vectors - 100 (picked randomly, psuedorandom)
- Top-10 vector retrieval

**Run:**
Direct:
```bash
./build/wikiall_cpu_flat
```
Output:
```bash
Read complete!
Read time: 0.880952 s
Index creation complete!
Index size: 1000000 vectors
Index dimensions:768
Index creation latency: 0.596113 s
Search complete!
Latency: 2.91372 s
Latency/query: 0.0291372 s
Throughput: 34.3204 queries/s
```

Perf stat (detailed):
```bash
perf stat -d ./build/wikiall_cpu_flat
```

Output:
```text
Read complete!
Read time: 0.884355 s
Index creation complete!
Index size: 1000000 vectors
Index dimensions:768
Index creation latency: 0.613298 s
Search complete!
Latency: 2.70386 s
Latency/query: 0.0270386 s
Throughput: 36.9841 queries/s

 Performance counter stats for './build/wikiall_cpu_flat':

    42,289,072,669      task-clock                       #    9.830 CPUs utilized             
             3,451      context-switches                 #   81.605 /sec                      
                26      cpu-migrations                   #    0.615 /sec                      
         1,500,917      page-faults                      #   35.492 K/sec                     
   363,746,372,909      instructions                     #    1.86  insn per cycle            
                                                  #    0.01  stalled cycles per insn     (71.44%)
   195,923,337,229      cycles                           #    4.633 GHz                         (71.38%)
     3,220,602,258      stalled-cycles-frontend          #    1.64% frontend cycles idle        (71.41%)
    14,457,126,328      branches                         #  341.864 M/sec                       (71.44%)
        36,497,814      branch-misses                    #    0.25% of all branches             (71.45%)
   169,755,165,885      L1-dcache-loads                  #    4.014 G/sec                       (71.45%)
     5,228,199,119      L1-dcache-load-misses            #    3.08% of all L1-dcache accesses   (71.47%)

       4.302257528 seconds time elapsed

      40.306320000 seconds user
       1.977347000 seconds sys
```

### Wiki-All CPU-Only IVF Flat Index

**Direct Run:**
```bash
perf record ./build/wikiall_cpu_ivf_flat
```

Output:
```
IVF Properties:
nlist: 100
nprobe: 10
Reading dataset from disk...
Read complete!
Index size: 1000000 
Index dimensions:768
Read time: 0.865372 s
Generating query vectors...
Query generation complete!
Query generation time: 3.2112e-05 s
Training the IVF index...
Training complete!
Training time: 0.5827 s
Adding vectors to the IVF index...
Index created!
Index creation latency: 0.894535 s
Search complete!
Latency: 0.658974 s
Latency/query: 0.00658974 s
Throughput: 151.751 queries/s
```

**Perf Stat (Detailed):**
```bash
perf stat -d ./build/wikiall_cpu_ivf_flat 
```
Output:
```
IVF Properties:
nlist: 100
nprobe: 10
Reading dataset from disk...
Read complete!
Index size: 1000000 
Index dimensions:768
Read time: 0.880251 s
Generating query vectors...
Query generation complete!
Query generation time: 3.8002e-05 s
Training the IVF index...
Training complete!
Training time: 0.528499 s
Adding vectors to the IVF index...
Index created!
Index creation latency: 0.885306 s
Search complete!
Latency: 0.656515 s
Latency/query: 0.00656515 s
Throughput: 152.319 queries/s

 Performance counter stats for './build/wikiall_cpu_ivf_flat':

    30,384,301,734      task-clock                       #    9.849 CPUs utilized             
            68,847      context-switches                 #    2.266 K/sec                     
                88      cpu-migrations                   #    2.896 /sec                      
         2,152,987      page-faults                      #   70.859 K/sec                     
   154,205,284,898      instructions                     #    1.00  insn per cycle            
                                                  #    0.07  stalled cycles per insn     (71.44%)
   153,502,141,029      cycles                           #    5.052 GHz                         (71.40%)
    11,480,906,802      stalled-cycles-frontend          #    7.48% frontend cycles idle        (71.44%)
    19,413,904,656      branches                         #  638.945 M/sec                       (71.44%)
        97,831,053      branch-misses                    #    0.50% of all branches             (71.41%)
    68,793,188,429      L1-dcache-loads                  #    2.264 G/sec                       (71.45%)
     2,596,253,840      L1-dcache-load-misses            #    3.77% of all L1-dcache accesses   (71.47%)

       3.084936270 seconds time elapsed

      20.885495000 seconds user
       9.503220000 seconds sys
```

**Iterative Run:**

The same file `wikiall-cpu-ivf-flat.cpp` has been modified to support iterative execution for finding important statistical performance metrics.
Run:
```bash
./build/wikiall_cpu_ivf_flat 0 20
```

Output:
```
IVF Properties:
nlist: 100
nprobe: 10
Reading dataset from disk...
Read complete!
Index size: 1000000 
Index dimensions:768
Read time: 0.875017 s
Training the IVF index...
Training complete!
Training time: 0.555977 s
Adding vectors to the IVF index...
Index created!
Index creation latency: 0.846025 s
Iteration: 1
Iteration: 2
Iteration: 3
Iteration: 4
Iteration: 5
Iteration: 6
Iteration: 7
Iteration: 8
Iteration: 9
Iteration: 10
Iteration: 11
Iteration: 12
Iteration: 13
Iteration: 14
Iteration: 15
Iteration: 16
Iteration: 17
Iteration: 18
Iteration: 19
Iteration: 20
P50 Latency: 0.636953s
P90 Latency: 0.655355s
P95 Latency: 0.657642s
P99 Latency: 0.657642s
Mean Latency: 0.635972s
Peak Throughput: 164.273qps
Mean Throughput: 157.305qps
```

### Wiki-All CPU-Only IVF PQ Index

**Direct Run:**

```
IVF Properties:
nlist: 100
nprobe: 10
Reading dataset from disk...
Read complete!
Index size: 1000000 
Index dimensions:768
Read time: 0.866255 s
Training the IVF index...
Training complete!
Training time: 81.6252 s
Adding vectors to the IVF index...
Index created!
Index creation latency: 7.51032 s
Iteration: 1
Iteration: 2
Iteration: 3
Iteration: 4
Iteration: 5
Iteration: 6
Iteration: 7
Iteration: 8
Iteration: 9
Iteration: 10
Iteration: 11
Iteration: 12
Iteration: 13
Iteration: 14
Iteration: 15
Iteration: 16
Iteration: 17
Iteration: 18
Iteration: 19
Iteration: 20
Iteration: 21
Iteration: 22
Iteration: 23
Iteration: 24
Iteration: 25
Iteration: 26
Iteration: 27
Iteration: 28
Iteration: 29
Iteration: 30
Iteration: 31
Iteration: 32
Iteration: 33
Iteration: 34
Iteration: 35
Iteration: 36
Iteration: 37
Iteration: 38
Iteration: 39
Iteration: 40
Iteration: 41
Iteration: 42
Iteration: 43
Iteration: 44
Iteration: 45
Iteration: 46
Iteration: 47
Iteration: 48
Iteration: 49
Iteration: 50
Iteration: 51
Iteration: 52
Iteration: 53
Iteration: 54
Iteration: 55
Iteration: 56
Iteration: 57
Iteration: 58
Iteration: 59
Iteration: 60
Iteration: 61
Iteration: 62
Iteration: 63
Iteration: 64
Iteration: 65
Iteration: 66
Iteration: 67
Iteration: 68
Iteration: 69
Iteration: 70
Iteration: 71
Iteration: 72
Iteration: 73
Iteration: 74
Iteration: 75
Iteration: 76
Iteration: 77
Iteration: 78
Iteration: 79
Iteration: 80
Iteration: 81
Iteration: 82
Iteration: 83
Iteration: 84
Iteration: 85
Iteration: 86
Iteration: 87
Iteration: 88
Iteration: 89
Iteration: 90
Iteration: 91
Iteration: 92
Iteration: 93
Iteration: 94
Iteration: 95
Iteration: 96
Iteration: 97
Iteration: 98
Iteration: 99
Iteration: 100
P50 Latency: 0.0240284s
P90 Latency: 0.0281369s
P95 Latency: 0.029406s
P99 Latency: 0.0366735s
Mean Latency: 0.0247756s
Peak Throughput: 4538.92qps
Mean Throughput: 4064.9qps
```

\*\*Note: Training time is significantly longer, most probably due to subspace k-means clustering
This training can be accelerated by the GPU, buton CPU-only execution, training remains the primary bottleneck

### Wiki-All CPU-Only HNSW Flat Index

**Direct Run:**
```
HNSW Properties:
efConstruction: 200
efSearch: 64
M: 16
Reading dataset from disk...
Read complete!
Index size: 1000000 
Index dimensions:768
Read time: 0.855234 s
Adding vectors to the IVF index...
Index created!
Index creation latency: 170.94 s
Iteration: 1
Iteration: 2
Iteration: 3
Iteration: 4
Iteration: 5
Iteration: 6
Iteration: 7
Iteration: 8
Iteration: 9
Iteration: 10
Iteration: 11
Iteration: 12
Iteration: 13
Iteration: 14
Iteration: 15
Iteration: 16
Iteration: 17
Iteration: 18
Iteration: 19
Iteration: 20
Iteration: 21
Iteration: 22
Iteration: 23
Iteration: 24
Iteration: 25
Iteration: 26
Iteration: 27
Iteration: 28
Iteration: 29
Iteration: 30
Iteration: 31
Iteration: 32
Iteration: 33
Iteration: 34
Iteration: 35
Iteration: 36
Iteration: 37
Iteration: 38
Iteration: 39
Iteration: 40
Iteration: 41
Iteration: 42
Iteration: 43
Iteration: 44
Iteration: 45
Iteration: 46
Iteration: 47
Iteration: 48
Iteration: 49
Iteration: 50
Iteration: 51
Iteration: 52
Iteration: 53
Iteration: 54
Iteration: 55
Iteration: 56
Iteration: 57
Iteration: 58
Iteration: 59
Iteration: 60
Iteration: 61
Iteration: 62
Iteration: 63
Iteration: 64
Iteration: 65
Iteration: 66
Iteration: 67
Iteration: 68
Iteration: 69
Iteration: 70
Iteration: 71
Iteration: 72
Iteration: 73
Iteration: 74
Iteration: 75
Iteration: 76
Iteration: 77
Iteration: 78
Iteration: 79
Iteration: 80
Iteration: 81
Iteration: 82
Iteration: 83
Iteration: 84
Iteration: 85
Iteration: 86
Iteration: 87
Iteration: 88
Iteration: 89
Iteration: 90
Iteration: 91
Iteration: 92
Iteration: 93
Iteration: 94
Iteration: 95
Iteration: 96
Iteration: 97
Iteration: 98
Iteration: 99
Iteration: 100
P50 Latency: 0.0074151s
P90 Latency: 0.0146785s
P95 Latency: 0.0148091s
P99 Latency: 0.0148241s
Mean Latency: 0.00881591s
Peak Throughput: 14779.5qps
Mean Throughput: 11992qps
```

## Inter-Index Comparison Benchmarks (Workload Sweeps)
- Generate a common query batch
- Run baseline
- Run all benchmarks, compare with baseline, check recall@k and latency
- Sweep over query batch size, k

### Single Query Batch Tests
- nq = 10
- k = 100
- iterations = 100

### Baseline (Contiguous array index - Flat - Inner-Product Metric)
```
Reading dataset from disk...
Index read complete!
Index read time: 0.855165 s
Query read complete!
Query read time: 1.7874e-05 s
Query batch size: 10
Search size (k): 100
Search iteration count: 100
Index creation complete!
Index size: 1000000 vectors
Index dimensions:768
Index creation latency: 0.600738 s
P50 Latency: 0.235964s
P90 Latency: 0.266453s
P95 Latency: 0.276967s
P99 Latency: 0.289969s
Mean Latency: 0.23453s
Peak Throughput: 53.4821qps
Mean Throughput: 43.0725qps
Writing results to disk...
Results writing complete!
Write time: 5.751e-06 s
```

### HNSW Flat
```
HNSW Properties:
efConstruction: 200
efSearch: 64
M: 16
Reading dataset from disk...
Read complete!
Index size: 1000000 
Index dimensions:768
Read time: 0.86229 s
Adding vectors to the HNSW index...
Index created!
Index creation latency: 181.369 s
Query read complete!
Query read time: 1.4819e-05 s
Query batch size: 10
Reading baseline results...
Baseline results read complete!
Read time: 2.0429e-05 s
P50 Latency: 0.00048397s
P90 Latency: 0.000988138s
P95 Latency: 0.00147359s
P99 Latency: 0.00240173s
Mean Latency: 0.00059506s
Peak Throughput: 24742.9qps
Mean Throughput: 19449.3qps
Peak recall@100: 100%
Mean recall@100: 90.8%
Results writing complete!
Write time: 9.5493e-05 s
```

### IVF Flat
```
IVF-Flat Properties:
nlist: 2048
nprobe: 36
Reading dataset from disk...
Read complete!
Index size: 1000000 
Index dimensions:768
Read time: 0.863535 s
Training coarse quantizer, adding vectors to the IVF-Flat index...
Index created!
Index creation latency: 21.1479 s
Query read complete!
Query read time: 1.4748e-05 s
Query batch size: 10
Reading baseline results...
Baseline results read complete!
Read time: 6.893e-06 s
P50 Latency: 0.0125564s
P90 Latency: 0.0160017s
P95 Latency: 0.0162424s
P99 Latency: 0.0176888s
Mean Latency: 0.0133619s
Peak Throughput: 814.807qps
Mean Throughput: 756.51qps
Peak recall@100: 100%
Mean recall@100: 98.8%
Results writing complete!
Write time: 7.7438e-05 s
```

### IVF PQ
```IVF-PQ Properties:
nlist: 2048
nprobe: 36
nbits: 8
m: 48
Reading dataset from disk...
Read complete!
Index size: 1000000 
Index dimensions:768
Read time: 0.857843 s
Training coarse quantizer, adding vectors to the IVF-PQ index...
Index created!
Index creation latency: 112.398 s
Query read complete!
Query read time: 1.6531e-05 s
Query batch size: 10
Reading baseline results...
Baseline results read complete!
Read time: 7.484e-06 s
P50 Latency: 0.00112923s
P90 Latency: 0.00149474s
P95 Latency: 0.00330318s
P99 Latency: 0.00656573s
Mean Latency: 0.00136252s
Peak Throughput: 9531.49qps
Mean Throughput: 8340qps
Peak recall@100: 83%
Mean recall@100: 58.7%
Results writing complete!
Write time: 0.000123467 s
```

