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
## Benchmarks

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