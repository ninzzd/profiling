cmake_minimum_required(VERSION 3.16)

project(RetrievalBenchmarks LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Packages and dependencies
find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# FAISS build variant to link against
set(FAISS_BUILD_VARIANT "generic" CACHE STRING "FAISS build variant to link against (generic, avx2, avx512, dd, gpu, cuda, cuvs)")

# Map variant to the actual FAISS library target and library subdirectory
if(FAISS_BUILD_VARIANT STREQUAL "generic")
    set(FAISS_LIB_TARGET faiss)
    set(FAISS_LIB_SUBDIR faiss)
elseif(FAISS_BUILD_VARIANT STREQUAL "avx2")
    set(FAISS_LIB_TARGET faiss_avx2)
    set(FAISS_LIB_SUBDIR faiss)
elseif(FAISS_BUILD_VARIANT STREQUAL "avx512")
    set(FAISS_LIB_TARGET faiss_avx512)
    set(FAISS_LIB_SUBDIR faiss)
elseif(FAISS_BUILD_VARIANT STREQUAL "dd")
    set(FAISS_LIB_TARGET faiss)
    set(FAISS_LIB_SUBDIR faiss)
elseif(FAISS_BUILD_VARIANT STREQUAL "gpu" OR FAISS_BUILD_VARIANT STREQUAL "cuda")
    set(FAISS_LIB_TARGET faiss)
    set(FAISS_LIB_SUBDIR faiss)
elseif(FAISS_BUILD_VARIANT STREQUAL "cuvs")
    set(FAISS_LIB_TARGET faiss)
    set(FAISS_LIB_SUBDIR faiss)
else()
    message(FATAL_ERROR "Unknown FAISS_BUILD_VARIANT: ${FAISS_BUILD_VARIANT}. Valid values: generic, avx2, avx512, dd, gpu, cuda, cuvs")
endif()

# Common paths
set(FAISS_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/../faiss)
set(FAISS_LIB_DIR    ${CMAKE_SOURCE_DIR}/../faiss-builds/${FAISS_BUILD_VARIANT}/${FAISS_LIB_SUBDIR})
set(BENCH_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)

message(STATUS "FAISS build variant: ${FAISS_BUILD_VARIANT}")
message(STATUS "FAISS library target: ${FAISS_LIB_TARGET}")
message(STATUS "FAISS library directory: ${FAISS_LIB_DIR}")

# Common dependencies
set(COMMON_LIBS
    ${FAISS_LIB_TARGET}
    OpenMP::OpenMP_CXX
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
)

# Common source files shared across executables
set(UTILS_SRC src/utils.cpp)

# Helper function to create a benchmark executable
function(add_benchmark_exec TARGET_NAME SOURCE_FILE)
    add_executable(${TARGET_NAME}
        ${SOURCE_FILE}
        ${ARGN}
    )
    target_include_directories(${TARGET_NAME} PRIVATE
        ${FAISS_INCLUDE_DIR}
        ${BENCH_INCLUDE_DIR}
    )
    target_link_directories(${TARGET_NAME} PRIVATE
        ${FAISS_LIB_DIR}
    )
    target_link_libraries(${TARGET_NAME} PRIVATE
        ${COMMON_LIBS}
    )
endfunction()

# --- Benchmarks ---

add_benchmark_exec(toy_cpu_flat
    src/toy-cpu-flat.cpp
)

add_benchmark_exec(wikiall_cpu_baseline
    src/wikiall-cpu-baseline.cpp
    ${UTILS_SRC}
)

add_benchmark_exec(wikiall_cpu_ivf_flat
    src/wikiall-cpu-ivf-flat.cpp
    ${UTILS_SRC}
)

add_benchmark_exec(wikiall_cpu_hnsw
    src/wikiall-cpu-hnsw.cpp
    ${UTILS_SRC}
)

add_benchmark_exec(wikiall_cpu_ivf_pq
    src/wikiall-cpu-ivf-pq.cpp
    ${UTILS_SRC}
)

add_benchmark_exec(idxgen_baseline
    src/idxgen-baseline.cpp
)

add_benchmark_exec(idxgen_hnsw
    src/idxgen-hnsw.cpp
)

add_benchmark_exec(idxgen_ivf_pq
    src/idxgen-ivf-pq.cpp
)

add_benchmark_exec(idxgen_ivf_flat
    src/idxgen-ivf-flat.cpp
)

add_benchmark_exec(query_gen
    src/query-gen.cpp
)

