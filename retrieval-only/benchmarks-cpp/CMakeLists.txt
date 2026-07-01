cmake_minimum_required(VERSION 3.16)

project(RetrievalBenchmarks LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Packages and dependencies
find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# toy-cpu-flat
add_executable(toy_cpu_flat
    src/toy-cpu-flat.cpp
)
target_include_directories(toy_cpu_flat PRIVATE
    ${CMAKE_SOURCE_DIR}/../faiss
)
target_link_directories(toy_cpu_flat PRIVATE
    ${CMAKE_SOURCE_DIR}/../faiss-builds/cpu-debug-cpp/faiss
)
target_link_libraries(toy_cpu_flat PRIVATE
    faiss
    OpenMP::OpenMP_CXX
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
)

# wikiall-cpu-baseline
add_executable(wikiall_cpu_baseline
    src/wikiall-cpu-baseline.cpp
)
target_include_directories(wikiall_cpu_baseline PRIVATE
    ${CMAKE_SOURCE_DIR}/../faiss
)
target_link_directories(wikiall_cpu_baseline PRIVATE
    ${CMAKE_SOURCE_DIR}/../faiss-builds/cpu-debug-cpp/faiss
)
target_link_libraries(wikiall_cpu_baseline PRIVATE
    faiss
    OpenMP::OpenMP_CXX
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
)

# wikiall-cpu-ivf-flat
add_executable(wikiall_cpu_ivf_flat
    src/wikiall-cpu-ivf-flat.cpp
)
target_include_directories(wikiall_cpu_ivf_flat PRIVATE
    ${CMAKE_SOURCE_DIR}/../faiss
)
target_link_directories(wikiall_cpu_ivf_flat PRIVATE
    ${CMAKE_SOURCE_DIR}/../faiss-builds/cpu-debug-cpp/faiss
)
target_link_libraries(wikiall_cpu_ivf_flat PRIVATE
    faiss
    OpenMP::OpenMP_CXX
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
)

# wikiall-cpu-hnsw
add_executable(wikiall_cpu_hnsw
    src/wikiall-cpu-hnsw.cpp
)
target_include_directories(wikiall_cpu_hnsw PRIVATE
    ${CMAKE_SOURCE_DIR}/../faiss
)
target_link_directories(wikiall_cpu_hnsw PRIVATE
    ${CMAKE_SOURCE_DIR}/../faiss-builds/cpu-debug-cpp/faiss
)
target_link_libraries(wikiall_cpu_hnsw PRIVATE
    faiss
    OpenMP::OpenMP_CXX
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
)

# wikiall-cpu-ivf-pq
add_executable(wikiall_cpu_ivf_pq
    src/wikiall-cpu-ivf-pq.cpp
)
target_include_directories(wikiall_cpu_ivf_pq PRIVATE
    ${CMAKE_SOURCE_DIR}/../faiss
)
target_link_directories(wikiall_cpu_ivf_pq PRIVATE
    ${CMAKE_SOURCE_DIR}/../faiss-builds/cpu-debug-cpp/faiss
)
target_link_libraries(wikiall_cpu_ivf_pq PRIVATE
    faiss
    OpenMP::OpenMP_CXX
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
)

# idxgen-baseline
add_executable(idxgen_baseline
    src/idxgen-baseline.cpp
)
target_include_directories(idxgen_baseline PRIVATE
    ${CMAKE_SOURCE_DIR}/../faiss
)
target_link_directories(idxgen_baseline PRIVATE
    ${CMAKE_SOURCE_DIR}/../faiss-builds/cpu-debug-cpp/faiss
)
target_link_libraries(idxgen_baseline PRIVATE
    faiss
    OpenMP::OpenMP_CXX
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
)

# idxgen-hnsw
add_executable(idxgen_hnsw
    src/idxgen-hnsw.cpp
)
target_include_directories(idxgen_hnsw PRIVATE
    ${CMAKE_SOURCE_DIR}/../faiss
)
target_link_directories(idxgen_hnsw PRIVATE
    ${CMAKE_SOURCE_DIR}/../faiss-builds/cpu-debug-cpp/faiss
)
target_link_libraries(idxgen_hnsw PRIVATE
    faiss
    OpenMP::OpenMP_CXX
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
)

# idxgen-ivf-pq
add_executable(idxgen_ivf_pq
    src/idxgen-ivf-pq.cpp
)
target_include_directories(idxgen_ivf_pq PRIVATE
    ${CMAKE_SOURCE_DIR}/../faiss
)
target_link_directories(idxgen_ivf_pq PRIVATE
    ${CMAKE_SOURCE_DIR}/../faiss-builds/cpu-debug-cpp/faiss
)
target_link_libraries(idxgen_ivf_pq PRIVATE
    faiss
    OpenMP::OpenMP_CXX
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
)

# idxgen-ivf-flat
add_executable(idxgen_ivf_flat
    src/idxgen-ivf-flat.cpp
)
target_include_directories(idxgen_ivf_flat PRIVATE
    ${CMAKE_SOURCE_DIR}/../faiss
)
target_link_directories(idxgen_ivf_flat PRIVATE
    ${CMAKE_SOURCE_DIR}/../faiss-builds/cpu-debug-cpp/faiss
)
target_link_libraries(idxgen_ivf_flat PRIVATE
    faiss
    OpenMP::OpenMP_CXX
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
)

# query_gen
add_executable(query_gen
    src/query-gen.cpp
)
target_include_directories(query_gen PRIVATE
    ${CMAKE_SOURCE_DIR}/../faiss
)
target_link_directories(query_gen PRIVATE
    ${CMAKE_SOURCE_DIR}/../faiss-builds/cpu-debug-cpp/faiss
)
target_link_libraries(query_gen PRIVATE
    faiss
    OpenMP::OpenMP_CXX
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
)