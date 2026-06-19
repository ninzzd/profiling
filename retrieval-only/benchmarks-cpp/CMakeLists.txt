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
    toy-cpu-flat.cpp
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

# wikiall-cpu-flat
add_executable(wikiall_cpu_flat
    wikiall-cpu-flat.cpp
)
target_include_directories(wikiall_cpu_flat PRIVATE
    ${CMAKE_SOURCE_DIR}/../faiss
)
target_link_directories(wikiall_cpu_flat PRIVATE
    ${CMAKE_SOURCE_DIR}/../faiss-builds/cpu-debug-cpp/faiss
)
target_link_libraries(wikiall_cpu_flat PRIVATE
    faiss
    OpenMP::OpenMP_CXX
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
)

# wikiall-cpu-ivf-flat
add_executable(wikiall_cpu_ivf_flat
    wikiall-cpu-ivf-flat.cpp
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