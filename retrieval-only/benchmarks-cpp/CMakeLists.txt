cmake_minimum_required(VERSION 3.16)

project(RetrievalBenchmarks LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Packages and dependencies
find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# Add your benchmark executable
add_executable(toy_cpu_flat
    toy-cpu-flat.cpp
)

add_executable(wikiall_cpu_flat
    wikiall-cpu-flat.cpp
)

# FAISS headers (source tree)
target_include_directories(toy_cpu_flat PRIVATE
    ${CMAKE_SOURCE_DIR}/../faiss
)
target_include_directories(wikiall_cpu_flat PRIVATE
    ${CMAKE_SOURCE_DIR}/../faiss
)

# Path to the built FAISS library
target_link_directories(toy_cpu_flat PRIVATE
    ${CMAKE_SOURCE_DIR}/../faiss-builds/cpu-debug-cpp/faiss
)
target_link_directories(wikiall_cpu_flat PRIVATE
    ${CMAKE_SOURCE_DIR}/../faiss-builds/cpu-debug-cpp/faiss
)

# Link against FAISS
target_link_libraries(toy_cpu_flat PRIVATE
    faiss
    OpenMP::OpenMP_CXX
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
)
target_link_libraries(wikiall_cpu_flat PRIVATE
    faiss
    OpenMP::OpenMP_CXX
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
)