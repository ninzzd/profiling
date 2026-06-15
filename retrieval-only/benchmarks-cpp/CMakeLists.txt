cmake_minimum_required(VERSION 3.16)

project(RetrievalBenchmarks LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Packages and dependencies
find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# Add your benchmark executable
add_executable(simple_cpu_debug
    simple-cpu-debug.cpp
)

# FAISS headers (source tree)
target_include_directories(simple_cpu_debug PRIVATE
    ${CMAKE_SOURCE_DIR}/../faiss
)

# Path to the built FAISS library
target_link_directories(simple_cpu_debug PRIVATE
    ${CMAKE_SOURCE_DIR}/../faiss-builds/cpu-debug-cpp/faiss
)

# Link against FAISS
target_link_libraries(simple_cpu_debug PRIVATE
    faiss
    OpenMP::OpenMP_CXX
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
)