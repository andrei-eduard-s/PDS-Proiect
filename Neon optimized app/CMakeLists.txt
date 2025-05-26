cmake_minimum_required(VERSION 3.10)

project(FingerprintProcessing VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Detectam arhitectura
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    message(STATUS "Compilare pentru ARM64 cu NEON si OpenMP activat")
    add_compile_options(
        -O3
        -march=armv8-a+simd
        -ftree-vectorize
        -funsafe-math-optimizations
        -fopenmp
    )
    set(OPENMP_LIB_NAMES "gomp")
    set(OPENMP_FOUND TRUE)

    # Find OpenMP
    find_package(OpenMP REQUIRED)
else()
    message(STATUS "Compilare pentru arhitectura non-ARM64 cu optimizari native si OpenMP")
    add_compile_options(-O3 -march=native -fopenmp)
    find_package(OpenMP REQUIRED)
endif()

# Cautam OpenCV
find_package(OpenCV REQUIRED)

add_executable(fingerprint_processing src/main.cpp)

target_include_directories(fingerprint_processing PRIVATE ${OpenCV_INCLUDE_DIRS})

target_link_libraries(fingerprint_processing PRIVATE
    ${OpenCV_LIBS}
    OpenMP::OpenMP_CXX
)

# Setam optiuni specifice targetului (dublu cu -march=armv8-a+simd pentru siguranta)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    target_compile_options(fingerprint_processing PRIVATE
        -march=armv8-a+simd
        -ftree-vectorize
        -funsafe-math-optimizations
    )
endif()
