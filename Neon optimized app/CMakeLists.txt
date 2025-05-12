cmake_minimum_required(VERSION 3.10)

project(FingerprintProcessing VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Verificam daca sistemul este pe arhitectura ARM64 (Raspberry Pi 4B cu Raspberry Pi OS pe 64-bit)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    # Activam optimizarile NEON pentru ARM64
    add_compile_options(-O3 -march=armv8-a+simd -ftree-vectorize -funsafe-math-optimizations)

    # Mesaj informativ pentru debugging
    message(STATUS "Compilare pentru ARM64 cu NEON activat")
else()
    # Daca nu este ARM64, foloseste doar optimizari generale pentru alte platforme
    add_compile_options(-O3 -march=native)
endif()

# Cautam OpenCV
find_package(OpenCV REQUIRED)

# Cream executabilul
add_executable(fingerprint_processing src/main.cpp)

# Le legam pe OpenCV si altele
target_link_libraries(fingerprint_processing PRIVATE ${OpenCV_LIBS})
target_link_libraries(fingerprint_processing PRIVATE stdc++fs)

# Setam caile de includere pentru OpenCV
target_include_directories(fingerprint_processing PRIVATE ${OpenCV_INCLUDE_DIRS})

# Daca esti pe ARM64, se asigura ca NEON este folosit corect
target_compile_options(fingerprint_processing PRIVATE -march=armv8-a+simd)
