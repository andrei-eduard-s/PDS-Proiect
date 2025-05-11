cmake_minimum_required(VERSION 3.10)

project(FingerprintProcessing VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# NEON and optimization flags
add_compile_options(-O3 -march=native)

find_package(OpenCV REQUIRED)

add_executable(fingerprint_processing src/main.cpp)

target_link_libraries(fingerprint_processing PRIVATE ${OpenCV_LIBS})
target_link_libraries(fingerprint_processing PRIVATE stdc++fs)

target_include_directories(fingerprint_processing PRIVATE ${OpenCV_INCLUDE_DIRS})