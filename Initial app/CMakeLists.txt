cmake_minimum_required(VERSION 3.10)
project(FingerprintPlain)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Gaseste OpenCV
find_package(OpenCV REQUIRED)

# Adauga sursa din subdirectorul `src`
add_executable(FingerprintPlain src/main.cpp)

# Leaga OpenCV
target_link_libraries(FingerprintPlain ${OpenCV_LIBS})
