cmake_minimum_required(VERSION 3.18)

project(
    binary-matcher
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES 87)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

find_package(CUDAToolkit REQUIRED)

add_executable(binary-matcher
    main.cpp
    binary_matcher.cu
)

target_include_directories(binary-matcher PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_compile_options(binary-matcher PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)

target_link_libraries(binary-matcher PRIVATE
    CUDA::cudart
)