cmake_minimum_required(VERSION 4.3.4)

project(
    video_cuda
    VERSION 0.1
    LANGUAGES CXX CUDA
)

add_executable(video_cuda
                src/main.cpp
                src/cuda/arange.cu
                src/cuda/brightness.cu
                src/cuda/crop.cu
                src/cuda/cuda_check.cu
                src/image/image_utils.cpp
                src/image/stb_impl.cpp
                )

target_include_directories(video_cuda PRIVATE
                            ${CMAKE_CURRENT_SOURCE_DIR}/include/image
                            ${CMAKE_CURRENT_SOURCE_DIR}/include/ops
                            ${CMAKE_CURRENT_SOURCE_DIR}/include/cuda
                            ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

set_target_properties(video_cuda PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)

target_compile_options(video_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)