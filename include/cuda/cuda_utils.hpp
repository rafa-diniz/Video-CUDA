#pragma once

#include "stb/stb_image.h"

#include <cstdint>
#include <vector>

void checkCuda(
    cudaError_t error
);



// Allocates memory on the GPU and returns the pointer to that data
template <class T>
T* moveToGPU(
	const T* data,
	std::size_t numElements
)
{
    T* gpuArray       = nullptr;
    std::size_t bytes = numElements * sizeof(T);

    checkCuda(
        cudaMalloc(
            (void**)&gpuArray,
            bytes
        )
    );

    // Copy Array to GPU
    checkCuda(
        cudaMemcpy(
            gpuArray,
            data,
            bytes,
            cudaMemcpyHostToDevice
        )
    );

	return gpuArray;
}


// Allocates a vector<T> on the CPU, copies the data in GPU to it and returns the pointer to the allocated vector
template <class T>
std::vector<T> moveFromGPU(
	const T* data,
	std::size_t numElements
)
{
    std::size_t bytes = numElements * sizeof(T);


    std::vector<T> dataCPU;
    dataCPU.reserve(bytes);
    dataCPU.resize(numElements);
    
    // Copy Array to GPU
    checkCuda(
        cudaMemcpy(
            dataCPU.data(),
            data,
            bytes,
            cudaMemcpyHostToDevice
        )
    );

	return dataCPU;
}