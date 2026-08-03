#include "normalize.hpp"

#include "stb/stb_image.h"
#include "cuda_check.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <iostream>

#include <cuda_runtime.h>

// TODO stdKernel doesn't actually calculate standard deviations. It's actually doing a sum of squared differences. Move it to a separate C++ function that actually returns the standard deviation.
// TODO The code is pretty messy. See if I can move stuff out of the main normalize() function.
// TODO The mean and standard deviation is done per-image, when it would be more reasonable to do it per-channel.

__global__ void sumKernel(
    stbi_uc* values,
    float* outputSum,
    std::size_t numElements
)
{
    const std::size_t i = 
        blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < numElements)
    {
        atomicAdd(outputSum, values[i]);
    }
}


__global__ void stdKernel(
    stbi_uc* values,
    float* outputStd,
    float mean,
    std::size_t numElements
)
{
    const std::size_t i = 
        blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < numElements)
    {
        float diff = values[i] - mean;
        diff       = diff * diff;
        atomicAdd(outputStd, diff);
    }
}


/*
    Normalizes the values in an image by centering each channel value
    around a global per-image mean.
*/
__global__ void normalizeKernel(
    stbi_uc* img, 
    float* normalizedImg,
    std::size_t numElements,
    float mean,
    float stdDev
    )
{
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < numElements)
    {
        normalizedImg[i] = (static_cast<float>(img[i]) - mean) / stdDev;
    }

}



std::vector<float> normalize(
    stbi_uc* img, 
    int height,
    int width,
    int channels
)
{
    const std::size_t numElements = 
        static_cast<std::size_t>(height) *
        static_cast<std::size_t>(width)  *
        static_cast<std::size_t>(channels);


    const std::size_t threadsPerBlock = 256;
    const std::size_t numBlocks       = (numElements + threadsPerBlock - 1) / threadsPerBlock;


    stbi_uc* gpuImg = nullptr;
    std::size_t imgBytes = numElements * sizeof(stbi_uc);
    checkCuda(
        cudaMalloc(
            &gpuImg,
            imgBytes
        )
    );

    // Copy Img to GPU
    checkCuda(
        cudaMemcpy(
            gpuImg,
            img,
            imgBytes,
            cudaMemcpyHostToDevice
        )
    );

    // Allocate sum on the GPU.
    float sum = 0;
    float* sumGpu;

    checkCuda(
        cudaMalloc(&sumGpu, sizeof(float))
    );

    checkCuda(
        cudaMemset(
            sumGpu,
            0,
            sizeof(float)
        )
    );

    sumKernel<<<numBlocks, threadsPerBlock>>>(
        gpuImg,
        sumGpu,
        numElements
    );


    checkCuda(cudaGetLastError());

    // Move sum result back to CPU
    checkCuda(
        cudaMemcpy(
            &sum,
            sumGpu,
            sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );

    float mean = sum / numElements;

    float stdDev = 0;
    float* stdDevGpu;

    checkCuda(
        cudaMalloc(&stdDevGpu, sizeof(float))
    );

    checkCuda(
        cudaMemset(
            stdDevGpu,
            0,
            sizeof(float)
        )
    );


    stdKernel<<<numBlocks, threadsPerBlock>>>(
        gpuImg,
        stdDevGpu,
        mean,
        numElements
    );

    checkCuda(cudaGetLastError());

    // Move sum result back to CPU
    checkCuda(
        cudaMemcpy(
            &stdDev,
            stdDevGpu,
            sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );

    stdDev = stdDev / numElements;
    stdDev = std::sqrt(stdDev);

    //  Allocate normalized img on the GPU
    float* normalizedImgGpu  = nullptr;
    std::size_t normImgBytes = numElements * sizeof(float);

    checkCuda(
        cudaMalloc(&normalizedImgGpu, normImgBytes)
    );

    if (stdDev == 0.0f)
    {
        checkCuda(
            cudaMemset(
                normalizedImgGpu,
                0,
                normImgBytes
            )
        );
    }
    else
    {    
        normalizeKernel<<<numBlocks, threadsPerBlock>>>(
            gpuImg,
            normalizedImgGpu,
            numElements,
            mean,
            stdDev 
        );
        
        checkCuda(cudaGetLastError());
        checkCuda(cudaDeviceSynchronize());
        
    }



    std::vector<float> normalizedImgCPU;
    normalizedImgCPU.reserve(numElements);
    normalizedImgCPU.resize(numElements);

    checkCuda(
        cudaMemcpy(
            normalizedImgCPU.data(),
            normalizedImgGpu,
            normImgBytes,
            cudaMemcpyDeviceToHost
        )
    );

    // Free VRAM
    checkCuda(cudaFree(sumGpu));
    checkCuda(cudaFree(stdDevGpu));
    checkCuda(cudaFree(normalizedImgGpu));
    checkCuda(cudaFree(gpuImg));

    for(auto i : normalizedImgCPU)
    {
        printf("%f\n", i);
    }

    return normalizedImgCPU;
}