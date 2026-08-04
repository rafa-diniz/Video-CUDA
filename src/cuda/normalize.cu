#include "normalize.hpp"

#include "stb/stb_image.h"
#include "math_utils.hpp"
#include "cuda_utils.hpp"


#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

// TODO The mean and standard deviation is calculated per-image, when it would be more reasonable to do it per-channel.


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


    stbi_uc* gpuImg = moveToGPU(
        img,
        numElements
    );
    
    double mean = meanArray(
        gpuImg,
        numElements,
        threadsPerBlock,
        numBlocks
    );

    double stdDev = stdDeviation(
        gpuImg,
        mean,
        numElements,
        threadsPerBlock,
        numBlocks
    );

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
        
        checkCuda(cudaDeviceSynchronize());
        checkCuda(cudaGetLastError());
        
    }


    // Move img back to the CPU
    std::vector<float> normalizedImgCPU = moveFromGPU(
        normalizedImgGpu,
        numElements
    );
   
    // Free VRAM
    checkCuda(cudaFree(normalizedImgGpu));
    checkCuda(cudaFree(gpuImg));

    for(auto i : normalizedImgCPU)
    {
        printf("%f\n", i);
    }

    return normalizedImgCPU;
}