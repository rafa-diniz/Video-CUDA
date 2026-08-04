#include "math_utils.hpp"
#include "image_utils.hpp"
#include "cuda_utils.hpp"


#include <cuda_runtime.h>


__global__ void sumKernel(
    stbi_uc* values,
    unsigned long long int* outputSum,
    std::size_t numElements
)
{
    const std::size_t i = 
        blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < numElements)
    {
        atomicAdd(
            outputSum, 
            static_cast<unsigned long long int>(values[i])
        );
    }
}


__global__ void sumOfSquaredDifferencesKernel(
    stbi_uc* values,
    double* outputStd,
    double mean,
    std::size_t numElements
)
{
    const std::size_t i = 
        blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < numElements)
    {
        double diff = static_cast<double>(values[i]) - mean;

        diff        = diff * diff;
        atomicAdd(outputStd, diff);
    }
}



std::uint64_t sumArray(
    stbi_uc* values,
    std::size_t numElements,
    std::size_t threadsPerBlock,
    std::size_t numBlocks
)
{
    // Allocate variables for the sum. 
    // Since it's a learning project I'm using a 64-bit uint, but it would be nice to see how this behaves on edge cases like very large images.
    unsigned long long int  sum = 0;
    unsigned long long int* sumGpu;

    checkCuda(
        cudaMalloc(&sumGpu, sizeof(unsigned long long int))
    );

    checkCuda(
        cudaMemset(
            sumGpu,
            0,
            sizeof(unsigned long long int)
        )
    );

    sumKernel<<<numBlocks, threadsPerBlock>>>(
        values,
        sumGpu,
        numElements
    );

    checkCuda(cudaGetLastError());

    // Move sum result back to CPU
    checkCuda(
        cudaMemcpy(
            &sum,
            sumGpu,
            sizeof(unsigned long long int),
            cudaMemcpyDeviceToHost
        )
    );

    checkCuda(cudaFree(sumGpu));

    return static_cast<std::uint64_t>(sum);
}


double stdDeviation(
    stbi_uc* values,
    double mean,
    std::size_t numElements,
    std::size_t threadsPerBlock,
    std::size_t numBlocks
)
{
    double stdDev;
    double* stdDevGpu;

    checkCuda(
        cudaMalloc(&stdDevGpu, sizeof(double))
    );

    checkCuda(
        cudaMemset(
            stdDevGpu,
            0,
            sizeof(double)
        )
    );


    sumOfSquaredDifferencesKernel<<<numBlocks, threadsPerBlock>>>(
        values,
        stdDevGpu,
        mean,
        numElements
    );

    checkCuda(cudaGetLastError());

    // Move result back to CPU
    checkCuda(
        cudaMemcpy(
            &stdDev,
            stdDevGpu,
            sizeof(double),
            cudaMemcpyDeviceToHost
        )
    );

    // Free VRAM
    checkCuda(cudaFree(stdDevGpu));

    stdDev = stdDev / numElements;
    stdDev = std::sqrt(stdDev);

    return stdDev;
}


double meanArray(
    stbi_uc* values,
    std::size_t numElements,
    std::size_t threadsPerBlock,
    std::size_t numBlocks
)
{
    std::uint64_t sum = sumArray(
        values,
        numElements,
        threadsPerBlock,
        numBlocks
    );

    double mean = static_cast<double>(sum) / static_cast<double>(numElements);

    return mean;
}