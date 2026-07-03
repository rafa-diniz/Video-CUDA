#include "brightness.hpp"
#include "stb/stb_image.h"

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

/*
CUDA kernel that adjusts brightness by dispatching one thread
per image channel value.
*/
__global__ void adjustBrightnessKernel(
    stbi_uc* img, 
    std::size_t pitch, 
    int height,
    int width,
    int channels,
    std::size_t numElements,
    std::int16_t brightnessChange
)
{
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i < numElements)
	{
        const int x = (i / channels) % width;
        const int y = (i / channels) / width;
        
        // The array in VRAM has valid bytes with a bunch of padding. Each row begins `pitch` bytes after the previous row, 
        // but a bunch of it is just padding. To discard the padding, the formula below uses the x and y values to correctly 
        // position the index over the valid image data.
        const std::size_t pos = (y * pitch) + (x * channels) + (i % channels);
        
        // Because simply overwriting can cause overflow/underflow since the values are uint8, 
        // the new value is first cast to int16 and then clipped so it is between 0 and 255.
        std::int16_t newValue = static_cast<std::int16_t>(img[pos]) + brightnessChange;
        if(newValue < 0)
        {
            newValue = 0;
        }
        else if(newValue > 255)
        {
            newValue = 255;
        }
        
        // Safely overwrite the data in the image with the clipped value
        img[pos] = static_cast<stbi_uc>(newValue);
    }
}


void adjustBrightness(
    stbi_uc* img, 
    int height,
    int width,
    int channels,
    std::int16_t brightnessChange
)
{
    const std::size_t numElements =
        static_cast<std::size_t>(height) *
        static_cast<std::size_t>(width) *
        static_cast<std::size_t>(channels);

    const std::size_t rowBytes =
        static_cast<std::size_t>(width) *
        static_cast<std::size_t>(channels) *
        sizeof(stbi_uc);


    const std::size_t threadsPerBlock = 256;
    const std::size_t numBlocks       = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    stbi_uc* gpuImg = nullptr;
    std::size_t pitch;
    
    // Allocate img in VRAM
    cudaMallocPitch(
        &gpuImg,
        &pitch,
        rowBytes,
        height
    );

    // Copy img to GPU
    cudaMemcpy2D(
        gpuImg, 
        pitch, 
        img, 
        rowBytes, 
        rowBytes, 
        height, 
        cudaMemcpyHostToDevice
    );
    
    // Dispatch GPU Call
    adjustBrightnessKernel<<<numBlocks, threadsPerBlock>>>(
        gpuImg, 
        pitch, 
        height, 
        width, 
        channels,
        numElements, 
        brightnessChange
    );

    // Copy back to CPU
    cudaMemcpy2D(
        img, 
        rowBytes, 
        gpuImg, 
        pitch, 
        rowBytes, 
        height, 
        cudaMemcpyDeviceToHost
    );

    cudaFree(gpuImg);

	return;
}