#include "crop.hpp"
#include "image_utils.hpp"
#include "cuda_check.hpp"

#include <stdexcept>

/*
CUDA kernel that crops a given image. Each thread indexes the cropped image and finds
the corresponding value in the original image that should be copied.
*/
__global__ void cropKernel(
    stbi_uc *img,
    stbi_uc *croppedImg,
    std::size_t numValuesCropped,
    int cropXStart,
    int cropYStart,
    int channels,
    std::size_t cropWidth,
    std::size_t bytesPerPixel,
    std::size_t bytesPerRow
)
{
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < numValuesCropped)
    {
        const std::size_t currentChannel = i % static_cast<std::size_t>(channels);

        // get the (x,y) coordinates in the cropped image and use them to get the 
        // corresponding position in the original image
        int xCroppedImg = (i / channels) % (cropWidth);
        int yCroppedImg = (i / channels) / (cropWidth);

        const std::size_t posOriginalImg =
                static_cast<std::size_t>(yCroppedImg + cropYStart) * bytesPerRow + 
                static_cast<std::size_t>(xCroppedImg + cropXStart) * bytesPerPixel + 
                currentChannel;

        croppedImg[i] = img[posOriginalImg];
    }
}


void crop(
    stbi_uc *img, 
    int height,
    int width,
    int channels,
    int cropXStart,
    int cropYStart,
    int cropXEnd,
    int cropYEnd
)
{
    const bool validCrop =
        cropXStart >= 0 &&
        cropYStart >= 0 &&
        cropXStart <= cropXEnd &&
        cropYStart <= cropYEnd &&
        cropXEnd < width  &&
        cropYEnd < height;
    
    if (!validCrop) {throw std::out_of_range("Crop rectangle is outside image bounds");}

    const std::size_t bytesPerPixel =
        static_cast<std::size_t>(channels) * sizeof(stbi_uc);

    const std::size_t bytesPerRow =
        static_cast<std::size_t>(width) * bytesPerPixel;

    const std::size_t numValuesImg = 
        static_cast<std::size_t>(height) * 
        static_cast<std::size_t>(width) * 
        static_cast<std::size_t>(channels);

    const std::size_t numBytesImg = 
        numValuesImg * sizeof(stbi_uc);
    
    // allocate bytes for original img and copy it
    stbi_uc* gpuImg = nullptr;
    checkCuda(
        cudaMalloc(
            &gpuImg, 
            numBytesImg
        )
    );
    checkCuda(
        cudaMemcpy(
            gpuImg, 
            img, 
            numBytesImg, 
            cudaMemcpyHostToDevice
        )
    );

    // +1 makes the crop inclusive. 
    // Without it, a crop with (startX, startY)= (0, 0) and (endX, endY) = (0, 0) wouldn't return anything.
    // I want to make it return pixel (0, 0), so that's why I add +1.
    const std::size_t cropHeight =
        static_cast<std::size_t>(cropYEnd) - static_cast<std::size_t>(cropYStart) + 1;
    const std::size_t cropWidth =
        static_cast<std::size_t>(cropXEnd) - static_cast<std::size_t>(cropXStart) + 1;

    const std::size_t numValuesCropped = 
        cropHeight * 
        cropWidth  *
        static_cast<std::size_t>(channels);
    
    const std::size_t numBytesCropped = 
        numValuesCropped * sizeof(stbi_uc);

    // allocate space for the cropped img
    stbi_uc* croppedImg = nullptr;
    checkCuda(
        cudaMalloc(
            &croppedImg, 
            numBytesCropped
        )
    );

    constexpr std::size_t threadsPerBlock = 256;
    const std::size_t numBlocks = 
        (numValuesCropped + threadsPerBlock - 1) / threadsPerBlock;

    // invoke GPU kernel
    cropKernel<<<numBlocks, threadsPerBlock>>>(
        gpuImg, 
        croppedImg, 
        numValuesCropped, 
        cropXStart, 
        cropYStart, 
        channels,
        cropWidth, 
        bytesPerPixel,
        bytesPerRow
    );

    checkCuda(
        cudaMemcpy(
            img, 
            croppedImg, 
            numBytesCropped, 
            cudaMemcpyDeviceToHost
        )
    );

    checkCuda(
        cudaFree(
            gpuImg
        )
    );
    checkCuda(
        cudaFree(
            croppedImg
        )
    );
    
    return;
}