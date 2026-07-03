#include "crop.hpp"
#include "image_utils.hpp"

#include <stdexcept>


__global__ void cropImgKernel(
    stbi_uc *img,
    stbi_uc *croppedImg,
    std::size_t numValuesImg,
    int cropXStart,
    int cropYStart,
    int cropXEnd,
    int cropYEnd,
    int width,
    int channels,
    std::size_t bytesPerPixel,
    std::size_t bytesPerRow,
    std::size_t bytesPerRowCropped
)
{
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < numValuesImg)
    {
        int x = (i / channels) % width;
        int y = (i / channels) / width;
        
        // check if x and y are inside the crop region
        if( (x >= cropXStart) && 
            (x <= cropXEnd) && 
            (y >= cropYStart) && 
            (y <= cropYEnd)
        )
        {
            const std::size_t currentChannel = i % static_cast<std::size_t>(channels);

            // calculate corresponding position in the original and cropped images
            const std::size_t pos = 
                static_cast<std::size_t>(y) * bytesPerRow + 
                static_cast<std::size_t>(x) * bytesPerPixel + 
                currentChannel;

            const std::size_t posCropped = 
                static_cast<std::size_t>(y-cropYStart) * bytesPerRowCropped + 
                static_cast<std::size_t>(x-cropXStart) * bytesPerPixel + 
                currentChannel;

            croppedImg[posCropped] = img[pos];
        }
    }
}


void cropImg(
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
    cudaMalloc(&gpuImg, numBytesImg);
    cudaMemcpy(gpuImg, img, numBytesImg, cudaMemcpyHostToDevice);

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

    const std::size_t bytesPerRowCropped =
        cropWidth * bytesPerPixel;

    // allocate space for the cropped img
    stbi_uc* croppedImg = nullptr;
    cudaMalloc(&croppedImg, numBytesCropped);

    constexpr std::size_t threadsPerBlock = 256;
    const std::size_t numBlocks = 
        (numValuesImg + threadsPerBlock - 1) / threadsPerBlock;

    // invoke GPU kernel
    cropImgKernel<<<numBlocks, threadsPerBlock>>>(
        gpuImg, 
        croppedImg, 
        numValuesImg, 
        cropXStart, 
        cropYStart, 
        cropXEnd, 
        cropYEnd, 
        width, 
        channels,
        bytesPerPixel,
        bytesPerRow,
        bytesPerRowCropped
    );

    cudaMemcpy(img, croppedImg, numBytesCropped, cudaMemcpyDeviceToHost);

    cudaFree(gpuImg);
    cudaFree(croppedImg);
    
    return;
}