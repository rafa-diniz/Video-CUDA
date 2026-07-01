#include "image_utils.hpp"

#include <cstdio>

void printImg(stbi_uc* img, int height, int width, int channels)
{
    std::printf(
        "Width: %d\nHeight: %d\nChannels: %d\n",
        width,
        height,
        channels
    );

	for(int y = 0; y < height; y++)
	{
		for(int x = 0; x < width; x++)
		{
			const int base = (y * width + x) * channels;
			
			std::printf(
				"[%u %u %u]",
				static_cast<unsigned>(img[base + 0]),
				static_cast<unsigned>(img[base + 1]),
				static_cast<unsigned>(img[base + 2])
			);
		}
        std::printf("\n");
	}
}