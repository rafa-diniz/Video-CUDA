#include <cstdio>
#include <cstdlib>
#include <cstdint>

#include "arange.hpp"
#include "brightness.hpp"
#include "crop.hpp"
#include "image_utils.hpp"

#include "stb/stb_image.h"
#include "stb/stb_image_write.h"



int main()
{	
	int width    = 0;
	int height   = 0;
	int channels = 0;
	constexpr int outputChannels = 3;

	stbi_uc* img = stbi_load(
		"assets/rgb.png", &width, &height, &channels, outputChannels
	);


	printImg(img, height, width, outputChannels);

	std::printf("Cropping Img: \n\n");

	int cropXStart = 0;
	int cropYStart = 1;
	int cropXEnd   = 2;
	int cropYEnd   = 1;

    // Clip crop end to correct limits
    if(cropXEnd >= width)  {cropXEnd = width  - 1;}
    if(cropYEnd >= height) {cropYEnd = height - 1;}

	cropImg(
		img, 
		height, 
		width, 
		channels, 
		cropXStart, 
		cropYStart, 
		cropXEnd, 
		cropYEnd
	);


	// +1 to make the crop inclusive
	int cropWidth  = cropXEnd - cropXStart + 1; 
	int cropHeight = cropYEnd - cropYStart + 1;
	printImg(
		img, 
		cropHeight, 
		cropWidth, 
		outputChannels
	);

	stbi_image_free(img);

	return EXIT_SUCCESS;
}
