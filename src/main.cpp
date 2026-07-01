#include <cstdio>
#include <cstdlib>
#include <cstdint>

#include "arange.hpp"
#include "brightness.hpp"
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

	std::printf("Adjusting brightness: \n\n");
	
	adjustBrightness(
		img,
		height,
		width,
		outputChannels, 
		static_cast<std::int16_t>(-10)
	);

	printImg(img, height, width, outputChannels);

	stbi_image_free(img);

	return EXIT_SUCCESS;
}
