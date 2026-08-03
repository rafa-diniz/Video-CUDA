#pragma once

#include <cstdint>
#include "stb/stb_image.h"

void crop(
    stbi_uc* img, 
    int height,
    int width,
    int channels,
    int cropX,
    int cropY,
    int cropWidth,
    int cropHeight
);