#pragma once

#include <cstdint>
#include "stb/stb_image.h"

void adjustBrightness(
    stbi_uc* img, 
    int height,
    int width,
    int channels,
    std::int16_t brightnessChange
);