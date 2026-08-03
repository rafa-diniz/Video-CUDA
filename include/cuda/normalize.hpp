#pragma once

#include <cstdint>
#include <vector>
#include "stb/stb_image.h"

std::vector<float> normalize(
    stbi_uc* img, 
    int height,
    int width,
    int channels
);