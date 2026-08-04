#pragma once

#include "stb/stb_image.h"

#include <cstdint>


double stdDeviation(
    stbi_uc* values,
    double mean,
    std::size_t numElements,
    std::size_t threadsPerBlock,
    std::size_t numBlocks
);

std::uint64_t sumArray(
    stbi_uc* values,
    std::size_t numElements,
    std::size_t threadsPerBlock,
    std::size_t numBlocks
);

double meanArray(
    stbi_uc* values,
    std::size_t numElements,
    std::size_t threadsPerBlock,
    std::size_t numBlocks
);