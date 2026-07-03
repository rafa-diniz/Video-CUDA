#include "cuda_check.hpp"

#include <stdexcept>

void checkCuda(cudaError_t error)
{
    if(error == cudaSuccess){ return ; }
    
    throw std::runtime_error(cudaGetErrorString(error));
}