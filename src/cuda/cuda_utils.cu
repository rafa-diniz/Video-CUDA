#include "cuda_utils.hpp"

#include <stdexcept>
#include <cuda_runtime.h>


void checkCuda(cudaError_t error)
{
    if(error == cudaSuccess){ return ; }
    
    throw std::runtime_error(cudaGetErrorString(error));
}