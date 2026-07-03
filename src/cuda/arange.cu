#include "arange.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>


__global__ void arangeKernel(
	std::int32_t *values, 
	std::int32_t start, 
	std::size_t len
)
{
	const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i < len)
	{
		values[i] = static_cast<std::int32_t>(i + start);
	}
}


std::vector<std::int32_t> arange(
	std::int32_t start, 
	std::int32_t stop
)
{

	// return empty vector if stop <= start
	if(stop <= start)
	{
		std::vector<std::int32_t> v;
		return v;
	}

	std::size_t numElements     = stop - start;
	std::size_t threadsPerBlock = 256;
	std::size_t numBytes        = numElements * sizeof(std::int32_t);

	std::vector<std::int32_t> v(numElements);

	// Allocate array in memory
	std::int32_t* gpuArray = nullptr;
	cudaMalloc(&gpuArray, numBytes);

	// Copy data in the vector to GPU. Not actually needed since arangeKernel already overwrites every value!
	// cudaMemcpy(gpuArray, v.data(), numBytes, cudaMemcpyHostToDevice);

	// If I have 256 threads per block, add the num of elements to that -1 and divide by the number of threads.
	std::size_t numBlocks = (numElements + threadsPerBlock - 1) / threadsPerBlock;

	arangeKernel<<<numBlocks, 256>>>(gpuArray, start, numElements);

	cudaMemcpy(v.data(), gpuArray, numBytes, cudaMemcpyDeviceToHost);

	cudaFree(gpuArray);

	return v;
}
