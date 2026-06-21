#include <cstdio>
#include <cstdlib>
#include <cstdint>

#include <cuda_runtime.h>

#include <vector>


__global__ void arange(std::int32_t *values, std::int32_t start, std::size_t len)
{
	std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i < len)
	{
		values[i] = static_cast<std::int32_t>(i + start);
	}
}

std::vector<std::int32_t> arange(std::int32_t start, std::int32_t stop)
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

	for(auto i : v)
	{
		printf("%d ", i);
	}
	printf("\n");

	// Allocate array in memory
	std::int32_t* gpuArray = nullptr;
	cudaMalloc(&gpuArray, numBytes);

	// Copy data in the vector to GPU. Not actually needed since __global__ void arange already overwrites every value!
	// cudaMemcpy(gpuArray, v.data(), numBytes, cudaMemcpyHostToDevice);

	// If I have 256 threads per block, add the num of elements to that -1 and divide by the number of threads.
	std::size_t numBlocks = (numElements + threadsPerBlock - 1) / threadsPerBlock;

	arange<<<numBlocks, 256>>>(gpuArray, start, numElements);

	cudaMemcpy(v.data(), gpuArray, numBytes, cudaMemcpyDeviceToHost);

	cudaFree(gpuArray);

	return v;
}

int main()
{
	std::vector<std::int32_t> myVec = arange(10, 1400'000'000);


	for(auto i : myVec)
	{
		printf("%d ", i);
	}
	printf("\n");


	return EXIT_SUCCESS;
}
