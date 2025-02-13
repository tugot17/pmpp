#ifndef GPU_RADIX_SORT_H
#define GPU_RADIX_SORT_H

#include <cuda_runtime.h>

// Common configuration
#define BLOCK_SIZE 1024
#define NUM_BITS 32
#define MAX_INPUT_SIZE 100000

// Unified error checking
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(err); \
    } \
}

// Function declarations
void gpuRadixSortThreeKernels(unsigned int *d_input, int N);
void gpuRadixSortSingleKernel(unsigned int *d_input, int N);
void gpuRadixSortWithMemoryCoalescing(unsigned int *d_input, int N);

#endif // GPU_RADIX_SORT_H