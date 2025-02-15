#ifndef GPU_RADIX_SORT_H
#define GPU_RADIX_SORT_H

#include <cuda_runtime.h>

// Common configuration
#define BLOCK_SIZE 1024
#define NUM_BITS 32
#define MAX_INPUT_SIZE 100000
#define RADIX 4
#define COARSE_FACTOR 4

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
void gpuRadixSortCoalescedMultibitRadix(unsigned int *d_input, int N, unsigned int r);
void gpuRadixSortCoalescedMultibitRadixThreadCoarsening(unsigned int *d_input, int N, unsigned int r);
void gpuMergeSort(float* d_input, int N);

#endif // GPU_RADIX_SORT_H