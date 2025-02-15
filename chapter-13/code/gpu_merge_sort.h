#ifndef GPU_MERGE_SORT_H
#define GPU_MERGE_SORT_H

#include <cuda_runtime.h>

// Use the same block size as in your other kernels.
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024
#endif

// Declaration for GPU merge sort on unsigned int arrays.
void gpuMergeSortUnsignedInt(unsigned int* d_input, int N);

#endif // GPU_MERGE_SORT_H
