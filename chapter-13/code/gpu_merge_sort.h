#ifndef CHAPTER_13_CODE_GPU_MERGE_SORT_H_
#define CHAPTER_13_CODE_GPU_MERGE_SORT_H_

#include <cuda_runtime.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024
#endif

void gpuMergeSort(unsigned int* d_input, int N);

#endif  // CHAPTER_13_CODE_GPU_MERGE_SORT_H_
