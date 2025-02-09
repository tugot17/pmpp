#ifndef GPU_RADIX_SORT_H
#define GPU_RADIX_SORT_H

void gpuRadixSortThreeKernels(unsigned int *d_input, int N);
void gpuRadixSortSingleKernel(unsigned int *d_input, int N);
void gpuRadixSortSingleKernelGridSync(unsigned int* d_input, int N);

#endif