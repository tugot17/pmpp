#include "gpu_radix_sort.h"
#include <cuda_runtime.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>

// Kernel to extract the bit of interest from each element
__global__ void extractBitsKernel(unsigned int* input, unsigned int* bits,
                                  unsigned int N, unsigned int iter) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        unsigned int key = input[tid];
        bits[tid] = (key >> iter) & 1;
    }
}

// Kernel to scatter the keys into the correct position based on the scanned bits.
__global__ void scatterKernel(unsigned int* input, unsigned int* output,
                              unsigned int* scannedBits, unsigned int N, unsigned int iter) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        unsigned int key = input[tid];
        unsigned int bit = (key >> iter) & 1;
        unsigned int numOnesBefore = scannedBits[tid];
        unsigned int numOnesTotal = scannedBits[N - 1];
        unsigned int dst = (bit == 0) ? (tid - numOnesBefore)
                                      : (N - numOnesTotal + numOnesBefore);
        output[dst] = key;
    }
}

// This function sorts an array stored in device memory.
void gpuRadixSortDevice(unsigned int *d_input, int N) {
    unsigned int *d_output, *d_bits;
    
    // Allocate temporary device memory
    cudaMalloc((void**)&d_output, N * sizeof(unsigned int));
    cudaMalloc((void**)&d_bits,   N * sizeof(unsigned int));
    
    const int threadsPerBlock = 256;
    const int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    const int numBits = 32;
    
    // Perform one pass per bit position
    for (unsigned int iter = 0; iter < numBits; iter++) {
        extractBitsKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_bits, N, iter);
        cudaDeviceSynchronize();
        
        // Use Thrust to perform an exclusive scan on the bits array
        thrust::device_ptr<unsigned int> d_bits_ptr(d_bits);
        thrust::exclusive_scan(d_bits_ptr, d_bits_ptr + N, d_bits_ptr);
        cudaDeviceSynchronize();
        
        scatterKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, d_bits, N, iter);
        cudaDeviceSynchronize();
        
        // Swap pointers so that d_input points to the latest sorted array.
        unsigned int* temp = d_input;
        d_input = d_output;
        d_output = temp;
    }
    
    // Free the temporary buffers. The sorted data is left in d_input.
    cudaFree(d_output);
    cudaFree(d_bits);
}
