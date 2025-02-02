#include "gpu_radix_sort.h"
#include <cuda_runtime.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>

__global__ void extractBitsKernel(unsigned int* input, unsigned int* bits,
                                  unsigned int N, unsigned int iter) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        unsigned int key = input[tid];
        bits[tid] = (key >> iter) & 1;
    }
}

__global__ void scatterKernel(unsigned int* input, unsigned int* output,
                              unsigned int* scannedBits, unsigned int N, 
                              unsigned int iter, unsigned int totalOnes) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        unsigned int key = input[tid];
        unsigned int bit = (key >> iter) & 1;
        unsigned int numOnesBefore = scannedBits[tid];
        unsigned int dst;
        if (bit == 0) {
            dst = tid - numOnesBefore;
        } else {
            dst = N - totalOnes + numOnesBefore;
        }
        output[dst] = key;
    }
}

// This function sorts an array stored in device memory using radix sort.
// It assumes the array contains 32-bit unsigned integers.
void gpuRadixSortThreeKenels(unsigned int *d_input, int N) {
    unsigned int *d_output, *d_bits;
    cudaMalloc((void**)&d_output, N * sizeof(unsigned int));
    cudaMalloc((void**)&d_bits,   N * sizeof(unsigned int));
    
    const int threadsPerBlock = 256;
    const int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    const int numBits = 32;
    
    // For each bit position...
    for (unsigned int iter = 0; iter < numBits; iter++) {
        // Extract the current bit.
        extractBitsKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_bits, N, iter);
        cudaDeviceSynchronize();
        
        // Read the original last bit before the scan.
        unsigned int lastBit;
        cudaMemcpy(&lastBit, d_bits + (N - 1), sizeof(unsigned int), cudaMemcpyDeviceToHost);
        
        // Perform an exclusive scan in place on the d_bits array.
        thrust::device_ptr<unsigned int> d_bits_ptr(d_bits);
        thrust::exclusive_scan(d_bits_ptr, d_bits_ptr + N, d_bits_ptr);
        cudaDeviceSynchronize();
        
        // Read back the last value from the scanned array.
        unsigned int scanned_last;
        cudaMemcpy(&scanned_last, d_bits + (N - 1), sizeof(unsigned int), cudaMemcpyDeviceToHost);
        // The total number of ones is the scanned value plus the last bit.
        unsigned int totalOnes = scanned_last + lastBit;
        
        // Scatter the keys into the correct positions using the computed total.
        scatterKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, d_bits, N, iter, totalOnes);
        cudaDeviceSynchronize();
        
        // Swap pointers so that d_input always points to the array with the latest data.
        unsigned int* temp = d_input;
        d_input = d_output;
        d_output = temp;
    }
    
    cudaFree(d_output);
    cudaFree(d_bits);
}
