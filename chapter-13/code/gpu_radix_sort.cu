#include "gpu_radix_sort.h"
#include <cuda_runtime.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <stdio.h>
#include <stdlib.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define THREADS_PER_BLOCK 256
#define MAX_BLOCKS 32
#define NUM_BITS 32

#define SECTION_SIZE 1024
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(err); \
    } \
}

#define cudaCheckError() { \
    cudaError_t err = cudaGetLastError(); \
    if(err != cudaSuccess) { \
        printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

// Three-kernel implementation (unchanged)
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

void gpuRadixSortThreeKernels(unsigned int *d_input, int N) {
    unsigned int *d_output, *d_bits;
    CHECK_CUDA(cudaMalloc((void**)&d_output, N * sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc((void**)&d_bits, N * sizeof(unsigned int)));

    const int threadsPerBlock = 256;
    const int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    const int numBits = 32;

    for (unsigned int iter = 0; iter < numBits; iter++) {
        extractBitsKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_bits, N, iter);
        CHECK_CUDA(cudaDeviceSynchronize());

        unsigned int lastBit;
        CHECK_CUDA(cudaMemcpy(&lastBit, d_bits + (N - 1), sizeof(unsigned int), cudaMemcpyDeviceToHost));

        thrust::device_ptr<unsigned int> d_bits_ptr(d_bits);
        thrust::exclusive_scan(d_bits_ptr, d_bits_ptr + N, d_bits_ptr);
        CHECK_CUDA(cudaDeviceSynchronize());

        unsigned int scanned_last;
        CHECK_CUDA(cudaMemcpy(&scanned_last, d_bits + (N - 1), sizeof(unsigned int), cudaMemcpyDeviceToHost));

        unsigned int totalOnes = scanned_last + lastBit;

        scatterKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, d_bits, N, iter, totalOnes);
        CHECK_CUDA(cudaDeviceSynchronize());

        unsigned int* temp = d_input;
        d_input = d_output;
        d_output = temp;
    }

    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_bits));
}

__device__ void hierarchical_kogge_stone_domino_exclusive_inplace(float* X,
    float* scan_value, int* flags, unsigned int N)
{
    extern __shared__ float buffer[];
    __shared__ float previous_sum;
    const unsigned int tid = threadIdx.x;
    const unsigned int bid = blockIdx.x;
    const unsigned int gid = bid * blockDim.x + tid;

    // Phase 1: Load input (or 0 if past end) into shared memory.
    if (gid < N) {
        buffer[tid] = X[gid];
    } else {
        buffer[tid] = 0.0f;
    }
    __syncthreads();

    // Phase 1a: Inclusive scan within the block (Kogge-Stone algorithm)
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        float temp = buffer[tid];
        if (tid >= stride) {
            temp += buffer[tid - stride];
        }
        __syncthreads();
        buffer[tid] = temp;
        __syncthreads();
    }

    // Phase 1b: Convert inclusive scan to exclusive scan.
    float exclusive_value = (tid == 0) ? 0.0f : buffer[tid - 1];

    // The total (inclusive) for this block.
    const float local_sum = buffer[blockDim.x - 1];

    // Phase 2: Propagate inter-block sums in grid order.
    if (tid == 0) {
        if (bid > 0) {
            // Wait until the previous block has written its partial sum.
            while (atomicAdd(&flags[bid], 0) == 0) { }
            previous_sum = scan_value[bid];
            scan_value[bid + 1] = previous_sum + local_sum;
            __threadfence();
            atomicAdd(&flags[bid + 1], 1);
        } else {
            scan_value[1] = local_sum;
            __threadfence();
            atomicAdd(&flags[1], 1);
        }
    }
    __syncthreads();

    // Phase 3: Write final exclusive scan results to global memory.
    if (gid < N) {
        if (bid > 0) {
            X[gid] = exclusive_value + previous_sum;
        } else {
            X[gid] = exclusive_value;
        }
    }
}

__global__ void radix_sort_iter(unsigned int* input, unsigned int* output,
    float* bits_float, float* scan_value, int* flags,
    unsigned int N, unsigned int iter)
{
    const unsigned int tid = threadIdx.x;
    const unsigned int bid = blockIdx.x;
    const unsigned int i = bid * blockDim.x + tid;
    
    // Initialize bits_float with the current bit.
    if (i < N) {
        unsigned int key = input[i];
        unsigned int bit = (key >> iter) & 1;
        bits_float[i] = (float)bit;
    }
    __syncthreads();
    
    // Determine whether this block is the last active block.
    __shared__ bool isLastBlock;
    if (tid == 0) {
        isLastBlock = (bid == ((N + blockDim.x - 1) / blockDim.x) - 1);
    }
    __syncthreads();
    
    // Only the last thread in the last block initializes the total count slot.
    if (isLastBlock && tid == ((N - 1) % blockDim.x)) {
        bits_float[N] = 0;
        __threadfence();  // Ensure write visibility.
        atomicAdd(&flags[gridDim.x], 1);  // Signal initialization is done.
    }
    
    // Perform the hierarchical scan over bits_float.
    hierarchical_kogge_stone_domino_exclusive_inplace(bits_float, scan_value, flags, N);
    
    // The last thread in the last block computes the total count of ones.
    if (isLastBlock && tid == ((N - 1) % blockDim.x)) {
        bits_float[N] = bits_float[N - 1] + ((input[N - 1] >> iter) & 1);
        __threadfence();
        atomicAdd(&flags[gridDim.x + 1], 1);  // Signal that total is ready.
    }
    
    // All threads wait until the total count is available.
    if (tid == 0) {
        while (atomicAdd(&flags[gridDim.x + 1], 0) == 0) { }
    }
    __syncthreads();
    
    // Use the scanned values to compute destination index and write out the key.
    if (i < N) {
        unsigned int key = input[i];
        unsigned int bit = (key >> iter) & 1;
        float numOnesBefore = bits_float[i];
        float numOnesTotal  = bits_float[N];
        
        unsigned int dst;
        if (bit == 0) {
            dst = i - (unsigned int)numOnesBefore;
        } else {
            dst = N - (unsigned int)numOnesTotal + (unsigned int)numOnesBefore;
        }
        output[dst] = key;
    }
}

void gpuRadixSortSingleKernel(unsigned int *d_input, int N) {
    assert(N <= 100000 && "Input size above 100k leads to potential deadlock due to grid-level synchronization issues.");

    unsigned int *d_output;
    float *d_bits_float, *d_scan_value;
    int *d_flags;
    const int threadsPerBlock = SECTION_SIZE;
    const int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate device memory.
    cudaMalloc((void**)&d_output, N * sizeof(unsigned int)); cudaCheckError();
    cudaMalloc((void**)&d_bits_float, (N + 1) * sizeof(float)); cudaCheckError();  // +1 for total count
    cudaMalloc((void**)&d_scan_value, (numBlocks + 1) * sizeof(float)); cudaCheckError();
    // Allocate extra flags (numBlocks + 3 ints)
    cudaMalloc((void**)&d_flags, (numBlocks + 3) * sizeof(int)); cudaCheckError();

    // Process all 32 bits.
    for (unsigned int iter = 0; iter < 32; iter++) {
        // Reset synchronization arrays for this iteration.
        cudaMemset(d_flags, 0, (numBlocks + 3) * sizeof(int)); cudaCheckError();
        cudaMemset(d_scan_value, 0, (numBlocks + 1) * sizeof(float)); cudaCheckError();
        cudaMemset(d_bits_float + N, 0, sizeof(float)); cudaCheckError();  // Clear the total count slot.
        
        // Launch the kernel.
        // Allocate shared memory of size threadsPerBlock*sizeof(float)
        radix_sort_iter<<<numBlocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
            d_input, d_output, d_bits_float, d_scan_value, d_flags, N, iter);
        cudaCheckError();
        cudaDeviceSynchronize(); cudaCheckError();

        // Swap pointers for the next iteration.
        unsigned int *temp = d_input;
        d_input = d_output;
        d_output = temp;
    }

    // Free temporary device memory.
    cudaFree(d_output); cudaCheckError();
    cudaFree(d_bits_float); cudaCheckError();
    cudaFree(d_scan_value); cudaCheckError();
    cudaFree(d_flags); cudaCheckError();
}
