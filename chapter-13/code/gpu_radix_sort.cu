#include "gpu_radix_sort.h"
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
// Three-kernel implementation
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
        unsigned int dst = (bit == 0) ? tid - numOnesBefore : 
                                      N - totalOnes + numOnesBefore;
        output[dst] = key;
    }
}

void gpuRadixSortThreeKernels(unsigned int *d_input, int N) {
    unsigned int *d_output, *d_bits;
    CUDA_CHECK(cudaMalloc((void**)&d_output, N * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc((void**)&d_bits, N * sizeof(unsigned int)));

    const int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (unsigned int iter = 0; iter < NUM_BITS; iter++) {
        extractBitsKernel<<<numBlocks, BLOCK_SIZE>>>(d_input, d_bits, N, iter);
        CUDA_CHECK(cudaDeviceSynchronize());

        unsigned int lastBit;
        CUDA_CHECK(cudaMemcpy(&lastBit, d_bits + (N - 1), sizeof(unsigned int), 
                             cudaMemcpyDeviceToHost));

        thrust::device_ptr<unsigned int> d_bits_ptr(d_bits);
        thrust::exclusive_scan(d_bits_ptr, d_bits_ptr + N, d_bits_ptr);
        CUDA_CHECK(cudaDeviceSynchronize());

        unsigned int scanned_last;
        CUDA_CHECK(cudaMemcpy(&scanned_last, d_bits + (N - 1), sizeof(unsigned int), 
                             cudaMemcpyDeviceToHost));

        unsigned int totalOnes = scanned_last + lastBit;

        scatterKernel<<<numBlocks, BLOCK_SIZE>>>(d_input, d_output, d_bits, 
                                                N, iter, totalOnes);
        CUDA_CHECK(cudaDeviceSynchronize());

        unsigned int* temp = d_input;
        d_input = d_output;
        d_output = temp;
    }

    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_bits));
}

// Single kernel implementation
__device__ void hierarchical_kogge_stone_scan(float* X, float* scan_value, 
                                            int* flags, unsigned int N) {
    extern __shared__ float buffer[];
    __shared__ float previous_sum;
    const unsigned int tid = threadIdx.x;
    const unsigned int bid = blockIdx.x;
    const unsigned int gid = bid * blockDim.x + tid;

    if (gid < N) {
        buffer[tid] = X[gid];
    } else {
        buffer[tid] = 0.0f;
    }
    __syncthreads();

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        float temp = buffer[tid];
        if (tid >= stride) {
            temp += buffer[tid - stride];
        }
        __syncthreads();
        buffer[tid] = temp;
        __syncthreads();
    }

    float exclusive_value = (tid == 0) ? 0.0f : buffer[tid - 1];
    const float local_sum = buffer[blockDim.x - 1];

    if (tid == 0) {
        if (bid > 0) {
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

    if (gid < N) {
        X[gid] = exclusive_value + (bid > 0 ? previous_sum : 0.0f);
    }
}

__global__ void radix_sort_iter(unsigned int* input, unsigned int* output,
                               float* bits_float, float* scan_value, int* flags,
                               unsigned int N, unsigned int iter) {
    const unsigned int tid = threadIdx.x;
    const unsigned int bid = blockIdx.x;
    const unsigned int i = bid * blockDim.x + tid;
    
    if (i < N) {
        unsigned int key = input[i];
        bits_float[i] = (float)((key >> iter) & 1);
    }
    __syncthreads();
    
    __shared__ bool isLastBlock;
    if (tid == 0) {
        isLastBlock = (bid == ((N + blockDim.x - 1) / blockDim.x) - 1);
    }
    __syncthreads();
    
    if (isLastBlock && tid == ((N - 1) % blockDim.x)) {
        bits_float[N] = 0;
        __threadfence();
        atomicAdd(&flags[gridDim.x], 1);
    }
    
    hierarchical_kogge_stone_scan(bits_float, scan_value, flags, N);
    
    if (isLastBlock && tid == ((N - 1) % blockDim.x)) {
        bits_float[N] = bits_float[N - 1] + ((input[N - 1] >> iter) & 1);
        __threadfence();
        atomicAdd(&flags[gridDim.x + 1], 1);
    }
    
    if (tid == 0) {
        while (atomicAdd(&flags[gridDim.x + 1], 0) == 0) { }
    }
    __syncthreads();
    
    if (i < N) {
        unsigned int key = input[i];
        unsigned int bit = (key >> iter) & 1;
        float numOnesBefore = bits_float[i];
        float numOnesTotal = bits_float[N];
        
        unsigned int dst = (bit == 0) ? 
            i - (unsigned int)numOnesBefore : 
            N - (unsigned int)numOnesTotal + (unsigned int)numOnesBefore;
        
        output[dst] = key;
    }
}

void gpuRadixSortSingleKernel(unsigned int *d_input, int N) {
    assert(N <= MAX_INPUT_SIZE && 
           "Input size above limit leads to potential deadlock due to grid-level synchronization issues.");

    unsigned int *d_output;
    float *d_bits_float, *d_scan_value;
    int *d_flags;
    const int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    CUDA_CHECK(cudaMalloc((void**)&d_output, N * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc((void**)&d_bits_float, (N + 1) * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_scan_value, (numBlocks + 1) * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_flags, (numBlocks + 3) * sizeof(int)));

    for (unsigned int iter = 0; iter < NUM_BITS; iter++) {
        CUDA_CHECK(cudaMemset(d_flags, 0, (numBlocks + 3) * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_scan_value, 0, (numBlocks + 1) * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_bits_float + N, 0, sizeof(float)));
        
        radix_sort_iter<<<numBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(
            d_input, d_output, d_bits_float, d_scan_value, d_flags, N, iter);
        CUDA_CHECK(cudaDeviceSynchronize());

        unsigned int *temp = d_input;
        d_input = d_output;
        d_output = temp;
    }

    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_bits_float));
    CUDA_CHECK(cudaFree(d_scan_value));
    CUDA_CHECK(cudaFree(d_flags));
}