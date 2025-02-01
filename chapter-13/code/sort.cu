#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <cuda_runtime.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <limits.h>

//=====================================================================
// Comment Out Approach 1 Kernels
/*
// --- Approach 1: Single kernel with block-level scan --- //

__device__ void blockScan(unsigned int *data, unsigned int tid, unsigned int blockSize) {
    __shared__ unsigned int temp[1024]; // Adjust size based on your block size
    
    // Load data into shared memory
    temp[tid] = data[tid];
    __syncthreads();
    
    // Upsweep phase
    for (unsigned int stride = 1; stride < blockSize; stride *= 2) {
        unsigned int index = (tid + 1) * 2 * stride - 1;
        if (index < blockSize) {
            temp[index] += temp[index - stride];
        }
        __syncthreads();
    }
    
    // Clear last element
    if (tid == 0) {
        temp[blockSize - 1] = 0;
    }
    __syncthreads();
    
    // Downsweep phase
    for (unsigned int stride = blockSize / 2; stride > 0; stride /= 2) {
        unsigned int index = (tid + 1) * 2 * stride - 1;
        if (index < blockSize) {
            unsigned int t = temp[index];
            temp[index] += temp[index - stride];
            temp[index - stride] = t;
        }
        __syncthreads();
    }
    
    // Write results back
    data[tid] = temp[tid];
    __syncthreads();
}

__global__ void radixSortKernel(unsigned int* input, unsigned int* output,
                               unsigned int* bits, unsigned int N, unsigned int iter) {
    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x + tid;
    unsigned int blockSize = blockDim.x;
    
    // Extract bits
    unsigned int key = 0, bit = 0;
    if (gid < N) {
        key = input[gid];
        bit = (key >> iter) & 1;
        bits[gid] = bit;
    }
    __syncthreads();
    
    // Perform block-level scan
    blockScan(bits + blockIdx.x * blockDim.x, tid, blockSize);
    
    // Calculate final position and write output
    if (gid < N) {
        unsigned int numOnesBefore = bits[gid];
        unsigned int numOnesTotal = bits[blockIdx.x * blockDim.x + blockSize - 1];
        unsigned int dst = (bit == 0) ? (gid - numOnesBefore)
                                      : (N - numOnesTotal + numOnesBefore);
        output[dst] = key;
    }
}
*/
//=====================================================================

// --- Approach 2: Multi-kernel implementation --- //

// Kernel to extract the relevant bit from each key.
__global__ void extractBitsKernel(unsigned int* input, unsigned int* bits,
                                  unsigned int N, unsigned int iter) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < N) {
        unsigned int key = input[gid];
        bits[gid] = (key >> iter) & 1;
    }
}

__global__ void scatterKernel(unsigned int* input, unsigned int* output,
                              unsigned int* scannedBits, unsigned int N, unsigned int iter) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < N) {
        unsigned int key = input[gid];
        unsigned int bit = (key >> iter) & 1;
        unsigned int numOnesBefore = scannedBits[gid];
        unsigned int numOnesTotal = scannedBits[N - 1];
        unsigned int dst = (bit == 0) ? (gid - numOnesBefore)
                                      : (N - numOnesTotal + numOnesBefore);
        output[dst] = key;
    }
}

int compare(const void* a, const void* b) {
    unsigned int ua = *(const unsigned int*) a;
    unsigned int ub = *(const unsigned int*) b;
    return (ua > ub) - (ua < ub);
}

bool isSorted(unsigned int arr[], int size) {
    for (int i = 0; i < size - 1; i++) {
        if (arr[i] > arr[i + 1]) {
            return false;
        }
    }
    return true;
}

void gpuRadixSort(unsigned int *arr, int N) {
    unsigned int *d_input, *d_output, *d_bits;
    
    cudaMalloc((void**)&d_input,  N * sizeof(unsigned int));
    cudaMalloc((void**)&d_output, N * sizeof(unsigned int));
    cudaMalloc((void**)&d_bits,   N * sizeof(unsigned int));
    
    cudaMemcpy(d_input, arr, N * sizeof(unsigned int), cudaMemcpyHostToDevice);
    
    const int threadsPerBlock = 256;
    const int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    const int numBits = 32;
    
    for (unsigned int iter = 0; iter < numBits; iter++) {
        extractBitsKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_bits, N, iter);
        cudaDeviceSynchronize();
        
        thrust::device_ptr<unsigned int> d_bits_ptr(d_bits);
        thrust::exclusive_scan(d_bits_ptr, d_bits_ptr + N, d_bits_ptr);
        cudaDeviceSynchronize();
        
        scatterKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, d_bits, N, iter);
        cudaDeviceSynchronize();
        
        unsigned int* temp = d_input;
        d_input = d_output;
        d_output = temp;
    }
    
    cudaMemcpy(arr, d_input, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_bits);
}


int main() {
    int N = 10;
    unsigned int arr[N];

    // Initialize the array with random unsigned ints.
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        arr[i] = (unsigned int)(rand() % UINT_MAX);
    }

    printf("Original array: ");
    for (int i = 0; i < N; i++) {
        printf("%u ", arr[i]);
    }
    printf("\n");

    // Call the GPU radix sort (using Approach 2).
    gpuRadixSort(arr, N);

    printf("Sorted array: ");
    for (int i = 0; i < N; i++) {
        printf("%u ", arr[i]);
    }
    printf("\n");

    printf("Is sorted array sorted? %s\n", isSorted(arr, N) ? "Yes" : "No");
    return 0;
}
