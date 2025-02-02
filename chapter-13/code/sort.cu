// nvcc sort.cu -o sort

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>
#include <cuda_runtime.h>
#include <limits.h>
#include "gpu_radix_sort.h"

int compare_uint(const void* a, const void* b) {
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

__device__ void hierarchical_kogge_stone_domino_device(
    float* X, float* Y, float* scan_value, int* flags, int* blockCounter,
    float* buffer, unsigned int N, unsigned int tid, unsigned int blockDim_x) {
    
    __shared__ unsigned int bid_s;
    __shared__ float previous_sum;

    // DEADLOCK PREVENTION: Dynamic block index assignment
    if (tid == 0) {
        bid_s = atomicAdd(blockCounter, 1);
    }
    __syncthreads();

    const unsigned int bid = bid_s;
    const unsigned int gid = bid * blockDim_x + tid;

    // Phase 1: Load data and perform exclusive scan using Kogge-Stone
    if (gid < N) {
        buffer[tid] = X[gid];
    } else {
        buffer[tid] = 0.0f;
    }
    __syncthreads();

    // Save original value for later use
    float original = buffer[tid];
    
    // Exclusive scan within block
    for (unsigned int stride = 1; stride < blockDim_x; stride *= 2) {
        __syncthreads();
        float temp = buffer[tid];
        if (tid >= stride) {
            temp += buffer[tid - stride];
        }
        __syncthreads();
        buffer[tid] = temp;
    }

    // Shift right by one to make it exclusive
    __syncthreads();
    float exclusive_sum;
    if (tid == 0) {
        exclusive_sum = 0;
    } else {
        exclusive_sum = buffer[tid - 1];
    }
    
    // Store local result
    if (gid < N) {
        Y[gid] = exclusive_sum;
    }

    // Get local sum for this block (inclusive sum of all elements)
    const float local_sum = buffer[blockDim_x - 1];

    // Phase 2: Inter-block sum propagation
    if (tid == 0) {
        if (bid > 0) {
            // Wait for previous block's flag
            while (atomicAdd(&flags[bid], 0) == 0) {
            }

            // Get sum from previous block
            previous_sum = scan_value[bid];

            // Add local sum and propagate
            const float total_sum = previous_sum + local_sum;
            scan_value[bid + 1] = total_sum;

            // Ensure scan_value is visible
            __threadfence();

            // Signal next block
            atomicAdd(&flags[bid + 1], 1);
        } else {
            // First block just propagates its sum
            scan_value[1] = local_sum;
            __threadfence();
            atomicAdd(&flags[1], 1);
        }
    }
    __syncthreads();

    // Phase 3: Add previous block's sum to local results
    if (bid > 0 && gid < N) {
        Y[gid] += previous_sum;
    }
}

__global__ void radix_sort_iter(unsigned int* input, unsigned int* output, unsigned int* bits, float* bits_in, float* bits_scanned,
                              float* scan_value, int* flags, int* blockCounter,
                              unsigned int N, unsigned int iter) {
    extern __shared__ float buffer[];
    const unsigned int tid = threadIdx.x;
    const unsigned int gid = blockIdx.x * blockDim.x + tid;
    
    // Extract bits into input array for scan
    if(gid < N) {
        unsigned int key = input[gid];
        bits_in[gid] = (float)((key >> iter) & 1);
    }
    __syncthreads();

    // Perform hierarchical exclusive scan
    hierarchical_kogge_stone_domino_device(bits_in, bits_scanned, scan_value, flags, blockCounter,
                                         buffer, N, tid, blockDim.x);
    __syncthreads();

    // Use scan results to reorder elements
    if(gid < N) {
        unsigned int key = input[gid];
        unsigned int bit = (key >> iter) & 1;
        float numOnesBefore = bits_scanned[gid];
        
        // Calculate total number of ones (exclusive scan of last element + last element's value)
        float numOnesTotal = 0.0f;
        if (gid == N-1) {
            numOnesTotal = bits_scanned[N-1] + bits_in[N-1];
            // Store this value for other threads
            scan_value[0] = numOnesTotal;
        }
        __syncthreads();
        
        numOnesTotal = scan_value[0];
        
        // Calculate destination index
        unsigned int dst;
        if(bit == 0) {
            dst = gid - (unsigned int)numOnesBefore;
        } else {
            dst = (N - (unsigned int)numOnesTotal) + (unsigned int)numOnesBefore;
        }
        
        if(dst < N) {  // Safety check
            output[dst] = key;
        }
    }
}

void gpuRadixSort(unsigned int *d_input, int N) {
    unsigned int *d_output, *d_bits;
    float *d_bits_in, *d_bits_scanned, *d_scan_value;
    int *d_flags, *d_blockCounter;
    const int threadsPerBlock = 256;
    const int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate device memory
    cudaMalloc((void**)&d_output, N * sizeof(unsigned int));
    cudaMalloc((void**)&d_bits, N * sizeof(unsigned int));
    cudaMalloc((void**)&d_bits_in, N * sizeof(float));
    cudaMalloc((void**)&d_bits_scanned, N * sizeof(float));
    cudaMalloc((void**)&d_scan_value, (numBlocks + 1) * sizeof(float));
    cudaMalloc((void**)&d_flags, (numBlocks + 1) * sizeof(int));
    cudaMalloc((void**)&d_blockCounter, sizeof(int));

    // For each bit position (32-bit integers)
    for (unsigned int iter = 0; iter < 32; iter++) {
        // Reset synchronization arrays
        cudaMemset(d_flags, 0, (numBlocks + 1) * sizeof(int));
        cudaMemset(d_blockCounter, 0, sizeof(int));
        cudaMemset(d_scan_value, 0, (numBlocks + 1) * sizeof(float));
        
        radix_sort_iter<<<numBlocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>
            (d_input, d_output, d_bits, d_bits_in, d_bits_scanned,
             d_scan_value, d_flags, d_blockCounter, N, iter);
        
        cudaDeviceSynchronize();

        // Swap input and output pointers
        unsigned int *temp = d_input;
        d_input = d_output;
        d_output = temp;
    }

    // Free device memory
    cudaFree(d_output);
    cudaFree(d_bits);
    cudaFree(d_bits_in);
    cudaFree(d_bits_scanned);
    cudaFree(d_scan_value);
    cudaFree(d_flags);
    cudaFree(d_blockCounter);
}

int main() {
    // int N = 1 << 20; // For example, 1M elements
    int N = 100;
    unsigned int* h_unsorted = (unsigned int*)malloc(N * sizeof(unsigned int));
    if (!h_unsorted) {
        fprintf(stderr, "Failed to allocate host array.\n");
        return EXIT_FAILURE;
    }
    srand((unsigned)time(NULL));
    for (int i = 0; i < N; i++) {
        h_unsorted[i] = rand();
        // h_unsorted[i] = 1;
    }
    
    //init d_array
    unsigned int* d_array;
    cudaMalloc(&d_array, N * sizeof(unsigned int));
    cudaMemcpy(d_array, h_unsorted, N * sizeof(unsigned int), cudaMemcpyHostToDevice);

    gpuRadixSort(d_array, N);

    unsigned int* h_sorted = (unsigned int*)malloc(N * sizeof(unsigned int));
    cudaMemcpy(h_sorted, d_array, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    if (isSorted(h_sorted, N)) {
        printf("GPU sorted array is correct.\n");
    } else {
        printf("GPU sorted array is NOT sorted correctly!\n");
    }

    for (unsigned int i = 0; i < N; i++){
        printf("%d, ", h_sorted[i]);
    }
    printf("\n");
    
    free(h_sorted);
    cudaFree(d_array);
    free(h_unsorted);

    return 0;
}
