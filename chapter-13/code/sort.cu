// nvcc sort.cu -o sort

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>
#include <cuda_runtime.h>
#include <limits.h>

#define SECTION_SIZE 256

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

#define cudaCheckError() { \
    cudaError_t err = cudaGetLastError(); \
    if(err != cudaSuccess) { \
        printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

// Custom implementation
__device__ void hierarchical_kogge_stone_domino_exclusive_inplace(float* X, float* scan_value, int* flags, int* blockCounter, unsigned int N) {
    extern __shared__ float buffer[];
    __shared__ unsigned int bid_s;
    __shared__ float previous_sum;
    const unsigned int tid = threadIdx.x;

    if (tid == 0) {
        bid_s = atomicAdd(blockCounter, 1);
    }
    __syncthreads();
    const unsigned int bid = bid_s;
    const unsigned int gid = bid * blockDim.x + tid;

    // Phase 1: Load into shared memory
    if (gid < N) {
        buffer[tid] = X[gid];
    } else {
        buffer[tid] = 0.0f;
    }
    __syncthreads();

    // Kogge-Stone scan within block
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        float temp = buffer[tid];
        if (tid >= stride) {
            temp += buffer[tid - stride];
        }
        __syncthreads();
        buffer[tid] = temp;
    }

    // Convert to exclusive scan
    float exclusive_value;
    if (tid == 0) {
        exclusive_value = 0.0f;
    } else {
        exclusive_value = buffer[tid - 1];
    }

    // Store block's total sum before modifying shared memory
    const float local_sum = buffer[blockDim.x - 1];
    
    // Phase 2: Inter-block sum propagation
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

    // Phase 3: Write final result
    if (gid < N) {
        if (bid > 0) {
            X[gid] = exclusive_value + previous_sum;
        } else {
            X[gid] = exclusive_value;
        }
    }
}

__global__ void radix_sort_iter(unsigned int* input, unsigned int* output,
    float* bits_float, float* scan_value, int* flags, int* blockCounter,
    unsigned int N, unsigned int iter) {
    const unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    
    // Initialize bits array
    if(i < N) {
        unsigned int key = input[i];
        unsigned int bit = (key >> iter) & 1;
        bits_float[i] = (float)bit;
    }
    if(i == N-1) {
        // Store the total number of ones at position N
        bits_float[N] = 0;
    }

    hierarchical_kogge_stone_domino_exclusive_inplace(bits_float, scan_value, flags, blockCounter, N);

    // After scan, compute total number of ones
    if(i == N-1) {
        bits_float[N] = bits_float[i] + ((input[i] >> iter) & 1);
    }
    __syncthreads();

    if(i < N) {
        unsigned int key = input[i];
        unsigned int bit = (key >> iter) & 1;
        float numOnesBefore = bits_float[i];
        float numOnesTotal = bits_float[N];
        
        unsigned int dst;
        if(bit == 0) {
            dst = i - (unsigned int)numOnesBefore;
        } else {
            dst = N - (unsigned int)numOnesTotal + (unsigned int)numOnesBefore;
        }
        output[dst] = key;
    }
}

void gpuRadixSort(unsigned int *d_input, int N) {
    unsigned int *d_output;
    float *d_bits_float, *d_scan_value;
    int *d_flags, *d_blockCounter;
    const int threadsPerBlock = SECTION_SIZE;
    const int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate device memory
    cudaMalloc((void**)&d_output, N * sizeof(unsigned int)); cudaCheckError();
    cudaMalloc((void**)&d_bits_float, (N + 1) * sizeof(float)); cudaCheckError();  // +1 for total count
    cudaMalloc((void**)&d_scan_value, (numBlocks + 1) * sizeof(float)); cudaCheckError();
    cudaMalloc((void**)&d_flags, (numBlocks + 1) * sizeof(int)); cudaCheckError();
    cudaMalloc((void**)&d_blockCounter, sizeof(int)); cudaCheckError();

    // For each bit position (32-bit integers)
    for (unsigned int iter = 0; iter < 32; iter++) {
        // Reset synchronization arrays for each iteration
        cudaMemset(d_flags, 0, (numBlocks + 1) * sizeof(int)); cudaCheckError();
        cudaMemset(d_blockCounter, 0, sizeof(int)); cudaCheckError();
        cudaMemset(d_scan_value, 0, (numBlocks + 1) * sizeof(float)); cudaCheckError();
        cudaMemset(d_bits_float + N, 0, sizeof(float)); cudaCheckError();  // Clear the total count position
        
        radix_sort_iter<<<numBlocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>
            (d_input, d_output, d_bits_float, d_scan_value, d_flags, d_blockCounter, N, iter);
        cudaCheckError();
        cudaDeviceSynchronize(); cudaCheckError();

        // Swap input and output pointers
        unsigned int *temp = d_input;
        d_input = d_output;
        d_output = temp;
    }

    // Free device memory
    cudaFree(d_output); cudaCheckError();
    cudaFree(d_bits_float); cudaCheckError();
    cudaFree(d_scan_value); cudaCheckError();
    cudaFree(d_flags); cudaCheckError();
    cudaFree(d_blockCounter); cudaCheckError();
}

int main() {
    // Ensure N is multiple of SECTION_SIZE
    int N = 10;  // Start with one block
    
    unsigned int* h_unsorted = (unsigned int*)malloc(N * sizeof(unsigned int));
    if (!h_unsorted) {
        fprintf(stderr, "Failed to allocate host array.\n");
        return EXIT_FAILURE;
    }
    
    srand((unsigned)time(NULL));
    for (int i = 0; i < N; i++) {
        h_unsorted[i] = rand() % 4;
    }

    printf("Input array:\n");
    for (unsigned int i = 0; i < N; i++){
        printf("%d, ", h_unsorted[i]);
    }
    printf("\n");
    
    unsigned int* d_array;
    cudaMalloc(&d_array, N * sizeof(unsigned int)); cudaCheckError();
    cudaMemcpy(d_array, h_unsorted, N * sizeof(unsigned int), cudaMemcpyHostToDevice); cudaCheckError();

    gpuRadixSort(d_array, N);

    unsigned int* h_sorted = (unsigned int*)malloc(N * sizeof(unsigned int));
    cudaMemcpy(h_sorted, d_array, N * sizeof(unsigned int), cudaMemcpyDeviceToHost); cudaCheckError();

    if (isSorted(h_sorted, N)) {
        printf("GPU sorted array is correct.\n");
    } else {
        printf("GPU sorted array is NOT sorted correctly!\n");
    }

    printf("Output array:\n");
    for (unsigned int i = 0; i < N; i++){
        printf("%d, ", h_sorted[i]);
    }
    printf("\n");
    
    free(h_sorted);
    cudaFree(d_array); cudaCheckError();
    free(h_unsorted);

    return 0;
}