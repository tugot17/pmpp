// nvcc sort.cu -o sort

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>
#include <cuda_runtime.h>
#include <limits.h>

#define SECTION_SIZE 1024

int compare_uint(const void* a, const void* b) {
    unsigned int ua = *(const unsigned int*) a;
    unsigned int ub = *(const unsigned int*) b;
    return (ua > ub) - (ua < ub);
}

bool compareArrays(unsigned int* arr1, unsigned int* arr2, int size) {
    for (int i = 0; i < size; i++) {
        if (arr1[i] != arr2[i]) {
            printf("Mismatch at index %d: GPU=%u, CPU=%u\n", i, arr1[i], arr2[i]);
            return false;
        }
    }
    return true;
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

/*
 * hierarchical_kogge_stone_domino_exclusive_inplace:
 *
 * This device function performs a per-block scan (using a Kogge-Stone scheme)
 * and then propagates the block sums in natural (blockIdx.x) order.
 * Note that we removed the atomic block counter and now use blockIdx.x consistently.
 *
 * Shared memory (buffer) is allocated with a size of (blockDim.x * sizeof(float)).
 */
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

/*
 * radix_sort_iter:
 *
 * This kernel performs one iteration (for one bit position) of the radix sort.
 * It extracts a bit, uses the scan to compute destination indices for stable reordering,
 * and then writes out the keys to the appropriate locations.
 *
 * Note that we now use blockIdx.x consistently (both for the scan and for
 * determining the "last block").
 */
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

/*
 * gpuRadixSort:
 *
 * This function allocates device memory, launches 32 iterations (one for each bit)
 * of the sort, and then cleans up.
 *
 * Note: We no longer need an extra counter variable because we rely on blockIdx.x.
 */
void gpuRadixSort(unsigned int *d_input, int N) {
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

int main() {
    // Set N (you can try larger arrays such as 100000 or more to test).
    int N = 100000;
    
    unsigned int* h_unsorted = (unsigned int*)malloc(N * sizeof(unsigned int));
    unsigned int* h_quicksort = (unsigned int*)malloc(N * sizeof(unsigned int));
    
    if (!h_unsorted || !h_quicksort) {
        fprintf(stderr, "Failed to allocate host arrays.\n");
        return EXIT_FAILURE;
    }
    
    // Fill arrays with random values.
    srand((unsigned)time(NULL));
    for (int i = 0; i < N; i++) {
        h_unsorted[i] = rand();
        h_quicksort[i] = h_unsorted[i];  // Copy for CPU sorting.
    }
    
    // Sort with quicksort (CPU) for later comparison.
    qsort(h_quicksort, N, sizeof(unsigned int), compare_uint);
    
    // Allocate and copy data to device.
    unsigned int* d_array;
    cudaMalloc(&d_array, N * sizeof(unsigned int)); cudaCheckError();
    cudaMemcpy(d_array, h_unsorted, N * sizeof(unsigned int), cudaMemcpyHostToDevice); cudaCheckError();

    // Run the GPU radix sort.
    gpuRadixSort(d_array, N);

    unsigned int* h_sorted = (unsigned int*)malloc(N * sizeof(unsigned int));
    cudaMemcpy(h_sorted, d_array, N * sizeof(unsigned int), cudaMemcpyDeviceToHost); cudaCheckError();

    // Verify that the array is sorted.
    if (isSorted(h_sorted, N)) {
        printf("GPU sorted array is correctly sorted.\n");
    } else {
        printf("GPU sorted array is NOT sorted correctly!\n");
    }

    // Compare with the CPU quicksort result.
    if (compareArrays(h_sorted, h_quicksort, N)) {
        printf("GPU radix sort matches CPU quicksort results.\n");
    } else {
        printf("GPU radix sort produces different results from CPU quicksort!\n");
        printf("\nFirst 10 elements comparison:\n");
        printf("Index\tGPU\tCPU\n");
        for (int i = 0; i < 10 && i < N; i++) {
            printf("%d\t%u\t%u\n", i, h_sorted[i], h_quicksort[i]);
        }
    }
    
    // Cleanup.
    free(h_sorted);
    free(h_quicksort);
    cudaFree(d_array); cudaCheckError();
    free(h_unsorted);

    return 0;
}
