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
    gpuRadixSortThreeKernels(d_array, N);

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