#include <cuda_runtime.h>
#include <stdio.h>

#include <algorithm>

#include "gpu_merge_sort.h"

// Define min and max if not already defined.
#ifndef min
#define min(a, b) ((a) < (b) ? (a) : (b))
#endif
#ifndef max
#define max(a, b) ((a) > (b) ? (a) : (b))
#endif

// If you already have a CUDA_CHECK macro defined elsewhere, you can remove or adjust this.
#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                                            \
    {                                                                                               \
        cudaError_t err = call;                                                                     \
        if (err != cudaSuccess) {                                                                   \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(err);                                                                              \
        }                                                                                           \
    }
#endif

// Utility: ceiling division.
__host__ __device__ int cdiv(int a, int b) {
    return (a + b - 1) / b;
}

// co_rank determines the split point between two sorted arrays A and B.
// (Here, we work on unsigned ints instead of floats.)
__host__ __device__ int co_rank(int k, unsigned int* A, int m, unsigned int* B, int n) {
    int i = min(k, m);
    int j = k - i;

    int i_low = max(0, k - n);
    int j_low = max(0, k - m);
    int delta;

    bool active = true;
    while (active) {
        // If A[i-1] is too big, decrease i.
        if (i > 0 && j < n && A[i - 1] > B[j]) {
            delta = cdiv(i - i_low, 2);
            j_low = j;
            i -= delta;
            j += delta;
        } else if (j > 0 && i < m && B[j - 1] >= A[i]) {  // If B[j-1] is too big, increase i.
            delta = cdiv(j - j_low, 2);
            i_low = i;
            i += delta;
            j -= delta;
        } else {
            active = false;
        }
    }
    return i;
}

// A sequential merge that merges two sorted segments.
__host__ __device__ void merge_sequential(unsigned int* A, int m, unsigned int* B, int n, unsigned int* C) {
    int i = 0, j = 0, k = 0;
    while (i < m && j < n) {
        if (A[i] <= B[j]) {
            C[k++] = A[i++];
        } else {
            C[k++] = B[j++];
        }
    }
    while (i < m) {
        C[k++] = A[i++];
    }
    while (j < n) {
        C[k++] = B[j++];
    }
}

// Each block merges a pair of sorted subarrays.
__global__ void merge_pass_kernel(unsigned int* d_in, unsigned int* d_out, int N, int width) {
    int pair = blockIdx.x;
    int start = pair * (2 * width);
    if (start >= N) {
        return;
    }

    int mid = min(start + width, N);
    int end = min(start + 2 * width, N);
    int lenA = mid - start;
    int lenB = end - mid;

    unsigned int* A = d_in + start;
    unsigned int* B = d_in + mid;
    unsigned int* C = d_out + start;

    int total = lenA + lenB;

    int tid = threadIdx.x;
    int numThreads = blockDim.x;
    int elementsPerThread = cdiv(total, numThreads);

    int k_start = tid * elementsPerThread;
    int k_end = min((tid + 1) * elementsPerThread, total);

    int i_start = co_rank(k_start, A, lenA, B, lenB);
    int j_start = k_start - i_start;
    int i_end = co_rank(k_end, A, lenA, B, lenB);
    int j_end = k_end - i_end;

    merge_sequential(A + i_start, i_end - i_start, B + j_start, j_end - j_start, C + k_start);
}

// The main GPU merge sort routine.
void gpuMergeSort(unsigned int* d_input, int N) {
    unsigned int* d_output;
    CUDA_CHECK(cudaMalloc((void**)&d_output, N * sizeof(unsigned int)));

    int width = 1;
    int numPasses = 0;

    while (width < N) {
        int numMerges = (N + 2 * width - 1) / (2 * width);

        merge_pass_kernel<<<numMerges, BLOCK_SIZE>>>(d_input, d_output, N, width);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Swap pointers between d_input and d_output.
        unsigned int* temp = d_input;
        d_input = d_output;
        d_output = temp;

        width *= 2;
        numPasses++;
    }

    // If an odd number of passes was executed, copy the result back.
    if (numPasses % 2 == 1) {
        CUDA_CHECK(cudaMemcpy(d_input, d_output, N * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
    }

    CUDA_CHECK(cudaFree(d_output));
}
