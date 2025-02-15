#include <assert.h>
#include <cuda_runtime.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <time.h>

#define BLOCK_SIZE 256
#define MAX_INPUT_SIZE 100000
#define NUM_BITS 32

#define CUDA_CHECK(call)                                                                            \
    do {                                                                                            \
        cudaError_t err = call;                                                                     \
        if (err != cudaSuccess) {                                                                   \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(err);                                                                              \
        }                                                                                           \
    } while (0)

__device__ void hierarchical_kogge_stone_scan(float* X, float* scan_value, int* flags, unsigned int N) {
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
            while (atomicAdd(&flags[bid], 0) == 0) {
            }
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

__global__ void radix_sort_iter(unsigned int* input, unsigned int* output, float* bits_float, float* scan_value,
                                int* flags, unsigned int N, unsigned int iter) {
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
        while (atomicAdd(&flags[gridDim.x + 1], 0) == 0) {
        }
    }
    __syncthreads();

    if (i < N) {
        unsigned int key = input[i];
        unsigned int bit = (key >> iter) & 1;
        float numOnesBefore = bits_float[i];
        float numOnesTotal = bits_float[N];

        unsigned int dst =
            (bit == 0) ? i - (unsigned int)numOnesBefore : N - (unsigned int)numOnesTotal + (unsigned int)numOnesBefore;

        output[dst] = key;
    }
}

void gpuRadixSortSingleKernel(unsigned int* d_input, int N) {
    assert(N <= MAX_INPUT_SIZE &&
           "Input size above limit leads to potential deadlock due to grid-level synchronization issues.");

    unsigned int* d_output;
    float *d_bits_float, *d_scan_value;
    int* d_flags;
    const int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    CUDA_CHECK(cudaMalloc((void**)&d_output, N * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc((void**)&d_bits_float, (N + 1) * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_scan_value, (numBlocks + 1) * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_flags, (numBlocks + 3) * sizeof(int)));

    for (unsigned int iter = 0; iter < NUM_BITS; iter++) {
        CUDA_CHECK(cudaMemset(d_flags, 0, (numBlocks + 3) * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_scan_value, 0, (numBlocks + 1) * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_bits_float + N, 0, sizeof(float)));

        radix_sort_iter<<<numBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_input, d_output, d_bits_float,
                                                                               d_scan_value, d_flags, N, iter);
        CUDA_CHECK(cudaDeviceSynchronize());

        unsigned int* temp = d_input;
        d_input = d_output;
        d_output = temp;
    }

    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_bits_float));
    CUDA_CHECK(cudaFree(d_scan_value));
    CUDA_CHECK(cudaFree(d_flags));
}

int compare_uint(const void* a, const void* b) {
    unsigned int ua = *(const unsigned int*)a;
    unsigned int ub = *(const unsigned int*)b;
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

int main() {
    int N = 100000;

    unsigned int* h_unsorted = (unsigned int*)malloc(N * sizeof(unsigned int));
    unsigned int* h_quicksort = (unsigned int*)malloc(N * sizeof(unsigned int));
    if (!h_unsorted || !h_quicksort) {
        fprintf(stderr, "Failed to allocate host arrays.\n");
        return EXIT_FAILURE;
    }

    srand((unsigned)time(NULL));
    for (int i = 0; i < N; i++) {
        h_unsorted[i] = rand();
        h_quicksort[i] = h_unsorted[i];
    }

    qsort(h_quicksort, N, sizeof(unsigned int), compare_uint);

    unsigned int* d_array;
    CUDA_CHECK(cudaMalloc(&d_array, N * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(d_array, h_unsorted, N * sizeof(unsigned int), cudaMemcpyHostToDevice));

    gpuRadixSortSingleKernel(d_array, N);

    unsigned int* h_sorted = (unsigned int*)malloc(N * sizeof(unsigned int));
    CUDA_CHECK(cudaMemcpy(h_sorted, d_array, N * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    if (isSorted(h_sorted, N)) {
        printf("GPU sorted array is correctly sorted.\n");
    } else {
        printf("GPU sorted array is NOT sorted correctly!\n");
    }

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

    free(h_sorted);
    free(h_quicksort);
    free(h_unsorted);
    CUDA_CHECK(cudaFree(d_array));

    return 0;
}
