#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <time.h>

namespace cg = cooperative_groups;

#define THREADS_PER_BLOCK 256
#define MAX_BLOCKS 32
#define NUM_BITS 32

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(2); \
    } \
}

bool isSorted(unsigned int* arr, int size) {
    for (int i = 0; i < size - 1; i++) {
        if (arr[i] > arr[i + 1]) return false;
    }
    return true;
}

__device__ void exclusive_scan(
    float* X, float* block_sums, unsigned int N) 
{
    cg::grid_group grid = cg::this_grid();
    
    extern __shared__ float shared[];
    const unsigned int tid = threadIdx.x;
    const unsigned int bid = blockIdx.x;
    const unsigned int gid = bid * blockDim.x + tid;
    
    // Load into shared memory
    if (gid < N) {
        shared[tid] = X[gid];
    } else {
        shared[tid] = 0.0f;
    }
    __syncthreads();

    // Kogge-Stone scan within block
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        float temp = 0.0f;
        if (tid >= stride) {
            temp = shared[tid - stride];
        }
        __syncthreads();
        if (tid >= stride) {
            shared[tid] += temp;
        }
        __syncthreads();
    }

    // Save block sum and make exclusive
    if (tid == blockDim.x - 1) {
        block_sums[bid] = shared[tid];
    }
    float my_val = (tid > 0) ? shared[tid - 1] : 0.0f;
    
    grid.sync();

    // Add in sums from previous blocks
    float prefix_sum = 0.0f;
    if (tid == 0) {
        for (int i = 0; i < bid; i++) {
            prefix_sum += block_sums[i];
        }
    }
    __syncthreads();
    
    // Share prefix sum
    __shared__ float block_prefix;
    if (tid == 0) {
        block_prefix = prefix_sum;
    }
    __syncthreads();
    
    // Write final result
    if (gid < N) {
        X[gid] = my_val + block_prefix;
    }
}

__global__ void radix_sort_kernel(
    unsigned int* input, unsigned int* output,
    float* bits, float* block_sums,
    unsigned int N, unsigned int bit_pos)
{
    cg::grid_group grid = cg::this_grid();
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Extract bits
    if (gid < N) {
        bits[gid] = (float)((input[gid] >> bit_pos) & 1);
    }
    
    // Count ones
    if (gid == 0) {
        float ones = 0.0f;
        for (int i = 0; i < N; i++) {
            ones += bits[i];
        }
        bits[N] = ones;
    }

    // Exclusive scan
    exclusive_scan(bits, block_sums, N);
    grid.sync();
    
    // Reorder elements
    if (gid < N) {
        unsigned int val = input[gid];
        unsigned int bit = (val >> bit_pos) & 1;
        float pos = bits[gid];
        float total_ones = bits[N];
        
        unsigned int new_pos;
        if (bit == 0) {
            new_pos = gid - (unsigned int)pos;
        } else {
            new_pos = (N - (unsigned int)total_ones) + (unsigned int)pos;
        }
        output[new_pos] = val;
    }
}

void gpuRadixSort(unsigned int* d_input, int N) {
    unsigned int* d_output;
    float* d_bits;
    float* d_block_sums;
    
    int num_blocks = min((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, MAX_BLOCKS);
    
    CHECK_CUDA(cudaMalloc(&d_output, N * sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&d_bits, (N + 1) * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_block_sums, num_blocks * sizeof(float)));
    
    for (int bit = 0; bit < NUM_BITS; bit++) {
        CHECK_CUDA(cudaMemset(d_bits, 0, (N + 1) * sizeof(float)));
        CHECK_CUDA(cudaMemset(d_block_sums, 0, num_blocks * sizeof(float)));
        
        void* args[] = {
            (void*)&d_input,
            (void*)&d_output,
            (void*)&d_bits,
            (void*)&d_block_sums,
            (void*)&N,
            (void*)&bit
        };
        
        dim3 grid(num_blocks);
        dim3 block(THREADS_PER_BLOCK);
        
        CHECK_CUDA(cudaLaunchCooperativeKernel(
            (void*)radix_sort_kernel,
            grid, block, args,
            THREADS_PER_BLOCK * sizeof(float)
        ));
        
        unsigned int* temp = d_input;
        d_input = d_output;
        d_output = temp;
    }
    
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_bits));
    CHECK_CUDA(cudaFree(d_block_sums));
}

int main() {
    // Set array size.
    int N = 200;
    
    // Allocate and initialize host array.
    unsigned int* h_unsorted = (unsigned int*)malloc(N * sizeof(unsigned int));
    if (!h_unsorted) {
        fprintf(stderr, "Failed to allocate host array.\n");
        return EXIT_FAILURE;
    }
    
    srand((unsigned)time(NULL));
    for (int i = 0; i < N; i++) {
        h_unsorted[i] = rand() % 100;  // Numbers between 0 and 99
    }
    
    printf("Input array:\n");
    for (int i = 0; i < N; i++) {
        printf("%u, ", h_unsorted[i]);
    }
    printf("\n");
    
    // Allocate device array and copy input.
    unsigned int* d_array;
    CHECK_CUDA(cudaMalloc(&d_array, N * sizeof(unsigned int)));
    CHECK_CUDA(cudaMemcpy(d_array, h_unsorted, N * sizeof(unsigned int), cudaMemcpyHostToDevice));
    
    // Check for cooperative launch support.
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (!prop.cooperativeLaunch) {
        printf("Device does not support cooperative launch, exiting.\n");
        exit(EXIT_FAILURE);
    }
    
    // Run the GPU radix sort.
    gpuRadixSort(d_array, N);
    
    // Copy back the result.
    unsigned int* h_sorted = (unsigned int*)malloc(N * sizeof(unsigned int));
    CHECK_CUDA(cudaMemcpy(h_sorted, d_array, N * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    
    if (isSorted(h_sorted, N))
        printf("GPU sorted array is correct.\n");
    else
        printf("GPU sorted array is NOT sorted correctly!\n");
    
    printf("Output array:\n");
    for (int i = 0; i < N; i++) {
        printf("%u, ", h_sorted[i]);
    }
    printf("\n");
    
    free(h_unsorted);
    free(h_sorted);
    CHECK_CUDA(cudaFree(d_array));
    
    return 0;
}