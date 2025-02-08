// sort.cu
// Compile with: nvcc sort.cu -o sort -rdc=true

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <time.h>
#include <stdbool.h>
#include <string.h>

namespace cg = cooperative_groups;

#define SECTION_SIZE 4  // Threads per block

// Utility: check for CUDA errors.
#define cudaCheckError() {                                         \
    cudaError_t err = cudaGetLastError();                          \
    if(err != cudaSuccess) {                                        \
        printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1);                                                  \
    }                                                              \
}

// Returns true if the array is sorted (ascending).
bool isSorted(unsigned int arr[], int size) {
    for (int i = 0; i < size - 1; i++) {
        if (arr[i] > arr[i + 1])
            return false;
    }
    return true;
}

// *************************************************************************
// Device function: Hierarchical Kogge–Stone Domino Exclusive Scan
// This function scans the array X[0..N-1] (here X holds the bit values)
// and writes the exclusive scan back into X. It also uses the per-block
// ordering given by blockIdx.x.  
//
// We add an extra grid-wide sync (via cooperative groups) after writing
// the per-block sums so that every block can safely accumulate the sums 
// from previous blocks.
// *************************************************************************
__device__ void hierarchical_kogge_stone_domino_exclusive_inplace(
    float* X, float* scan_value, unsigned int N)
{
    // Create a grid group (all threads in the kernel).
    cg::grid_group grid = cg::this_grid();

    // Use dynamically allocated shared memory (size = blockDim.x * sizeof(float))
    extern __shared__ float buffer[];
    const unsigned int tid = threadIdx.x;
    // Use blockIdx.x as the block’s order.
    const unsigned int bid = blockIdx.x;
    const unsigned int gid = bid * blockDim.x + tid;

    // Phase 1: Load data (or 0 for out-of-bound indices) into shared memory.
    if (gid < N)
        buffer[tid] = X[gid];
    else
        buffer[tid] = 0.0f;
    __syncthreads();

    // Phase 2: Perform an inclusive scan (Kogge–Stone style) in shared memory.
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        float temp = buffer[tid];
        if (tid >= stride)
            temp += buffer[tid - stride];
        __syncthreads();
        buffer[tid] = temp;
        __syncthreads();
    }
    
    // Compute the exclusive scan value (shift right by one).
    float exclusive_value = (tid == 0) ? 0.0f : buffer[tid - 1];
    float local_sum = buffer[blockDim.x - 1];

    // Phase 3: Write this block’s total sum to the per-block array.
    if (tid == 0)
        scan_value[bid] = local_sum;
    __syncthreads();

    // *** NEW GRID-WIDE SYNC ***
    // Ensure that every block has written its sum before we accumulate.
    grid.sync();

    // Phase 4: Compute the sum of all previous blocks (using blockIdx.x order).
    float prev_sum = 0.0f;
    if (tid == 0) {
        for (int i = 0; i < bid; i++) {
            prev_sum += scan_value[i];
        }
    }
    __syncthreads();
    // Broadcast prev_sum to all threads in the block.
    __shared__ float shared_prev;
    if (tid == 0)
        shared_prev = prev_sum;
    __syncthreads();
    prev_sum = shared_prev;

    // Phase 5: Write the final exclusive scan result back to global memory.
    if (gid < N)
        X[gid] = exclusive_value + prev_sum;
    __syncthreads();
}

// *************************************************************************
// Kernel: radix_sort_iter
//
// For a given bit-position (iter), this kernel computes for each element:
//   - The bit value (0 or 1).
//   - The grid-wide total number of ones (stored at X[N]).
//   - An exclusive scan of the bit values (using our hierarchical scan).
// Then it uses these results to scatter the keys to output positions.
// 
// Note: The scatter formula is:
//   if (bit == 0): dst = i - (exclusive prefix of ones)
//   else:          dst = (N - total ones) + (exclusive prefix of ones)
// *************************************************************************
__global__ void radix_sort_iter(
    unsigned int* input, unsigned int* output,
    float* bits_float, float* scan_value,
    unsigned int N, unsigned int iter)
{
    // Create a grid group representing all threads.
    cg::grid_group grid = cg::this_grid();

    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Step 1: Compute the bit for each element.
    if (i < N) {
        unsigned int key = input[i];
        unsigned int bit = (key >> iter) & 1;
        bits_float[i] = (float)bit;
    }
    __syncthreads(); // Block-level sync

    // Step 2: Compute the total number of ones (only one thread does this).
    if (i == 0) {
        float total = 0.0f;
        for (int j = 0; j < N; j++) {
            total += (float)((input[j] >> iter) & 1);
        }
        bits_float[N] = total;
    }
    // Grid-wide sync to ensure the total is visible to all threads.
    grid.sync();

    // Step 3: Perform the exclusive scan on bits_float[0..N-1].
    hierarchical_kogge_stone_domino_exclusive_inplace(bits_float, scan_value, N);
    __syncthreads();

    // Step 4: Scatter keys to output positions.
    // For an element:
    //   If bit==0: destination = i - (exclusive prefix sum)
    //   If bit==1: destination = (N - total ones) + (exclusive prefix sum)
    if (i < N) {
        unsigned int key = input[i];
        unsigned int bit = (key >> iter) & 1;
        float numOnesBefore = bits_float[i];   // Exclusive prefix sum.
        float numOnesTotal  = bits_float[N];     // Total number of ones.
        unsigned int dst;
        if (bit == 0)
            dst = i - (unsigned int)numOnesBefore;
        else
            dst = (N - (unsigned int)numOnesTotal) + (unsigned int)numOnesBefore;
        output[dst] = key;
    }
}

// *************************************************************************
// Host function: gpuRadixSort
//
// Allocates temporary arrays on the device and launches the cooperative
// kernel for each iteration.
// *************************************************************************
void gpuRadixSort(unsigned int *d_input, int N) {
    unsigned int *d_output;
    float *d_bits_float, *d_scan_value;
    const int threadsPerBlock = SECTION_SIZE;
    const int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate device memory.
    cudaMalloc((void**)&d_output, N * sizeof(unsigned int));            cudaCheckError();
    cudaMalloc((void**)&d_bits_float, (N + 1) * sizeof(float));            cudaCheckError();
    cudaMalloc((void**)&d_scan_value, (numBlocks) * sizeof(float));          cudaCheckError();

    // For debugging: copy the initial array to host.
    unsigned int *h_input = (unsigned int*)malloc(N * sizeof(unsigned int));
    cudaMemcpy(h_input, d_input, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // For demonstration, perform two iterations (one per bit).
    for (unsigned int iter = 0; iter < 32; iter++) {
        printf("\n=== Iteration %u ===\n", iter);
        printf("Current array: ");
        for (int i = 0; i < N; i++) {
            printf("%u ", h_input[i]);
        }
        printf("\n");

        // Reset the per-block scan array and the extra element for total.
        cudaMemset(d_scan_value, 0, numBlocks * sizeof(float));            cudaCheckError();
        cudaMemset(d_bits_float + N, 0, sizeof(float));                      cudaCheckError();

        // Debug: print the binary bits for this iteration.
        printf("Binary bits at position %u: ", iter);
        for (int i = 0; i < N; i++) {
            printf("%u ", (h_input[i] >> iter) & 1);
        }
        printf("\n");

        // Set up kernel parameters.
        void* kernelArgs[] = {
            (void*)&d_input,
            (void*)&d_output,
            (void*)&d_bits_float,
            (void*)&d_scan_value,
            (void*)&N,
            (void*)&iter
        };

        dim3 blockDim(threadsPerBlock);
        dim3 gridDim(numBlocks);

        // Launch the kernel cooperatively.
        cudaError_t err = cudaLaunchCooperativeKernel(
            (void*)radix_sort_iter, gridDim, blockDim, kernelArgs,
            threadsPerBlock * sizeof(float) // shared memory per block
        );
        if (err != cudaSuccess) {
            printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
            exit(1);
        }
        cudaDeviceSynchronize();  cudaCheckError();

        // Swap input and output for the next iteration.
        unsigned int *temp = d_input;
        d_input = d_output;
        d_output = temp;

        // Update host copy.
        cudaMemcpy(h_input, d_input, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    }

    free(h_input);
    cudaFree(d_output);       cudaCheckError();
    cudaFree(d_bits_float);   cudaCheckError();
    cudaFree(d_scan_value);   cudaCheckError();
}

// *************************************************************************
// Main function
// *************************************************************************
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
        h_unsorted[i] = rand() % 100;  // Numbers between 0 and 3.
    }

    printf("Input array:\n");
    for (int i = 0; i < N; i++){
        printf("%u, ", h_unsorted[i]);
    }
    printf("\n");

    // Allocate device array and copy input.
    unsigned int* d_array;
    cudaMalloc(&d_array, N * sizeof(unsigned int));  cudaCheckError();
    cudaMemcpy(d_array, h_unsorted, N * sizeof(unsigned int), cudaMemcpyHostToDevice);  cudaCheckError();

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
    cudaMemcpy(h_sorted, d_array, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);  cudaCheckError();

    if (isSorted(h_sorted, N))
        printf("GPU sorted array is correct.\n");
    else
        printf("GPU sorted array is NOT sorted correctly!\n");

    printf("Output array:\n");
    for (int i = 0; i < N; i++){
        printf("%u, ", h_sorted[i]);
    }
    printf("\n");

    free(h_unsorted);
    free(h_sorted);
    cudaFree(d_array);  cudaCheckError();

    return 0;
}
