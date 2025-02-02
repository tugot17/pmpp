#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <cuda_runtime.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <limits.h>

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

__global__ void extractBitsKernel(unsigned int* input, unsigned int* bits,
                                  unsigned int N, unsigned int iter) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        unsigned int key = input[tid];
        bits[tid] = (key >> iter) & 1;
    }
}

__global__ void scatterKernel(unsigned int* input, unsigned int* output,
                              unsigned int* scannedBits, unsigned int N, unsigned int iter) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        unsigned int key = input[tid];
        unsigned int bit = (key >> iter) & 1;
        unsigned int numOnesBefore = scannedBits[tid];
        unsigned int numOnesTotal = scannedBits[N - 1];
        unsigned int dst = (bit == 0) ? (tid - numOnesBefore)
                                      : (N - numOnesTotal + numOnesBefore);
        output[dst] = key;
    }
}

void gpuRadixSort(unsigned int *arr, int N) {
    unsigned int *d_input, *d_output, *d_bits;
    
    cudaMalloc((void**)&d_input,  N * sizeof(unsigned int));
    cudaMalloc((void**)&d_output, N * sizeof(unsigned int));
    cudaMalloc((void**)&d_bits,   N * sizeof(unsigned int));
    
    cudaMemcpy(d_input, arr, N * sizeof(unsigned int), cudaMemcpyHostToDevice);
    
    const int threadsPerBlock = 256;
    const int numBlocks
     = (N + threadsPerBlock - 1) / threadsPerBlock;
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

    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        arr[i] = (unsigned int)(rand() % UINT_MAX);
    }

    printf("Original array: ");
    for (int i = 0; i < N; i++) {
        printf("%u ", arr[i]);
    }
    printf("\n");

    gpuRadixSort(arr, N);

    printf("Sorted array: ");
    for (int i = 0; i < N; i++) {
        printf("%u ", arr[i]);
    }
    printf("\n");

    printf("Is sorted array sorted? %s\n", isSorted(arr, N) ? "Yes" : "No");
    return 0;
}
