#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <cuda_runtime.h>
#include <limits.h>
#include "gpu_radix_sort.h"

bool isSorted(unsigned int arr[], int size) {
    for (int i = 0; i < size - 1; i++) {
        if (arr[i] > arr[i + 1]) {
            return false;
        }
    }
    return true;
}

int main() {
    int N = 10;
    unsigned int arr[N];

    srand((unsigned)time(NULL));
    for (int i = 0; i < N; i++) {
        arr[i] = (unsigned int)(rand() % UINT_MAX);
    }

    printf("Original array: ");
    for (int i = 0; i < N; i++) {
        printf("%u ", arr[i]);
    }
    printf("\n");

    unsigned int *d_arr;
    cudaMalloc((void**)&d_arr, N * sizeof(unsigned int));
    cudaMemcpy(d_arr, arr, N * sizeof(unsigned int), cudaMemcpyHostToDevice);

    gpuRadixSortDevice(d_arr, N);
    cudaMemcpy(arr, d_arr, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(d_arr);

    printf("Sorted array: ");
    for (int i = 0; i < N; i++) {
        printf("%u ", arr[i]);
    }
    printf("\n");

    printf("Is sorted array sorted? %s\n", isSorted(arr, N) ? "Yes" : "No");

    return 0;
}
