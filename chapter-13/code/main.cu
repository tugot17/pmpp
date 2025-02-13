// main.cu
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>
#include <cuda_runtime.h>
#include <limits.h>
#include "gpu_radix_sort.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void clear_l2() {
    static int l2_clear_size = 0;
    static unsigned char* gpu_scratch_l2_clear = NULL;
    if (!gpu_scratch_l2_clear) {
        int dev;
        gpuErrchk(cudaGetDevice(&dev));
        gpuErrchk(cudaDeviceGetAttribute(&l2_clear_size, cudaDevAttrL2CacheSize, dev));
        l2_clear_size *= 2;  // Use a buffer twice the L2 size
        gpuErrchk(cudaMalloc(&gpu_scratch_l2_clear, l2_clear_size));
    }
    gpuErrchk(cudaMemset(gpu_scratch_l2_clear, 0, l2_clear_size));
}

float do_bench(void (*sort_func)(unsigned int*, int),
               unsigned int* d_array,
               const unsigned int* h_unsorted,
               int N,
               int warmup,
               int reps)
{
    for (int i = 0; i < warmup; i++) {
        gpuErrchk(cudaMemcpy(d_array, h_unsorted, N * sizeof(unsigned int),
                             cudaMemcpyHostToDevice));
        sort_func(d_array, N);
    }

    cudaEvent_t start, stop;
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));

    float totalTime_ms = 0.0f;
    for (int i = 0; i < reps; i++) {
        // Before each run, copy the unsorted data into device memory.
        gpuErrchk(cudaMemcpy(d_array, h_unsorted, N * sizeof(unsigned int),
                             cudaMemcpyHostToDevice));

        clear_l2();  // Clear L2 cache to reduce inter-run effects

        gpuErrchk(cudaEventRecord(start, 0));
        sort_func(d_array, N);
        gpuErrchk(cudaEventRecord(stop, 0));
        gpuErrchk(cudaEventSynchronize(stop));

        float iterTime = 0.0f;
        gpuErrchk(cudaEventElapsedTime(&iterTime, start, stop));
        totalTime_ms += iterTime;
    }
    gpuErrchk(cudaEventDestroy(start));
    gpuErrchk(cudaEventDestroy(stop));

    return totalTime_ms / reps;
}

double do_bench_cpu(void (*sort_func)(unsigned int*, int),
                    const unsigned int* h_unsorted,
                    int N,
                    int warmup,
                    int reps)
{
    unsigned int* temp = (unsigned int*)malloc(N * sizeof(unsigned int));
    if (!temp) {
        fprintf(stderr, "Failed to allocate temporary memory for CPU benchmark.\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < warmup; i++) {
        memcpy(temp, h_unsorted, N * sizeof(unsigned int));
        sort_func(temp, N);
    }

    double totalTime_ms = 0.0;
    for (int i = 0; i < reps; i++) {
        memcpy(temp, h_unsorted, N * sizeof(unsigned int));
        clock_t start = clock();
        sort_func(temp, N);
        clock_t end = clock();
        double elapsed_ms = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
        totalTime_ms += elapsed_ms;
    }
    free(temp);
    return totalTime_ms / reps;
}

int compare_uint(const void* a, const void* b) {
    unsigned int ua = *(const unsigned int*) a;
    unsigned int ub = *(const unsigned int*) b;
    return (ua > ub) - (ua < ub);
}

void cpuSortWrapper(unsigned int* arr, int N) {
    qsort(arr, N, sizeof(unsigned int), compare_uint);
}

bool isSorted(unsigned int arr[], int size) {
    for (int i = 0; i < size - 1; i++) {
        if (arr[i] > arr[i + 1]) {
            return false;
        }
    }
    return true;
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>
#include <cuda_runtime.h>
#include <limits.h>
#include "gpu_radix_sort.h"

// [Previous code remains the same until the benchmarking section]

int main() {
    int N = 100000;
    int warmup = 5;
    int reps = 15;

    unsigned int* h_unsorted = (unsigned int*)malloc(N * sizeof(unsigned int));
    if (!h_unsorted) {
        fprintf(stderr, "Failed to allocate host array.\n");
        return EXIT_FAILURE;
    }

    srand((unsigned)time(NULL));
    for (int i = 0; i < N; i++) {
        h_unsorted[i] = rand();
    }

    unsigned int* d_array;
    gpuErrchk(cudaMalloc(&d_array, N * sizeof(unsigned int)));

    unsigned int* h_sorted = NULL;

    // Test and benchmark basic three-kernel implementation
    float gpu_three_kernel_ms = do_bench(gpuRadixSortThreeKernels, d_array, h_unsorted, N, warmup, reps);
    printf("Average GPU basic three-kernel sort time: %f ms\n", gpu_three_kernel_ms);

    h_sorted = (unsigned int*)malloc(N * sizeof(unsigned int));
    gpuErrchk(cudaMemcpy(h_sorted, d_array, N * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    if (isSorted(h_sorted, N)) {
        printf("Basic three-kernel GPU sort is correct.\n");
    } else {
        printf("Basic three-kernel GPU sort is NOT sorted correctly!\n");
    }
    free(h_sorted);

    // // Test and benchmark single-kernel sort
    // float gpu_single_kernel_ms = do_bench(gpuRadixSortSingleKernel, d_array, h_unsorted, N, warmup, reps);
    // printf("Average GPU single-kernel sort time: %f ms\n", gpu_single_kernel_ms);

    // h_sorted = (unsigned int*)malloc(N * sizeof(unsigned int));
    // gpuErrchk(cudaMemcpy(h_sorted, d_array, N * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    // if (isSorted(h_sorted, N)) {
    //     printf("Single-kernel GPU sort is correct.\n");
    // } else {
    //     printf("Single-kernel GPU sort is NOT sorted correctly!\n");
    // }
    // free(h_sorted);

    // Test and benchmark memory-coalesced version
    float gpu_coalesced_ms = do_bench(gpuRadixSortWithMemoryCoalescing, d_array, h_unsorted, N, warmup, reps);
    printf("Average GPU memory-coalesced sort time: %f ms\n", gpu_coalesced_ms);

    h_sorted = (unsigned int*)malloc(N * sizeof(unsigned int));
    gpuErrchk(cudaMemcpy(h_sorted, d_array, N * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    if (isSorted(h_sorted, N)) {
        printf("Memory-coalesced GPU sort is correct.\n");
    } else {
        printf("Memory-coalesced GPU sort is NOT sorted correctly!\n");
    }
    free(h_sorted);

    // Test and benchmark CPU sort
    double cpu_avg_ms = do_bench_cpu(cpuSortWrapper, h_unsorted, N, warmup, reps);
    printf("Average CPU sort time (qsort): %f ms\n", cpu_avg_ms);

    printf("\nPerformance Summary:\n");
    printf("Basic three-kernel GPU Speedup vs CPU: %.2fx\n", cpu_avg_ms / gpu_three_kernel_ms);
    // printf("Single-kernel GPU Speedup vs CPU: %.2fx\n", cpu_avg_ms / gpu_single_kernel_ms);
    printf("Memory-coalesced GPU Speedup vs CPU: %.2fx\n", cpu_avg_ms / gpu_coalesced_ms);
    // printf("Single-kernel vs Three-kernel ratio: %.2fx\n", gpu_three_kernel_ms / gpu_single_kernel_ms);
    printf("Memory-coalesced vs basic three-kernel speedup: %.2fx\n", gpu_three_kernel_ms / gpu_coalesced_ms);
    // printf("Memory-coalesced vs Single-kernel ratio: %.2fx\n", gpu_single_kernel_ms / gpu_improved_ms);

    gpuErrchk(cudaFree(d_array));
    free(h_unsorted);

    return 0;
}
