#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess) {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }

    printf("Detected %d CUDA capable device(s)\n", deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
        printf("  Major revision number:         %d\n", deviceProp.major);
        printf("  Minor revision number:         %d\n", deviceProp.minor);
        printf("  Total amount of global memory: %.2f GB\n", (float)deviceProp.totalGlobalMem / (1024 * 1024 * 1024));
        printf("  Number of multiprocessors:     %d\n", deviceProp.multiProcessorCount);
        printf("  Total amount of constant memory: %lu bytes\n", deviceProp.totalConstMem);
        printf("  Total amount of shared memory per block: %lu bytes\n", deviceProp.sharedMemPerBlock);
        printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
        printf("  Warp size:                     %d\n", deviceProp.warpSize);
        printf("  Maximum number of threads per block: %d\n", deviceProp.maxThreadsPerBlock);
        printf("  Maximum sizes of each dimension of a block: %d x %d x %d\n",
               deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("  Maximum sizes of each dimension of a grid: %d x %d x %d\n",
               deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        printf("  Clock rate:                    %.2f GHz\n", deviceProp.clockRate * 1e-6f);
        printf("  Memory clock rate:             %d MHz\n", deviceProp.memoryClockRate / 1000);
        printf("  Memory bus width:              %d-bit\n", deviceProp.memoryBusWidth);
        printf("  L2 cache size:                 %d bytes\n", deviceProp.l2CacheSize);
    }

    return 0;
}
