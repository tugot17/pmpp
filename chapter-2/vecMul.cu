#include <stdio.h>
#include <cuda_runtime.h>


void vecMulHost(float* A_h, float* B_h, float* C_h, int n){
    for (int i=0; i < n; i++){
        C_h[i] = A_h[i] * B_h[i];        
    }
}

__global__
void vecMulKernel(float* A, float* B, float* C, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n){
        C[i] = A[i] * B[i];
    }
}

void vecMulDevice(float* A_h, float* B_h, float* C_h, int n){
    int size = n * sizeof(float);
    float *A_d, *B_d, *C_d;

    //allocate the memory on the decive
    cudaMalloc((void**) &A_d, size);
    cudaMalloc((void**) &B_d, size);
    cudaMalloc((void**) &C_d, size);

    //sent data to the device
    //pntr destination, pntr source, num bytes copied, direction of transfer
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    //invoke a kernel
    vecMulKernel<<<ceil(n/256.0), 256>>>(A_d, B_d, C_d, n);

    //copy the result back to the host
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    //free up the allocated memory
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main() {
    int n = 123397730; // Example array size
    float *A, *B, *C;

    // Allocate host memory
    A = (float*)malloc(n * sizeof(float));
    B = (float*)malloc(n * sizeof(float));
    C = (float*)malloc(n * sizeof(float));

    // Initialize A and B with values
    for(int i = 0; i < n; i++) {
        A[i] = i; // Just an example value
        B[i] = i; // Just an example value
    }

    // Timing for vecMulHost
    clock_t start_host = clock();
    vecMulHost(A, B, C, n);
    clock_t end_host = clock();
    double time_host = (double)(end_host - start_host) / CLOCKS_PER_SEC;

    // Timing for vecMulDevice
    clock_t start_device = clock();
    vecMulDevice(A, B, C, n);
    clock_t end_device = clock();
    double time_device = (double)(end_device - start_device) / CLOCKS_PER_SEC;

    // for(int i = 0; i < n; i++){
    //     printf("Element %d = %.1f\n", i, *(C + i));
    // }

    printf("Time taken for vecMulHost: %f seconds\n", time_host);
    printf("Time taken for vecMulDevice: %f seconds\n", time_device);

    return 0;
}
