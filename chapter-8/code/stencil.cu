#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <iostream>
#define OUT_TILE_DIM 4
#define IN_TILE_DIM (OUT_TILE_DIM+1)

int c0 = 0;
int c1 = 1;
int c2 = 1;
int c3 = 1;
int c4 = 1;
int c5 = 1;
int c6 = 1;

__constant__ int d_c0, d_c1, d_c2, d_c3, d_c4, d_c5, d_c6;


inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

void stencil_3d_sequential(float* in, float* out, unsigned int N) {
    for (int i = 1; i < N - 1; i++) {
        for (int j = 1; j < N - 1; j++) {
            for (int k = 1; k < N - 1; k++) {
                out[i * N * N + j * N + k] =
                    c0 * in[i * N * N + j * N + k] +
                    c1 * in[i * N * N + j * N + (k-1)] +
                    c2 * in[i * N * N + j * N + (k+1)] +
                    c3 * in[i * N * N + (j-1) * N + k] +
                    c4 * in[i * N * N + (j+1) * N + k] +
                    c5 * in[(i-1) * N * N + j * N + k] +
                    c6 * in[(i+1) * N * N + j * N + k];
            }
        }
    }
}

__global__ void stencil_kernel(float* in, float* out, unsigned int N) {
    unsigned int i = blockIdx.z*blockDim.z + threadIdx.z;
    unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
        out[i*N*N + j*N + k] = d_c0*in[i*N*N + j*N + k]
                             + d_c1*in[i*N*N + j*N + (k - 1)]
                             + d_c2*in[i*N*N + j*N + (k + 1)]
                             + d_c3*in[i*N*N + (j - 1)*N + k]
                             + d_c4*in[i*N*N + (j + 1)*N + k]
                             + d_c5*in[(i - 1)*N*N + j*N + k]
                             + d_c6*in[(i + 1)*N*N + j*N + k];
    }
}

void stencil_3d_parallel_basic(float* in, float* out, unsigned int N){
    float *d_in, *d_out;
    cudaError_t error;

    error = cudaMalloc((void**)&d_in, N*N*N * sizeof(float));
    if (error != cudaSuccess) {
        std::cout << "cudaMalloc d_in failed: " << cudaGetErrorString(error) << std::endl;
        return;
    }

    error = cudaMalloc((void**)&d_out, N*N*N * sizeof(float));
    if (error != cudaSuccess) {
        std::cout << "cudaMalloc d_out failed: " << cudaGetErrorString(error) << std::endl;
        return;
    }

    error = cudaMemcpy(d_in, in, N * N * N * sizeof(float), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        std::cout << "cudaMemcpy to device failed: " << cudaGetErrorString(error) << std::endl;
        return;
    }

    dim3 dimBlock(OUT_TILE_DIM, OUT_TILE_DIM, OUT_TILE_DIM);
    dim3 dimGrid(cdiv(N, dimBlock.x), cdiv(N, dimBlock.y), cdiv(N, dimBlock.z));

    stencil_kernel<<<dimGrid, dimBlock>>>(d_in, d_out, N);

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cout << "Kernel launch failed: " << cudaGetErrorString(error) << std::endl;
    }
    cudaDeviceSynchronize();

    error = cudaMemcpy(out, d_out, N * N * N * sizeof(float), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        std::cout << "cudaMemcpy to host failed: " << cudaGetErrorString(error) << std::endl;
    }

    cudaFree(d_in);
    cudaFree(d_out);
}

__global__ void stencil_kernel_shared_memory(float* in, float* out, unsigned int N) {
    int i = blockIdx.z*OUT_TILE_DIM + threadIdx.z - 1;
    int j = blockIdx.y*OUT_TILE_DIM + threadIdx.y - 1;
    int k = blockIdx.x*OUT_TILE_DIM + threadIdx.x - 1;
    __shared__ float in_s[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];
    if(i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
        in_s[threadIdx.z][threadIdx.y][threadIdx.x] = in[i*N*N + j*N + k];
    }
    __syncthreads();
    if(i >= 1 && i < N-1 && j >= 1 && j < N-1 && k >= 1 && k < N-1) {
        if(threadIdx.z >= 1 && threadIdx.z < IN_TILE_DIM-1 && threadIdx.y >= 1
           && threadIdx.y<IN_TILE_DIM-1 && threadIdx.x>=1 && threadIdx.x<IN_TILE_DIM-1) {
            out[i*N*N + j*N + k] = d_c0*in_s[threadIdx.z][threadIdx.y][threadIdx.x]
                                 + d_c1*in_s[threadIdx.z][threadIdx.y][threadIdx.x-1]
                                 + d_c2*in_s[threadIdx.z][threadIdx.y][threadIdx.x+1]
                                 + d_c3*in_s[threadIdx.z][threadIdx.y-1][threadIdx.x]
                                 + d_c4*in_s[threadIdx.z][threadIdx.y+1][threadIdx.x]
                                 + d_c5*in_s[threadIdx.z-1][threadIdx.y][threadIdx.x]
                                 + d_c6*in_s[threadIdx.z+1][threadIdx.y][threadIdx.x];
        }
    }
}

void stencil_3d_parallel_shared_memory(float* in, float* out, unsigned int N){
    float *d_in, *d_out;
    cudaError_t error;

    error = cudaMalloc((void**)&d_in, N*N*N * sizeof(float));
    if (error != cudaSuccess) {
        std::cout << "cudaMalloc d_in failed: " << cudaGetErrorString(error) << std::endl;
        return;
    }

    error = cudaMalloc((void**)&d_out, N*N*N * sizeof(float));
    if (error != cudaSuccess) {
        std::cout << "cudaMalloc d_out failed: " << cudaGetErrorString(error) << std::endl;
        return;
    }

    error = cudaMemcpy(d_in, in, N * N * N * sizeof(float), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        std::cout << "cudaMemcpy to device failed: " << cudaGetErrorString(error) << std::endl;
        return;
    }

    dim3 dimBlock(OUT_TILE_DIM, OUT_TILE_DIM, OUT_TILE_DIM);
    dim3 dimGrid(cdiv(N, dimBlock.x), cdiv(N, dimBlock.y), cdiv(N, dimBlock.z));

    stencil_kernel_shared_memory<<<dimGrid, dimBlock>>>(d_in, d_out, N);

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cout << "Kernel launch failed: " << cudaGetErrorString(error) << std::endl;
    }
    cudaDeviceSynchronize();

    error = cudaMemcpy(out, d_out, N * N * N * sizeof(float), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        std::cout << "cudaMemcpy to host failed: " << cudaGetErrorString(error) << std::endl;
    }

    cudaFree(d_in);
    cudaFree(d_out);
}

__global__ void stencil_kernel_thread_coarsening(float* in, float* out, unsigned int N) {
    int iStart = blockIdx.z*OUT_TILE_DIM;
    int j = blockIdx.y*OUT_TILE_DIM + threadIdx.y - 1;
    int k = blockIdx.x*OUT_TILE_DIM + threadIdx.x - 1;
    __shared__ float inPrev_s[IN_TILE_DIM][IN_TILE_DIM];
    __shared__ float inCurr_s[IN_TILE_DIM][IN_TILE_DIM];
    __shared__ float inNext_s[IN_TILE_DIM][IN_TILE_DIM];
    if(iStart-1 >= 0 && iStart-1 < N && j >= 0 && j < N && k >= 0 && k < N) {
        inPrev_s[threadIdx.y][threadIdx.x] = in[(iStart - 1)*N*N + j*N + k];
    }
    if(iStart >= 0 && iStart < N && j >= 0 && j < N && k >= 0 && k < N) {
        inCurr_s[threadIdx.y][threadIdx.x] = in[iStart*N*N + j*N + k];
    }
    for(int i = iStart; i < iStart + OUT_TILE_DIM; ++i) {
        if(i + 1 >= 0 && i + 1 < N && j >= 0 && j < N && k >= 0 && k < N) {
            inNext_s[threadIdx.y][threadIdx.x] = in[(i + 1)*N*N + j*N + k];
        }
        __syncthreads();
        if(i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
            if(threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM - 1
               && threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM - 1) {
                out[i*N*N + j*N + k] = d_c0*inCurr_s[threadIdx.y][threadIdx.x]
                                     + d_c1*inCurr_s[threadIdx.y][threadIdx.x-1]
                                     + d_c2*inCurr_s[threadIdx.y][threadIdx.x+1]
                                     + d_c3*inCurr_s[threadIdx.y+1][threadIdx.x]
                                     + d_c4*inCurr_s[threadIdx.y-1][threadIdx.x]
                                     + d_c5*inPrev_s[threadIdx.y][threadIdx.x]
                                     + d_c6*inNext_s[threadIdx.y][threadIdx.x];
            }
        }
        __syncthreads();
        inPrev_s[threadIdx.y][threadIdx.x] = inCurr_s[threadIdx.y][threadIdx.x];
        inCurr_s[threadIdx.y][threadIdx.x] = inNext_s[threadIdx.y][threadIdx.x];
    }
}

void stencil_3d_parallel_thread_coarsening(float* in, float* out, unsigned int N){
    float *d_in, *d_out;
    cudaError_t error;

    error = cudaMalloc((void**)&d_in, N*N*N * sizeof(float));
    if (error != cudaSuccess) {
        std::cout << "cudaMalloc d_in failed: " << cudaGetErrorString(error) << std::endl;
        return;
    }

    error = cudaMalloc((void**)&d_out, N*N*N * sizeof(float));
    if (error != cudaSuccess) {
        std::cout << "cudaMalloc d_out failed: " << cudaGetErrorString(error) << std::endl;
        return;
    }

    error = cudaMemcpy(d_in, in, N * N * N * sizeof(float), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        std::cout << "cudaMemcpy to device failed: " << cudaGetErrorString(error) << std::endl;
        return;
    }

    dim3 dimBlock(IN_TILE_DIM, IN_TILE_DIM, 1);  // 5x5x1 = 25 threads
    dim3 dimGrid(cdiv(N, OUT_TILE_DIM), cdiv(N, OUT_TILE_DIM), cdiv(N, OUT_TILE_DIM));

    stencil_kernel_thread_coarsening<<<dimGrid, dimBlock>>>(d_in, d_out, N);

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cout << "Kernel launch failed: " << cudaGetErrorString(error) << std::endl;
    }
    cudaDeviceSynchronize();

    error = cudaMemcpy(out, d_out, N * N * N * sizeof(float), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        std::cout << "cudaMemcpy to host failed: " << cudaGetErrorString(error) << std::endl;
    }

    cudaFree(d_in);
    cudaFree(d_out);
}


void print_3d_slice(float* data, int N, int slice_i) {
    printf("Slice i=%d:\n", slice_i);
    for (int j = 0; j < N; j++) {
        for (int k = 0; k < N; k++) {
            printf("%6.1f ", data[slice_i * N * N + j * N + k]);
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    cudaMemcpyToSymbol(d_c0, &c0, sizeof(int));
    cudaMemcpyToSymbol(d_c1, &c1, sizeof(int));
    cudaMemcpyToSymbol(d_c2, &c2, sizeof(int));
    cudaMemcpyToSymbol(d_c3, &c3, sizeof(int));
    cudaMemcpyToSymbol(d_c4, &c4, sizeof(int));
    cudaMemcpyToSymbol(d_c5, &c5, sizeof(int));
    cudaMemcpyToSymbol(d_c6, &c6, sizeof(int));


    // Test with a small 4x4x4 grid
    unsigned int N = 4;
    int total_size = N * N * N;
    
    // Allocate memory
    float* in = (float*)malloc(total_size * sizeof(float));
    float* out = (float*)malloc(total_size * sizeof(float));
    
    // Initialize input data with simple pattern
    // Set all to 0 first
    memset(in, 0, total_size * sizeof(float));
    memset(out, 0, total_size * sizeof(float));
    
    // Put a "hot spot" in the center
    in[1 * N * N + 1 * N + 1] = 10.0f;  // Center point
    in[1 * N * N + 1 * N + 2] = 5.0f;   // Adjacent points
    in[1 * N * N + 2 * N + 1] = 5.0f;
    in[2 * N * N + 1 * N + 1] = 5.0f;
    
    printf("Input data:\n");
    for (int i = 0; i < N; i++) {
        print_3d_slice(in, N, i);
    }
    
    // Run the stencil
    // stencil_3d_sequential(in, out, N);
    stencil_3d_parallel_shared_memory(in, out, N);
    
    printf("Output data:\n");
    for (int i = 0; i < N; i++) {
        print_3d_slice(out, N, i);
    }
    
    return 0;
}