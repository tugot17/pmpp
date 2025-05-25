#include <iostream>

#include "stencil.h"

// Global coefficient variables definition
int c0 = 0;
int c1 = 1;
int c2 = 1;
int c3 = 1;
int c4 = 1;
int c5 = 1;
int c6 = 1;

void stencil_3d_sequential(float* in, float* out, unsigned int N, int c0, int c1, int c2, int c3, int c4, int c5,
                           int c6) {
    for (unsigned int i = 1; i < N - 1; i++) {
        for (unsigned int j = 1; j < N - 1; j++) {
            for (unsigned int k = 1; k < N - 1; k++) {
                out[i * N * N + j * N + k] =
                    c0 * in[i * N * N + j * N + k] + c1 * in[i * N * N + j * N + (k - 1)] +
                    c2 * in[i * N * N + j * N + (k + 1)] + c3 * in[i * N * N + (j - 1) * N + k] +
                    c4 * in[i * N * N + (j + 1) * N + k] + c5 * in[(i - 1) * N * N + j * N + k] +
                    c6 * in[(i + 1) * N * N + j * N + k];
            }
        }
    }
}

__global__ void stencil_kernel(float* in, float* out, unsigned int N, int c0, int c1, int c2, int c3, int c4, int c5,
                               int c6) {
    unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
        out[i * N * N + j * N + k] = c0 * in[i * N * N + j * N + k] + c1 * in[i * N * N + j * N + (k - 1)] +
                                     c2 * in[i * N * N + j * N + (k + 1)] + c3 * in[i * N * N + (j - 1) * N + k] +
                                     c4 * in[i * N * N + (j + 1) * N + k] + c5 * in[(i - 1) * N * N + j * N + k] +
                                     c6 * in[(i + 1) * N * N + j * N + k];
    }
}

void stencil_3d_parallel_basic(float* in, float* out, unsigned int N, int c0, int c1, int c2, int c3, int c4, int c5,
                               int c6) {
    float *d_in, *d_out;
    cudaError_t error;

    error = cudaMalloc((void**)&d_in, N * N * N * sizeof(float));
    if (error != cudaSuccess) {
        std::cout << "cudaMalloc d_in failed: " << cudaGetErrorString(error) << std::endl;
        return;
    }

    error = cudaMalloc((void**)&d_out, N * N * N * sizeof(float));
    if (error != cudaSuccess) {
        std::cout << "cudaMalloc d_out failed: " << cudaGetErrorString(error) << std::endl;
        return;
    }

    error = cudaMemcpy(d_in, in, N * N * N * sizeof(float), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        std::cout << "cudaMemcpy to device failed: " << cudaGetErrorString(error) << std::endl;
        return;
    }

    dim3 dimBlock(OUT_TILE_DIM_SMALL, OUT_TILE_DIM_SMALL, OUT_TILE_DIM_SMALL);
    dim3 dimGrid(cdiv(N, dimBlock.x), cdiv(N, dimBlock.y), cdiv(N, dimBlock.z));

    stencil_kernel<<<dimGrid, dimBlock>>>(d_in, d_out, N, c0, c1, c2, c3, c4, c5, c6);

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

__global__ void stencil_kernel_shared_memory(float* in, float* out, unsigned int N, int c0, int c1, int c2, int c3,
                                             int c4, int c5, int c6) {
    int i = blockIdx.z * OUT_TILE_DIM_SMALL + threadIdx.z - 1;
    int j = blockIdx.y * OUT_TILE_DIM_SMALL + threadIdx.y - 1;
    int k = blockIdx.x * OUT_TILE_DIM_SMALL + threadIdx.x - 1;
    __shared__ float in_s[IN_TILE_DIM_SMALL][IN_TILE_DIM_SMALL][IN_TILE_DIM_SMALL];
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
        in_s[threadIdx.z][threadIdx.y][threadIdx.x] = in[i * N * N + j * N + k];
    }
    __syncthreads();
    if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
        if (threadIdx.z >= 1 && threadIdx.z < IN_TILE_DIM_SMALL - 1 && threadIdx.y >= 1 &&
            threadIdx.y < IN_TILE_DIM_SMALL - 1 && threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM_SMALL - 1) {
            out[i * N * N + j * N + k] = c0 * in_s[threadIdx.z][threadIdx.y][threadIdx.x] +
                                         c1 * in_s[threadIdx.z][threadIdx.y][threadIdx.x - 1] +
                                         c2 * in_s[threadIdx.z][threadIdx.y][threadIdx.x + 1] +
                                         c3 * in_s[threadIdx.z][threadIdx.y - 1][threadIdx.x] +
                                         c4 * in_s[threadIdx.z][threadIdx.y + 1][threadIdx.x] +
                                         c5 * in_s[threadIdx.z - 1][threadIdx.y][threadIdx.x] +
                                         c6 * in_s[threadIdx.z + 1][threadIdx.y][threadIdx.x];
        }
    }
}

void stencil_3d_parallel_shared_memory(float* in, float* out, unsigned int N, int c0, int c1, int c2, int c3, int c4,
                                       int c5, int c6) {
    float *d_in, *d_out;
    cudaError_t error;

    error = cudaMalloc((void**)&d_in, N * N * N * sizeof(float));
    if (error != cudaSuccess) {
        std::cout << "cudaMalloc d_in failed: " << cudaGetErrorString(error) << std::endl;
        return;
    }

    error = cudaMalloc((void**)&d_out, N * N * N * sizeof(float));
    if (error != cudaSuccess) {
        std::cout << "cudaMalloc d_out failed: " << cudaGetErrorString(error) << std::endl;
        return;
    }

    error = cudaMemcpy(d_in, in, N * N * N * sizeof(float), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        std::cout << "cudaMemcpy to device failed: " << cudaGetErrorString(error) << std::endl;
        return;
    }

    dim3 dimBlock(IN_TILE_DIM_SMALL, IN_TILE_DIM_SMALL, IN_TILE_DIM_SMALL);
    dim3 dimGrid(cdiv(N, OUT_TILE_DIM_SMALL), cdiv(N, OUT_TILE_DIM_SMALL), cdiv(N, OUT_TILE_DIM_SMALL));

    stencil_kernel_shared_memory<<<dimGrid, dimBlock>>>(d_in, d_out, N, c0, c1, c2, c3, c4, c5, c6);

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

__global__ void stencil_kernel_thread_coarsening(float* in, float* out, unsigned int N, int c0, int c1, int c2, int c3,
                                                 int c4, int c5, int c6) {
    int iStart = blockIdx.z * OUT_TILE_DIM_BIG;
    int j = blockIdx.y * OUT_TILE_DIM_BIG + threadIdx.y - 1;
    int k = blockIdx.x * OUT_TILE_DIM_BIG + threadIdx.x - 1;
    __shared__ float inPrev_s[IN_TILE_DIM_BIG][IN_TILE_DIM_BIG];
    __shared__ float inCurr_s[IN_TILE_DIM_BIG][IN_TILE_DIM_BIG];
    __shared__ float inNext_s[IN_TILE_DIM_BIG][IN_TILE_DIM_BIG];

    inPrev_s[threadIdx.y][threadIdx.x] = 0.0f;
    inCurr_s[threadIdx.y][threadIdx.x] = 0.0f;
    inNext_s[threadIdx.y][threadIdx.x] = 0.0f;

    if (iStart - 1 >= 0 && iStart - 1 < N && j >= 0 && j < N && k >= 0 && k < N) {
        inPrev_s[threadIdx.y][threadIdx.x] = in[(iStart - 1) * N * N + j * N + k];
    }
    if (iStart >= 0 && iStart < N && j >= 0 && j < N && k >= 0 && k < N) {
        inCurr_s[threadIdx.y][threadIdx.x] = in[iStart * N * N + j * N + k];
    }
    for (int i = iStart; i < iStart + OUT_TILE_DIM_BIG; ++i) {
        inNext_s[threadIdx.y][threadIdx.x] = 0.0f;
        if (i + 1 >= 0 && i + 1 < N && j >= 0 && j < N && k >= 0 && k < N) {
            inNext_s[threadIdx.y][threadIdx.x] = in[(i + 1) * N * N + j * N + k];
        }
        __syncthreads();
        if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
            if (threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM_BIG - 1 && threadIdx.x >= 1 &&
                threadIdx.x < IN_TILE_DIM_BIG - 1) {
                out[i * N * N + j * N + k] =
                    c0 * inCurr_s[threadIdx.y][threadIdx.x] + c1 * inCurr_s[threadIdx.y][threadIdx.x - 1] +
                    c2 * inCurr_s[threadIdx.y][threadIdx.x + 1] + c3 * inCurr_s[threadIdx.y - 1][threadIdx.x] +
                    c4 * inCurr_s[threadIdx.y + 1][threadIdx.x] + c5 * inPrev_s[threadIdx.y][threadIdx.x] +
                    c6 * inNext_s[threadIdx.y][threadIdx.x];
            }
        }
        __syncthreads();
        inPrev_s[threadIdx.y][threadIdx.x] = inCurr_s[threadIdx.y][threadIdx.x];
        inCurr_s[threadIdx.y][threadIdx.x] = inNext_s[threadIdx.y][threadIdx.x];
    }
}

void stencil_3d_parallel_thread_coarsening(float* in, float* out, unsigned int N, int c0, int c1, int c2, int c3,
                                           int c4, int c5, int c6) {
    float *d_in, *d_out;
    cudaError_t error;

    error = cudaMalloc((void**)&d_in, N * N * N * sizeof(float));
    if (error != cudaSuccess) {
        std::cout << "cudaMalloc d_in failed: " << cudaGetErrorString(error) << std::endl;
        return;
    }

    error = cudaMalloc((void**)&d_out, N * N * N * sizeof(float));
    if (error != cudaSuccess) {
        std::cout << "cudaMalloc d_out failed: " << cudaGetErrorString(error) << std::endl;
        return;
    }

    error = cudaMemcpy(d_in, in, N * N * N * sizeof(float), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        std::cout << "cudaMemcpy to device failed: " << cudaGetErrorString(error) << std::endl;
        return;
    }

    dim3 dimBlock(IN_TILE_DIM_BIG, IN_TILE_DIM_BIG, 1);
    dim3 dimGrid(cdiv(N, OUT_TILE_DIM_BIG), cdiv(N, OUT_TILE_DIM_BIG), cdiv(N, OUT_TILE_DIM_BIG));

    stencil_kernel_thread_coarsening<<<dimGrid, dimBlock>>>(d_in, d_out, N, c0, c1, c2, c3, c4, c5, c6);

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

__global__ void stencil_kernel_register_tiling(float* in, float* out, unsigned int N, int c0, int c1, int c2, int c3,
                                               int c4, int c5, int c6) {
    int iStart = blockIdx.z * OUT_TILE_DIM_SMALL;
    int j = blockIdx.y * OUT_TILE_DIM_SMALL + threadIdx.y - 1;
    int k = blockIdx.x * OUT_TILE_DIM_SMALL + threadIdx.x - 1;
    float inPrev;
    __shared__ float inCurr_s[IN_TILE_DIM_SMALL][IN_TILE_DIM_SMALL];
    float inCurr;
    float inNext;
    if (iStart - 1 >= 0 && iStart - 1 < N && j >= 0 && j < N && k >= 0 && k < N) {
        inPrev = in[(iStart - 1) * N * N + j * N + k];
    }

    if (iStart >= 0 && iStart < N && j >= 0 && j < N && k >= 0 && k < N) {
        inCurr = in[iStart * N * N + j * N + k];
        inCurr_s[threadIdx.y][threadIdx.x] = inCurr;
    }

    for (int i = iStart; i < iStart + OUT_TILE_DIM_SMALL; ++i) {
        if (i + 1 >= 0 && i + 1 < N && j >= 0 && j < N && k >= 0 && k < N) {
            inNext = in[(i + 1) * N * N + j * N + k];
        }

        __syncthreads();
        if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
            if (threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM_SMALL - 1 && threadIdx.x >= 1 &&
                threadIdx.x < IN_TILE_DIM_SMALL - 1) {
                out[i * N * N + j * N + k] = c0 * inCurr + c1 * inCurr_s[threadIdx.y][threadIdx.x - 1] +
                                             c2 * inCurr_s[threadIdx.y][threadIdx.x + 1] +
                                             c3 * inCurr_s[threadIdx.y + 1][threadIdx.x] +
                                             c4 * inCurr_s[threadIdx.y - 1][threadIdx.x] + c5 * inPrev + c6 * inNext;
            }
        }
        __syncthreads();
        inPrev = inCurr;
        inCurr = inNext;
        inCurr_s[threadIdx.y][threadIdx.x] = inNext;
    }
}

void stencil_3d_parallel_register_tiling(float* in, float* out, unsigned int N, int c0, int c1, int c2, int c3, int c4,
                                         int c5, int c6) {
    float *d_in, *d_out;
    cudaError_t error;

    error = cudaMalloc((void**)&d_in, N * N * N * sizeof(float));
    if (error != cudaSuccess) {
        std::cout << "cudaMalloc d_in failed: " << cudaGetErrorString(error) << std::endl;
        return;
    }

    error = cudaMalloc((void**)&d_out, N * N * N * sizeof(float));
    if (error != cudaSuccess) {
        std::cout << "cudaMalloc d_out failed: " << cudaGetErrorString(error) << std::endl;
        return;
    }

    error = cudaMemcpy(d_in, in, N * N * N * sizeof(float), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        std::cout << "cudaMemcpy to device failed: " << cudaGetErrorString(error) << std::endl;
        return;
    }

    dim3 dimBlock(IN_TILE_DIM_SMALL, IN_TILE_DIM_SMALL, 1);
    dim3 dimGrid(cdiv(N, OUT_TILE_DIM_SMALL), cdiv(N, OUT_TILE_DIM_SMALL), cdiv(N, OUT_TILE_DIM_SMALL));

    stencil_kernel_register_tiling<<<dimGrid, dimBlock>>>(d_in, d_out, N, c0, c1, c2, c3, c4, c5, c6);

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
