// nvcc -o matrix_mul matrix_mul_benchmark.cu

#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <iomanip>
#define TILE_WIDTH 64

__device__ void printDeviceMatrix(float *matrix, int width, int height)
{
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            printf("%f ", matrix[i * width + j]);
        }
        printf("\n");
    }
    printf("\n");
}

__global__ void MatrixMulKernel(float *M, float *N, float *P, int m, int n, int o)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < m && col < o)
    {
        float sum = 0;
        for (int i = 0; i < n; ++i)
        {
            sum += M[row * n + i] * N[i * o + col];
        }
        P[row * o + col] = sum;
    }
}

__global__ void TiledMatrixMulKernel(float *M, float *N, float *P, int m, int n, int o)
{

    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    // let's save these for convinience
    int by = blockIdx.y;
    int bx = blockIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    // we use this to identify the current P element
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float PValue = 0;
    for (int ph = 0; ph < (n + TILE_WIDTH - 1) / TILE_WIDTH; ph++)
    {
        if (row < m && (ph * TILE_WIDTH + tx) < n)
            Mds[ty][tx] = M[row * n + ph * TILE_WIDTH + tx]; // row + phase + right row in a phase
        else
            Mds[ty][tx] = 0.0f;

        if ((ph * TILE_WIDTH + ty) < n && (col < o))
            Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * o + col]; // col is from ty + phase + actuall col in the phase
        else
            Nds[ty][tx] = 0.0f;

        __syncthreads(); // make sure everything is loaded to both tile matrices

        for (int k = 0; k < TILE_WIDTH; k++)
        {
            PValue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads(); // make sure we update this for every thread and we can start overwriting
    }

    if (row < m && col < o)
        P[row * o + col] = PValue;
}

void matrixMul(float *M, float *N, float *P, int m, int n, int o)
{
    float *d_M, *d_N, *d_P;

    cudaMalloc((void **)&d_M, m * n * sizeof(float));
    cudaMalloc((void **)&d_N, n * o * sizeof(float));
    cudaMalloc((void **)&d_P, m * o * sizeof(float));

    cudaMemcpy(d_M, M, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, N, n * o * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((o + dimBlock.x - 1) / dimBlock.x, (m + dimBlock.y - 1) / dimBlock.y);

    MatrixMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, m, n, o);

    cudaMemcpy(P, d_P, m * o * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
}

void matrixMulTiling(float *M, float *N, float *P, int m, int n, int o)
{
    float *d_M, *d_N, *d_P;

    cudaMalloc((void **)&d_M, m * n * sizeof(float));
    cudaMalloc((void **)&d_N, n * o * sizeof(float));
    cudaMalloc((void **)&d_P, m * o * sizeof(float));

    cudaMemcpy(d_M, M, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, N, n * o * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((o + dimBlock.x - 1) / dimBlock.x, (m + dimBlock.y - 1) / dimBlock.y);

    TiledMatrixMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, m, n, o);

    cudaMemcpy(P, d_P, m * o * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
}

float benchmark(void (*func)(float *, float *, float *, int, int, int),
                float *M, float *N, float *P, int m, int n, int o,
                int warmup = 25, int reps = 100)
{
    // Warmup
    for (int i = 0; i < warmup; ++i)
    {
        func(M, N, P, m, n, o);
    }

    // Timing with CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start timing
    cudaEventRecord(start);
    for (int i = 0; i < reps; ++i)
    {
        func(M, N, P, m, n, o);
    }
    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Return the average time per run
    return milliseconds / reps;
}

bool allclose(float *M, float *N, int m, int n, float tol = 1e-5)
{
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            if (fabs(M[i * n + j] - N[i * n + j]) > tol)
                return false;
    return true;
}

void printMatrix(float *matrix, int rows, int cols)
{
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            std::cout << std::setw(6) << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main()
{
    // change these to experiment with sizes, here I get a substantial boost just via using TILING
    int m = 1271, n = 8771, o = 1831;

    float *M = new float[m * n];
    float *N = new float[n * o];
    float *P1 = new float[m * o];
    float *P2 = new float[m * o];

    for (int i = 0; i < m * n; ++i)
        M[i] = static_cast<float>(1);
    for (int i = 0; i < n * o; ++i)
        N[i] = static_cast<float>(1.5);

    // Benchmark matrixMul function
    float avgTimeMatrixMulTiling = benchmark(matrixMulTiling, M, N, P1, m, n, o);
    std::cout << "Average time for matrixMulTiling: " << avgTimeMatrixMulTiling << " ms" << std::endl;

    float avgTimeMatrixMul = benchmark(matrixMul, M, N, P2, m, n, o);
    std::cout << "Average time for matrixMul: " << avgTimeMatrixMul << " ms" << std::endl;

    bool same = allclose(P1, P2, m, o);
    std::cout << "Outputs are " << (same ? "approximately the same" : "different") << std::endl;

    // if (true && !same)
    // {
    //     std::cout << "\nMatrix P1 (from matrixMulTiling):" << std::endl;
    //     printMatrix(P1, m, o);

    //     std::cout << "\nMatrix P2 (from matrixMul):" << std::endl;
    //     printMatrix(P2, m, o);
    // }

    delete[] M;
    delete[] N;
    delete[] P1;
    delete[] P2;

    return 0;
}
