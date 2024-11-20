// nvcc -o matrix_mul_dynamic_tile matrix_mul_with_optimal_dynamic_tile_size.cu
// here we make the tile dynamic, the tile size is calculated based on the hardware specyfication not hardcodedd as before

#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <iomanip>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <iostream>

int calculateOptimalTileWidth(int m, int n, int o)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    // Get hardware limits
    int maxThreadsPerBlock = prop.maxThreadsPerBlock;
    int maxBlockDimX = prop.maxThreadsDim[0];
    int maxBlockDimY = prop.maxThreadsDim[1];
    int sharedMemPerBlock = prop.sharedMemPerBlock;

    // Calculate maximum possible tile size based on hardware constraints

    // 1. Based on max threads per block (square tiles)
    int tileWidth = static_cast<int>(sqrt(maxThreadsPerBlock));

    // 2. Based on max block dimensions
    tileWidth = std::min(tileWidth, std::min(maxBlockDimX, maxBlockDimY));

    // 3. Based on shared memory (we need 2 tiles worth of shared memory)
    int maxTileWidthBySharedMem = static_cast<int>(sqrt(sharedMemPerBlock / (2 * sizeof(float))));
    tileWidth = std::min(tileWidth, maxTileWidthBySharedMem);

    // 4. Based on matrix dimensions (no point in having tiles larger than matrices)
    tileWidth = std::min(tileWidth, std::min(m, std::min(n, o)));

    // 5. Round down to nearest power of 2 for better memory alignment
    tileWidth = 1 << static_cast<int>(log2(tileWidth));

    // 6. Ensure minimum practical size
    tileWidth = std::max(16, tileWidth); // minimum tile size of 16

    // Print diagnostic information
    // std::cout << "Calculated optimal tile width: " << tileWidth << std::endl;
    // std::cout << "Based on:" << std::endl;
    // std::cout << "- Max threads per block: " << maxThreadsPerBlock << std::endl;
    // std::cout << "- Max block dimensions: " << maxBlockDimX << "x" << maxBlockDimY << std::endl;
    // std::cout << "- Shared memory per block: " << sharedMemPerBlock << " bytes" << std::endl;
    // std::cout << "- Matrix dimensions: " << m << "x" << n << "x" << o << std::endl;

    return tileWidth;
}

__device__ void printDeviceMatrix(float *matrix, int width, int height, const char *matrixName)
{
    printf("%s:\n", matrixName);
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
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

__global__ void TiledMatrixMulKernel(float *M, float *N, float *P, int m, int n, int o, int tileWidth)
{
    extern __shared__ float sharedMem[];
    // Split shared memory into two parts, one for Mds and one for Nds
    float *Mds = sharedMem;
    float *Nds = &sharedMem[tileWidth * tileWidth];

    // let's save these for convenience
    int by = blockIdx.y;
    int bx = blockIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    // we use this to identify the current P element
    int row = by * tileWidth + ty;
    int col = bx * tileWidth + tx;

    float PValue = 0.0;
    for (int ph = 0; ph < (n + tileWidth - 1) / tileWidth; ph++)
    {
        if (row < m && (ph * tileWidth + tx) < n)
            Mds[ty * tileWidth + tx] = M[row * n + ph * tileWidth + tx]; // row + phase + right row in a phase
        else
            Mds[ty * tileWidth + tx] = 0.0f;

        if ((ph * tileWidth + ty) < n && (col < o))
            Nds[ty * tileWidth + tx] = N[(ph * tileWidth + ty) * o + col]; // col is from ty + phase + actual col in the phase
        else
            Nds[ty * tileWidth + tx] = 0.0f;

        __syncthreads(); // make sure everything is loaded to both tile matrices

        for (int k = 0; k < tileWidth; k++)
        {
            PValue += Mds[ty * tileWidth + k] * Nds[k * tileWidth + tx];
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

    dim3 dimBlock(16, 16);
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
    // int tileWidth = calculateOptimalTileWidth();
    int tileWidth = calculateOptimalTileWidth(m, n, o);

    // for now we work just with the square matrices
    // int width = m;

    cudaMalloc((void **)&d_M, m * n * sizeof(float));
    cudaMalloc((void **)&d_N, n * o * sizeof(float));
    cudaMalloc((void **)&d_P, m * o * sizeof(float));

    cudaMemcpy(d_M, M, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, N, n * o * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock(tileWidth, tileWidth);
    dim3 dimGrid((o + dimBlock.x - 1) / dimBlock.x, (m + dimBlock.y - 1) / dimBlock.y);

    int sharedMemSize = 2 * tileWidth * tileWidth * sizeof(float);
    TiledMatrixMulKernel<<<dimGrid, dimBlock, sharedMemSize>>>(d_M, d_N, d_P, m, n, o, tileWidth);

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
    int m = 2010, n = 3200, o = 9111;

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

    if (true && !same)
    {
        std::cout << "\nMatrix P1 (from matrixMulTiling):" << std::endl;
        printMatrix(P1, m, o);

        std::cout << "\nMatrix P2 (from matrixMul):" << std::endl;
        printMatrix(P2, m, o);
    }

    delete[] M;
    delete[] N;
    delete[] P1;
    delete[] P2;

    return 0;
}