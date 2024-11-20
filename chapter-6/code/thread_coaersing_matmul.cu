// nvcc -o thread_coaersing_matmul thread_coaersing_matmul.cu

#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <iomanip>
#define TILE_WIDTH 32
#define COARSE_FACTOR 4

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
    // Get actual L2 size via CUDA on first call of this function
    static int l2_clear_size = 0;
    static unsigned char* gpu_scratch_l2_clear = NULL;
    if (!gpu_scratch_l2_clear) {
        cudaDeviceGetAttribute(&l2_clear_size, cudaDevAttrL2CacheSize, 0);
        l2_clear_size *= 2; // just to be extra safe (cache is not necessarily strict LRU)
        gpuErrchk(cudaMalloc(&gpu_scratch_l2_clear, l2_clear_size));
    }
    // Clear L2 cache (this is run on every call unlike the above code)
    gpuErrchk(cudaMemset(gpu_scratch_l2_clear, 0, l2_clear_size));
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

__global__ void TiledMatrixMulKernelWithThreadCoarsening(float *M, float *N, float *P, int m, int n, int o)
{
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int by = blockIdx.y;
    int bx = blockIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    // we use this to identify the current P element
    int row = by * TILE_WIDTH + ty;
    int colStart = bx * TILE_WIDTH * COARSE_FACTOR + tx;

    float Pvalue[COARSE_FACTOR] = {0.0f};

    for (int ph = 0; ph < (n + TILE_WIDTH - 1) / TILE_WIDTH; ph++)
    {
        if (row < m && (ph * TILE_WIDTH + tx) < n)
            Mds[ty][tx] = M[row * n + ph * TILE_WIDTH + tx]; // row + phase + right row in a phase
        else
            Mds[ty][tx] = 0.0f;

        for (int c=0; c < COARSE_FACTOR; c++){
            int col = colStart + c * COARSE_FACTOR;
            
            if ((ph * TILE_WIDTH + ty) < n && (col < o))
                Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * o + col]; // col is from ty + phase + actual col in the phase
            else
                Nds[ty][tx] = 0.0f;    
            __syncthreads(); // make sure everything is loaded to both tile matrices for the consecutive c

            for (int k = 0; k < TILE_WIDTH; k++)
            {
                Pvalue[c] += Mds[ty][k] * Nds[k][tx];
            }
            __syncthreads();
        }
    }

    for (int c=0; c < COARSE_FACTOR; c++){
        int col = colStart + c * COARSE_FACTOR;
        if (row < m && col < o)
            P[row * o + col] = Pvalue[c];
    }
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

inline unsigned int cdiv(unsigned int a, unsigned int b) {
  return (a + b - 1) / b;
}


void matrixMulTilingWithThreadCoarsing(float *M, float *N, float *P, int m, int n, int o)
{
    float *d_M, *d_N, *d_P;

    cudaMalloc((void **)&d_M, m * n * sizeof(float));
    cudaMalloc((void **)&d_N, n * o * sizeof(float));
    cudaMalloc((void **)&d_P, m * o * sizeof(float));

    cudaMemcpy(d_M, M, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, N, n * o * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(cdiv(o, dimBlock.x * COARSE_FACTOR), cdiv(m, dimBlock.y));

    TiledMatrixMulKernelWithThreadCoarsening<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, m, n, o);

    cudaMemcpy(P, d_P, m * o * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
}
    
float benchmark(void (*func)(float *, float *, float *, int, int, int),
                float *M, float *N, float *P, int m, int n, int o,
                int warmup = 25, int reps = 100)
{
    for (int i = 0; i < warmup; ++i)
    {
        func(M, N, P, m, n, o);
    }

    cudaEvent_t iterStart, iterStop;
    cudaEventCreate(&iterStart);
    cudaEventCreate(&iterStop);

    float totalTime_ms = 0.0f;

    for (int i = 0; i < reps; ++i)
    {
        cudaEventRecord(iterStart);
        func(M, N, P, m, n, o);
        cudaEventRecord(iterStop);
        cudaEventSynchronize(iterStop);

        float iterTime = 0.0f;
        cudaEventElapsedTime(&iterTime, iterStart, iterStop);
        totalTime_ms += iterTime;

        clear_l2();    
    }

    cudaEventDestroy(iterStart);
    cudaEventDestroy(iterStop);

    return totalTime_ms / reps;
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
    int m = 4096, n = 4096, o = 4096;

    float *M = new float[m * n];
    float *N = new float[n * o];
    float *P1 = new float[m * o];
    float *P2 = new float[m * o];

    for (int i = 0; i < m * n; ++i)
        M[i] = static_cast<float>(1);
    for (int i = 0; i < n * o; ++i)
        N[i] = static_cast<float>(1.5);

    // Benchmark matrixMul function
    float avgTimeMatrixMulTiling = benchmark(matrixMulTilingWithThreadCoarsing, M, N, P1, m, n, o);
    std::cout << "Average time for matrixMulTilingWithThreadCoarsing: " << avgTimeMatrixMulTiling << " ms" << std::endl;

    float avgTimeMatrixMul = benchmark(matrixMul, M, N, P2, m, n, o);
    std::cout << "Average time for matrixMul: " << avgTimeMatrixMul << " ms" << std::endl;

    bool same = allclose(P1, P2, m, o);
    std::cout << "Outputs are " << (same ? "approximately the same" : "different") << std::endl;

    // printf("\n");
    // printMatrix(P1, m, o);
    // printf("\n");
    // printMatrix(P2, m, o);

    delete[] M;
    delete[] N;
    delete[] P1;
    delete[] P2;

    return 0;
}
