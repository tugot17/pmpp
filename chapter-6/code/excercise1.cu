// nvcc -o exercise1 excercise1.cu

#include <iostream>
#include <cuda_runtime.h>
#include <iomanip>
#define TILE_WIDTH 2

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

__global__ void TransposeMatrixKernel(float *M, int m, int n){
    
    //idenfiy the row and col of the resulting matrix
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < m && col < n){
        //this goes to the register, we load from [k][i] and put it in [i][k]
        float value = M[row * n + col];
        __syncthreads(); // we do this to not override the values unnecessarily 
        M[col*m + row] = value;
    }
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
  return (a + b - 1) / b;
}

__global__ void TiledMatrixMulKernel(float *M, float *N, float *P, int m, int n, int o)
{
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int by = blockIdx.y;
    int bx = blockIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int row = by * TILE_WIDTH + ty;  // Row in final matrix P
    int col = bx * TILE_WIDTH + tx;  // Column in final matrix P

    float PValue = 0;
    
    for (int ph = 0; ph < (n + TILE_WIDTH - 1) / TILE_WIDTH; ph++) 
    {
        if (row < m && (ph * TILE_WIDTH + tx) < n)
            Mds[ty][tx] = M[row * n + ph * TILE_WIDTH + tx];
        else
            Mds[ty][tx] = 0.0f;

        if ((ph * TILE_WIDTH + ty) < n && col < o)
            Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * o + col];
        else
            Nds[ty][tx] = 0.0f;
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++)
        {
            PValue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }

    if (row < m && col < o)
        P[row * o + col] = PValue;
}

__global__ void TiledMatrixMulKernelColMajorOrder(float *M, float *N, float *P, int m, int n, int o)
{
    // We do corner turning: instead of accessing N row-by-row (which would cause non-coalesced memory accesses), 
    // we access it column-by-column to maintain coalescing. The shared memory tile then allows us to 
    // efficiently reorder these elements for the actual computation, avoiding the performance penalty 
    // of non-coalesced accesses in global memory
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int by = blockIdx.y;
    int bx = blockIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

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
             Nds[ty][tx] = N[col * n + (ph * TILE_WIDTH + ty)]; //n is row len in the transposed, so we need to jump col * n + find the right phase and a correct thread in phase.
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


void inPlaceMatrixTranspose(float *M, int m, int n){
    float *d_M;
    cudaMalloc((void **)&d_M, m * n * sizeof(float));
    cudaMemcpy(d_M, M, m* n * sizeof(float), cudaMemcpyHostToDevice);

    //this is a slightly hacky version, we run basically a single block, this won't scale to larger matrices
    dim3 dimBlock(n, m);
    dim3 dimGrid(cdiv(n, dimBlock.x), cdiv(m, dimBlock.y));

    TransposeMatrixKernel<<<dimGrid, dimBlock>>>(d_M, m, n);

    cudaMemcpy(M, d_M, m* n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_M);
}


void matrixMulTiling(float *M, float *N, float *P, int m, int n, int o, bool isRowMajor=true)
{   
    float *d_M, *d_N, *d_P;

    cudaMalloc((void **)&d_M, m * n * sizeof(float));
    cudaMalloc((void **)&d_N, n * o * sizeof(float));
    cudaMalloc((void **)&d_P, m * o * sizeof(float));

    cudaMemcpy(d_M, M, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, N, n * o * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((o + dimBlock.x - 1) / dimBlock.x, (m + dimBlock.y - 1) / dimBlock.y);


    if (isRowMajor)
        TiledMatrixMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, m, n, o);
    else
        TiledMatrixMulKernelColMajorOrder<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, m, n, o);

    cudaMemcpy(P, d_P, m * o * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
}


int main(int argc, char const *argv[])
{
    int m = 4;
    int n = 3;
    int o = 2;

    float *M = new float[m * n];
    float *N = new float[n * o];
    float *N_transposed = new float[o * n];

    float *P1 = new float[m * o];
    float *P2 = new float[m * o];

    for(int i = 0; i < m * n; i++) {
        M[i] = i; 
    }

    for(int i = 0; i < n*o; i++) {
        N[i] = i + 1; 
    }
    std::copy(N, N + (n * o), N_transposed);

    std::cout << "M:\n";
    printMatrix(M, m, n);
    std::cout << "\n";

    std::cout << "N:\n";
    printMatrix(N, n, o);
    std::cout << "\n";

    inPlaceMatrixTranspose(N_transposed, n, o);
    std::cout << "\nN transposed:\n";
    printMatrix(N_transposed, o, n);
    std::cout << "\n";


    std::cout << "First multiply:\n";
    matrixMulTiling(M, N, P1, m, n, o);
    printMatrix(P1, m, o);
    std::cout << "\n";

    std::cout << "Second (col major order) multiply:\n";
    matrixMulTiling(M, N_transposed, P2, m, n, o, false);
    printMatrix(P2, m, o);
    std::cout << "\n";


    delete[] M;
    delete[] N;
    delete[] N_transposed;
    delete[] P1;
    delete[] P2;

    return 0;
}



