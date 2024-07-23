#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

inline unsigned int cdiv(unsigned int a, unsigned int b) {
  return (a + b - 1) / b;
}


__global__
void naiveMatrixMulKernel(float* M, float* N, float* P, int m, int n, int o){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < m && col < o){
        float sum = 0;
        for (int i=0; i<n; ++i){
            //M: (m x n); N(n x o), get everything from the row and everything from the column
            sum += M[row * n + i] * N[i * o + col];
        }
        //the resulting one is m x o
        P[row * o + col] = sum;
    }
}

torch::Tensor naiveMatrixMul(torch::Tensor M, torch::Tensor N){
    assert(M.device().type() == torch::kCUDA && N.device().type() == torch::kCUDA);
    assert(M.dtype() == torch::kFloat32 && N.dtype() == torch::kFloat32);
    assert(M.size(1) == N.size(0));
    
    //matrices are m x n and n x o
    const auto m = M.size(0);
    const auto n = M.size(1);
    const auto o = N.size(1);

    auto P = torch::empty({m, o}, torch::TensorOptions().dtype(N.dtype()).device(N.device()));

    dim3 dimBlock(16, 16);
    dim3 dimGrid(cdiv(o, dimBlock.x), cdiv(m, dimBlock.y));


    naiveMatrixMulKernel<<<dimBlock, dimGrid, 0,  torch::cuda::getCurrentCUDAStream()>>>(M.data_ptr<float>(), N.data_ptr<float>(), P.data_ptr<float>(), m, n, o);

    return P;
}

#define TILE_WIDTH 16

__global__
void tiledSquareMatrixMulKernel(float* M, float* N, float* P, int width){
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockDim.x; int by = blockDim.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * TILE_WIDTH  + ty;
    int col = bx * TILE_WIDTH  + tx;

    float Pvalue = 0;
    for (int ph = 0; ph < width / TILE_WIDTH; ++ph){
        Mds[ty][tx] = M[row * width + ph * TILE_WIDTH + tx];
        Nds[ty][tx] = N[(ty + ph * TILE_WIDTH) * width  + col];
        __syncthreads();

        for (int k=0; k<TILE_WIDTH; ++k){
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }
    P[row * width + col] = Pvalue;
}

torch::Tensor tiledSquareMatrixMul(torch::Tensor M, torch::Tensor N){
    assert(M.device().type() == torch::kCUDA && N.device().type() == torch::kCUDA);
    assert(M.dtype() == torch::kFloat32 && N.dtype() == torch::kFloat32);
    assert(M.size(0) == M.size(1));
    assert(M.size(1) == N.size(0));
    assert(N.size(0) == N.size(1));
    
    //both matrices are m x m
    const auto m = M.size(0);
    assert(m % TILE_WIDTH == 0);

    auto P = torch::empty({m, m}, torch::TensorOptions().dtype(N.dtype()).device(N.device()));

    dim3 dimBlock(4, 4);
    dim3 dimGrid(cdiv(m, TILE_WIDTH), cdiv(m, TILE_WIDTH));


    tiledSquareMatrixMulKernel<<<dimBlock, dimGrid, 0,  torch::cuda::getCurrentCUDAStream()>>>(M.data_ptr<float>(), N.data_ptr<float>(), P.data_ptr<float>(), m);

    return P;
}