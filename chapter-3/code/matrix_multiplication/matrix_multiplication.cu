#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>


__global__
void MatrixMulKernel(float* M, float* N, float* P, int m, int n, int o){
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

inline unsigned int cdiv(unsigned int a, unsigned int b) {
  return (a + b - 1) / b;
}


torch::Tensor matrixMul(torch::Tensor M, torch::Tensor N){
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


    MatrixMulKernel<<<dimBlock, dimGrid, 0,  torch::cuda::getCurrentCUDAStream()>>>(M.data_ptr<float>(), N.data_ptr<float>(), P.data_ptr<float>(), m, n, o);

    return P;
}