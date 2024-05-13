#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>


__global__
void matrixMulRowKernel(float* M, float* N, float* P, int size){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    //assuming a square matrix
    if (row < size){
        //do this for every element in the row:
        for (int i=0; i<size; ++i){
            float sum = 0;
            for (int j=0; j<size; ++j){
                sum += M[row * size + j] * N[j * size + row];
            }
            P[row * size + i] = sum;
        }
    }
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
  return (a + b - 1) / b;
}


torch::Tensor matrixRowMul(torch::Tensor M, torch::Tensor N){
    //for now we only support the square matrices
    assert(M.device().type() == torch::kCUDA && N.device().type() == torch::kCUDA);
    assert(M.dtype() == torch::kFloat32 && N.dtype() == torch::kFloat32);
    assert(M.size(0) == M.size(1) && N.size(0) == N.size(1) && M.size(0) == N.size(0));

    const auto size = M.size(0);
    auto P = torch::empty_like(N);

    dim3 dimBlock(16);
    dim3 dimGrid(cdiv(size, dimBlock.x));

    matrixMulRowKernel<<<dimBlock, dimGrid, 0,  torch::cuda::getCurrentCUDAStream()>>>(M.data_ptr<float>(), N.data_ptr<float>(), P.data_ptr<float>(), size);

    return P;
}