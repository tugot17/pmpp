#include <stdio.h>
#include <cuda_runtime.h>

__global__
void vecMulKernel(float* A, float* B, float* C, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n){
        C[i] = A[i] * B[i];
    }
}

torch::Tensor vecMulDevice(torch::Tensor A, torch::Tensor B_h){
    assert(A.device().type() == torch::kCUDA && B.device().type() == torch::kCUDA);
    assert(A.dtype() == torch::float && B.dtype() == torch::float);
    assert(A.size(0) == B.size(0));

    int n = A.size(0);

    auto C = torch::empty({n,}, torch::TensorOptions().dtype(torch::float).device(A.device()));

    //invoke a kernel
    vecMulKernel<<<ceil(n/256.0), 256>>>(A.data_ptr<unsigned char>(), B.data_ptr<unsigned char>(), C.data_ptr<unsigned char>(), n);

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return C;
}

