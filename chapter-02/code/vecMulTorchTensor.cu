#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

__global__ void vecMulKernel(float* A, float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] * B[i];
    }
}

torch::Tensor vector_multiplication(torch::Tensor A, torch::Tensor B) {
    assert(A.device().type() == torch::kCUDA && B.device().type() == torch::kCUDA);
    assert(A.dtype() == torch::kFloat32 && B.dtype() == torch::kFloat32);
    assert(A.size(0) == B.size(0));

    int n = A.size(0);
    auto C = torch::empty({n}, torch::TensorOptions().dtype(torch::kFloat32).device(A.device()));

    // // Number of threads and blocks
    int threads_per_block = 256;
    int number_of_blocks = (n + threads_per_block - 1) / threads_per_block;

    vecMulKernel<<<number_of_blocks, threads_per_block, 0, torch::cuda::getCurrentCUDAStream()>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), n);

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return C;
}
