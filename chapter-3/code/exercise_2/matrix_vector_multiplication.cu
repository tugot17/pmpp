#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

__global__
void matrixVecMulKernel(float* B, float* c, float* result, int vector_size, int matrix_rows){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < matrix_rows){
        float sum = 0;
        for (int j=0; j < vector_size; ++j){
            sum += B[i * vector_size + j] * c[j];
        }
        result[i] = sum;
    }
}

torch::Tensor matrix_vector_multiplication(torch::Tensor B, torch::Tensor c){
    assert(B.device().type() == torch::kCUDA && c.device().type() == torch::kCUDA);
    assert(B.dtype() == torch::kFloat32 && c.dtype() == torch::kFloat32);
    assert(B.size(1) == c.size(0));

    int vector_size = c.size(0);
    int matrix_rows = B.size(0);
    
    auto result = torch::empty({matrix_rows}, torch::TensorOptions().dtype(torch::kFloat32).device(B.device()));

    // // Number of threads and blocks
    int threads_per_block = 16;
    int number_of_blocks = (matrix_rows + threads_per_block - 1) / threads_per_block;

    matrixVecMulKernel<<<number_of_blocks, threads_per_block,  0, torch::cuda::getCurrentCUDAStream()>>>(B.data_ptr<float>(), c.data_ptr<float>(), result.data_ptr<float>(), vector_size, matrix_rows);

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return result;
}
