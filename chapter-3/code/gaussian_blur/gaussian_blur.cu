#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>


__global__
void blur_kernel(unsigned char* Pin, unsigned char* Pout, int width, int height, int blur_size){

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col <= width && row <= height){
        //row major order 
        int grayOffset = row * width + col;
        
        for(int blurRow= -blur_size, blurRow < blur_size+1, ++blurRow){
            for(int blurCol= -blur_size, blurCol < blur_size+1, ++blurCol){

            }
        }
    }

}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
  return (a + b - 1) / b;
}


torch::Tensor gaussian_blur(torch::Tensor img){
    assert(img.device().type() == torch::kCUDA);
    assert(img.dtype() == torch::kByte);
    
    const auto height = img.size(0);
    const auto width = img.size(1);
    
    dim3 dimBlock(32, 32);
    dim3 dimGrid(cdiv(width, dimBlock.x), cdiv(height, dimBlock.y));
    
    auto result = torch::empty({height, width, 1}, torch::TensorOptions().dtype(torch::kByte).device(img.device()));

    rgtToGrayscaleKernel<<<dimGrid, dimBlock, 0, torch::cuda::getCurrentCUDAStream()>>>(img.data_ptr<unsigned char>(), result.data_ptr<unsigned char>(), width, height);
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return result;
}