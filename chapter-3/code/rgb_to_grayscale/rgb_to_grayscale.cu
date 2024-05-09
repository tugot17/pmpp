#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>


__global__
void rgbToGrayscaleKernel(unsigned char* Pin, unsigned char* Pout, int width, int height){

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    const int CHANNELS = 3;

    if (col < width && row < height){
        //row major order 
        int grayOffset = row * width + col;
        int rgbOffset = grayOffset * CHANNELS;

        unsigned char r = Pin[rgbOffset];
        unsigned char g = Pin[rgbOffset + 1];
        unsigned char b = Pin[rgbOffset + 2];
        
        Pout[grayOffset] = 0.21*r + 0.71*g + 0.07*b;
    }

}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
  return (a + b - 1) / b;
}


torch::Tensor rgb_to_gray(torch::Tensor img){
    assert(img.device().type() == torch::kCUDA);
    assert(img.dtype() == torch::kByte);
    
    const auto height = img.size(0);
    const auto width = img.size(1);
    
    dim3 dimBlock(32, 32);
    dim3 dimGrid(cdiv(width, dimBlock.x), cdiv(height, dimBlock.y));
    
    auto result = torch::empty({height, width, 1}, torch::TensorOptions().dtype(torch::kByte).device(img.device()));

    rgbToGrayscaleKernel<<<dimGrid, dimBlock, 0, torch::cuda::getCurrentCUDAStream()>>>(img.data_ptr<unsigned char>(), result.data_ptr<unsigned char>(), width, height);
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return result;
}