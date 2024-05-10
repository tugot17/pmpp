#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>


__global__
void blur_kernel(unsigned char* Pin, unsigned char* Pout, int width, int height, int blur_size){

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int channel = threadIdx.z;
    int baseOffset = channel * height * width;

    if (col < width && row < height){
        //row major order 
        int grayOffset = row * width + col;

        int pixelValues = 0;
        int pixels = 0;
        
        for(int blurRow= -blur_size; blurRow < blur_size+1; ++blurRow){
            for(int blurCol= -blur_size; blurCol < blur_size+1; ++blurCol){
                int currCol = col + blurCol;
                int currRow = row + blurRow;

                if (currCol >= 0 && currCol < width && currRow >= 0 && currRow < height){
                    pixelValues += Pin[baseOffset + currRow * width + currCol];
                    ++pixels;
                }
            }
        }
        Pout[baseOffset + row * width + col] = (unsigned char)(pixelValues/pixels);
    }

}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
  return (a + b - 1) / b;
}


torch::Tensor gaussian_blur(torch::Tensor img, int blurSize){
    assert(img.device().type() == torch::kCUDA);
    assert(img.dtype() == torch::kByte);
    
    const auto channels = img.size(0);
    const auto height = img.size(1);
    const auto width = img.size(2);
    

    dim3 dimBlock(16, 16, channels);
    dim3 dimGrid(cdiv(width, dimBlock.x), cdiv(height, dimBlock.y));
    
    // auto result = torch::empty_like(img, torch::TensorOptions().dtype(torch::kByte));
    auto result = torch::empty_like(img);

    blur_kernel<<<dimGrid, dimBlock, 0, torch::cuda::getCurrentCUDAStream()>>>(img.data_ptr<unsigned char>(), result.data_ptr<unsigned char>(), width, height, blurSize);
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return result;
}