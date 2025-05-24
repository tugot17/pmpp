# Chapter 8

## Code

## Exercises

### Exercise 1
**Consider a 3D stencil computation on a grid of size `120 x 120 x 120`, including boundary cells.**

**a. What is the number of output grid points that is computed during each stencil sweep?**

`118 x 118 x 118 = 1,643,032` points.

**b. For the basic kernel in Fig. 8.6, what is the number of thread blocks that are needed, assuming a block size of `8 x 8 x 8`?**

```cpp
__global__ void stencil_kernel(float* in, float* out, unsigned int N,
                              int c0, int c1, int c2, int c3, int c4, int c5, int c6) {
    unsigned int i = blockIdx.z*blockDim.z + threadIdx.z;
    unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
        out[i*N*N + j*N + k] = c0*in[i*N*N + j*N + k]
                             + c1*in[i*N*N + j*N + (k - 1)]
                             + c2*in[i*N*N + j*N + (k + 1)]
                             + c3*in[i*N*N + (j - 1)*N + k]
                             + c4*in[i*N*N + (j + 1)*N + k]
                             + c5*in[(i - 1)*N*N + j*N + k]
                             + c6*in[(i + 1)*N*N + j*N + k];
    }
}

...
dim3 dimBlock(OUT_TILE_DIM_SMALL, OUT_TILE_DIM_SMALL, OUT_TILE_DIM_SMALL);
dim3 dimGrid(cdiv(N, dimBlock.x), cdiv(N, dimBlock.y), cdiv(N, dimBlock.z));
stencil_kernel<<<dimGrid, dimBlock>>>(d_in, d_out, N, c0, c1, c2, c3, c4, c5, c6);
```

We launch the kernel in Fig. 8.6. for every point in the input grid. We have `120 x 120 x 120` points and we lauch in blocks of `8 x 8 x 8`. Hence we will have `120 x 120 x 120 / 8 x 8 x 8 -> 15 x 15 x 15 = 3375` blocks. 

**c. For the kernel with shared memory tiling in Fig. 8.8, what is the number of thread blocks that are needed, assuming a block size of `8 x 8 x 8`?**

```cpp
__global__ void stencil_kernel_shared_memory(float* in, float* out, unsigned int N,
                                           int c0, int c1, int c2, int c3, int c4, int c5, int c6) {
    int i = blockIdx.z*OUT_TILE_DIM_SMALL+ threadIdx.z - 1;
    int j = blockIdx.y*OUT_TILE_DIM_SMALL+ threadIdx.y - 1;
    int k = blockIdx.x*OUT_TILE_DIM_SMALL+ threadIdx.x - 1;
    __shared__ float in_s[IN_TILE_DIM_SMALL][IN_TILE_DIM_SMALL][IN_TILE_DIM_SMALL];
    if(i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
        in_s[threadIdx.z][threadIdx.y][threadIdx.x] = in[i*N*N + j*N + k];
    }
    __syncthreads();
    if(i >= 1 && i < N-1 && j >= 1 && j < N-1 && k >= 1 && k < N-1) {
        if(threadIdx.z >= 1 && threadIdx.z < IN_TILE_DIM_SMALL-1 && threadIdx.y >= 1
           && threadIdx.y<IN_TILE_DIM_SMALL-1 && threadIdx.x>=1 && threadIdx.x<IN_TILE_DIM_SMALL-1) {
            out[i*N*N + j*N + k] = c0*in_s[threadIdx.z][threadIdx.y][threadIdx.x]
                                 + c1*in_s[threadIdx.z][threadIdx.y][threadIdx.x-1]
                                 + c2*in_s[threadIdx.z][threadIdx.y][threadIdx.x+1]
                                 + c3*in_s[threadIdx.z][threadIdx.y-1][threadIdx.x]
                                 + c4*in_s[threadIdx.z][threadIdx.y+1][threadIdx.x]
                                 + c5*in_s[threadIdx.z-1][threadIdx.y][threadIdx.x]
                                 + c6*in_s[threadIdx.z+1][threadIdx.y][threadIdx.x];
        }
    }
}
...
dim3 dimBlock(IN_TILE_DIM, IN_TILE_DIM, IN_TILE_DIM);
dim3 dimGrid(cdiv(N, OUT_TILE_DIM), cdiv(N, OUT_TILE_DIM), cdiv(N, OUT_TILE_DIM));
stencil_kernel_shared_memory<<<dimGrid, dimBlock>>>(d_in, d_out, N, c0, c1, c2, c3, c4, c5, c6);
```

We launch blocks of `8 x 8 x 8`, but for this kernel we launch block size `IN_TILE_DIM`. The `OUT_TILE_DIM` will be `IN_TILE_DIM - 2 = 8-2=6`

We launch `cdiv(N, OUT_TILE_DIM_SMALL) = cdiv(120, 6) = 20` blocks in every exis so `20 x 20 x 20 = 8,000` blocks. 

**d. For the kernel with shared memory tiling and thread coarsening in Fig. 8.10, what is the number of thread blocks that are needed, assuming a block size of `32 x 32` ?**

```cpp
__global__ void stencil_kernel_thread_coarsening(float* in, float* out, unsigned int N,
                                                int c0, int c1, int c2, int c3, int c4, int c5, int c6) {
    int iStart = blockIdx.z*OUT_TILE_DIM_BIG;
    int j = blockIdx.y*OUT_TILE_DIM_BIG+ threadIdx.y - 1;
    int k = blockIdx.x*OUT_TILE_DIM_BIG+ threadIdx.x - 1;
    __shared__ float inPrev_s[IN_TILE_DIM_BIG][IN_TILE_DIM_BIG];
    __shared__ float inCurr_s[IN_TILE_DIM_BIG][IN_TILE_DIM_BIG];
    __shared__ float inNext_s[IN_TILE_DIM_BIG][IN_TILE_DIM_BIG];
    
    inPrev_s[threadIdx.y][threadIdx.x] = 0.0f;
    inCurr_s[threadIdx.y][threadIdx.x] = 0.0f;
    inNext_s[threadIdx.y][threadIdx.x] = 0.0f;
    
    if(iStart-1 >= 0 && iStart-1 < N && j >= 0 && j < N && k >= 0 && k < N) {
        inPrev_s[threadIdx.y][threadIdx.x] = in[(iStart - 1)*N*N + j*N + k];
    }
    if(iStart >= 0 && iStart < N && j >= 0 && j < N && k >= 0 && k < N) {
        inCurr_s[threadIdx.y][threadIdx.x] = in[iStart*N*N + j*N + k];
    }
    for(int i = iStart; i < iStart + OUT_TILE_DIM_BIG; ++i) {
        inNext_s[threadIdx.y][threadIdx.x] = 0.0f;
        if(i + 1 >= 0 && i + 1 < N && j >= 0 && j < N && k >= 0 && k < N) {
            inNext_s[threadIdx.y][threadIdx.x] = in[(i + 1)*N*N + j*N + k];
        }
        __syncthreads();
        if(i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
            if(threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM_BIG - 1
               && threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM_BIG - 1) {
                out[i*N*N + j*N + k] = c0*inCurr_s[threadIdx.y][threadIdx.x]
                                     + c1*inCurr_s[threadIdx.y][threadIdx.x-1]
                                     + c2*inCurr_s[threadIdx.y][threadIdx.x+1]
                                     + c3*inCurr_s[threadIdx.y-1][threadIdx.x]
                                     + c4*inCurr_s[threadIdx.y+1][threadIdx.x]
                                     + c5*inPrev_s[threadIdx.y][threadIdx.x]
                                     + c6*inNext_s[threadIdx.y][threadIdx.x];
            }
        }
        __syncthreads();
        inPrev_s[threadIdx.y][threadIdx.x] = inCurr_s[threadIdx.y][threadIdx.x];
        inCurr_s[threadIdx.y][threadIdx.x] = inNext_s[threadIdx.y][threadIdx.x];
    }
}
...
dim3 dimBlock(IN_TILE_DIM, IN_TILE_DIM, 1);
dim3 dimGrid(cdiv(N, OUT_TILE_DIM), cdiv(N, OUT_TILE_DIM), cdiv(N, OUT_TILE_DIM));
```

We launch block of size `32 x 32`, meaning `IN_TILE_DIM = 32` and `OUT_TILE_DIM = IN_TILE_DIM - 2 = 32 - 2 = 30`.
Given that we launch `cdiv(N, OUT_TILE_DIM) = cdiv(120, 30) = 4` blocks in each direction. So `4x4x4=64` blocks in total. 

### Exercise 2

**Consider an implementation of a seven-point (3D) stencil with shared memory tiling and thread coarsening applied. The implementation is similar to those in Figs. 8.10 and 8.12, except that the tiles are not perfect cubes. Instead, a thread block size of 32 3 32 is used as well as a coarsening factor of 16 (i.e., each thread block processes 16 consecutive output planes in the z dimension).**

**a. What is the size of the input tile (in number of elements) that the thread block loads throughout its lifetime?**

**b. What is the size of the output tile (in number of elements) that the thread block processes throughout its lifetime?**

**c. What is the floating point to global memory access ratio (in OP/B) of the kernel?**

**d. How much shared memory (in bytes) is needed by each thread block if register tiling is not used, as in Fig. 8.10?**

**e. How much shared memory (in bytes) is needed by each thread block if register tiling is used, as in Fig. 8.12?**