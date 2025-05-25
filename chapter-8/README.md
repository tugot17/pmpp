# Chapter 8

## Code

For this chapter we implement all of the stencil kernels mentioned, in particular:
- sequential stencil
- basic parallelized stencil kernel
- stencil kernel utilizing shared memory
- stencil kernel utilizing thread coarsening
- stencil kernel utilizing register tiling

All of the kernels, alongside their host function code, can be found in [stencil.cu](./code/stencil.cu).

Additionally, we also introduce the benchmark comparing the execution speed of all of the kernels discussed. To run, use the provided `Makefile`.

```bash
cd code
make
```

and run

```bash
./stencil_benchmark
```

you will see

```logs
...
================================================================================
Benchmarking 3D Stencil Operations - Grid Size: 128x128x128
================================================================================
Configuration:
Grid size: 128x128x128
Total elements: 2097152
Memory per array: 8.00 MB
OUT_TILE_DIM: 8, IN_TILE_DIM: 8


Results:
Implementation           | Time (ms) | Speedup vs Sequential | Speedup vs Basic
-------------------------|-----------|----------------------|------------------
Sequential              |    4.634  |                1.00x |            0.76x
Parallel Basic          |    3.535  |                1.31x |            1.00x
Shared Memory           |    3.521  |                1.32x |            1.00x
Thread Coarsening       |    3.512  |                1.32x |            1.01x
Register Tiling         |    3.519  |                1.32x |            1.00x

Correctness Verification:
Parallel Basic vs Sequential: ✓ PASS
Shared Memory vs Sequential: ✓ PASS
Thread Coarsening vs Sequential: ✓ PASS
Register Tiling vs Sequential: ✓ PASS

Overall correctness: ✓ All implementations correct
```

### Heat simulation

We also explore the potential applications of stencil for real-world problems, and we implement a CUDA-accelerated heat diffusion simulation. The simulation leverages the `stencil_3d_parallel_register_tiling` kernel and computes the changes in heat over time.

To run it, first make sure that all of the CUDA code is already compiled. 

```bash
cd code
make
```

Then you can just run the code in [heat_simulation.py](./code/heat_simulation.py)

```bash
python heat_simulation.py
```

*Note that it might take a few minutes due to GIF being slowly created. 

The result should resemble this:

![our heat equation](./code/heat_equation_3d.gif)


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
dim3 dimBlock(OUT_TILE_DIM, OUT_TILE_DIM, OUT_TILE_DIM);
dim3 dimGrid(cdiv(N, dimBlock.x), cdiv(N, dimBlock.y), cdiv(N, dimBlock.z));
stencil_kernel<<<dimGrid, dimBlock>>>(d_in, d_out, N, c0, c1, c2, c3, c4, c5, c6);
```
We launch the kernel in Fig. 8.6. for every point in the input grid. We have `120 x 120 x 120` points, and we launch in blocks of `8 x 8 x 8`. Hence, we will have `120 x 120 x 120 / 8 x 8 x 8 -> 15 x 15 x 15 = 3375` blocks. 

**c. For the kernel with shared memory tiling in Fig. 8.8, what is the number of thread blocks that are needed, assuming a block size of `8 x 8 x 8`?**

```cpp
__global__ void stencil_kernel_shared_memory(float* in, float* out, unsigned int N,
                                           int c0, int c1, int c2, int c3, int c4, int c5, int c6) {
    int i = blockIdx.z*OUT_TILE_DIM+ threadIdx.z - 1;
    int j = blockIdx.y*OUT_TILE_DIM+ threadIdx.y - 1;
    int k = blockIdx.x*OUT_TILE_DIM+ threadIdx.x - 1;
    __shared__ float in_s[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];
    if(i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
        in_s[threadIdx.z][threadIdx.y][threadIdx.x] = in[i*N*N + j*N + k];
    }
    __syncthreads();
    if(i >= 1 && i < N-1 && j >= 1 && j < N-1 && k >= 1 && k < N-1) {
        if(threadIdx.z >= 1 && threadIdx.z < IN_TILE_DIM-1 && threadIdx.y >= 1
           && threadIdx.y<IN_TILE_DIM-1 && threadIdx.x>=1 && threadIdx.x<IN_TILE_DIM-1) {
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

We launch `cdiv(N, OUT_TILE_DIM) = cdiv(120, 6) = 20` blocks in every axis, so `20 x 20 x 20 = 8,000` blocks. 

**d. For the kernel with shared memory tiling and thread coarsening in Fig. 8.10, what is the number of thread blocks that are needed, assuming a block size of `32 x 32` ?**

```cpp
__global__ void stencil_kernel_thread_coarsening(float* in, float* out, unsigned int N,
                                                int c0, int c1, int c2, int c3, int c4, int c5, int c6) {
    int iStart = blockIdx.z*OUT_TILE_DIM;
    int j = blockIdx.y*OUT_TILE_DIM+ threadIdx.y - 1;
    int k = blockIdx.x*OUT_TILE_DIM+ threadIdx.x - 1;
    __shared__ float inPrev_s[IN_TILE_DIM][IN_TILE_DIM];
    __shared__ float inCurr_s[IN_TILE_DIM][IN_TILE_DIM];
    __shared__ float inNext_s[IN_TILE_DIM][IN_TILE_DIM];
    
    inPrev_s[threadIdx.y][threadIdx.x] = 0.0f;
    inCurr_s[threadIdx.y][threadIdx.x] = 0.0f;
    inNext_s[threadIdx.y][threadIdx.x] = 0.0f;
    
    if(iStart-1 >= 0 && iStart-1 < N && j >= 0 && j < N && k >= 0 && k < N) {
        inPrev_s[threadIdx.y][threadIdx.x] = in[(iStart - 1)*N*N + j*N + k];
    }
    if(iStart >= 0 && iStart < N && j >= 0 && j < N && k >= 0 && k < N) {
        inCurr_s[threadIdx.y][threadIdx.x] = in[iStart*N*N + j*N + k];
    }
    for(int i = iStart; i < iStart + OUT_TILE_DIM; ++i) {
        inNext_s[threadIdx.y][threadIdx.x] = 0.0f;
        if(i + 1 >= 0 && i + 1 < N && j >= 0 && j < N && k >= 0 && k < N) {
            inNext_s[threadIdx.y][threadIdx.x] = in[(i + 1)*N*N + j*N + k];
        }
        __syncthreads();
        if(i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
            if(threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM - 1
               && threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM - 1) {
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
stencil_kernel_thread_coarsening<<<dimGrid, dimBlock>>>(d_in, d_out, N, c0, c1, c2, c3, c4, c5, c6);
```

We launch a block of size `32 x 32`, meaning `IN_TILE_DIM = 32` and `OUT_TILE_DIM = IN_TILE_DIM - 2 = 32 - 2 = 30`.
Given that we launch `cdiv(N, OUT_TILE_DIM) = cdiv(120, 30) = 4` blocks in each direction. So `4x4x4=64` blocks in total. 

### Exercise 2

**Consider an implementation of a seven-point (3D) stencil with shared memory tiling and thread coarsening applied. The implementation is similar to those in Figs. 8.10 and 8.12, except that the tiles are not perfect cubes. Instead, a thread block size of `32 x 32` is used as well as a coarsening factor of 16 (i.e., each thread block processes 16 consecutive output planes in the z dimension).**

As a starting point, let's try to write such a kernel:

```cpp
#define OUT_TILE_DIM 30      
#define Z_COARSENING 16        
#define IN_TILE_DIM (OUT_TILE_DIM + 2)  // 32 (30 + 1 left + 1 right halo)

__global__ void stencil_7point_coarsened(
    float* in,float* out, int N,              
    float c0, float c1, float c2,float c3, float c4, float c5, float c6
) {
    int z_start = blockIdx.z * Z_COARSENING;      
    int global_y = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;  // Global y (with halo offset)
    int global_x = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;  // Global x (with halo offset)
    
    __shared__ float s_prev[IN_TILE_DIM][IN_TILE_DIM];  
    __shared__ float s_curr[IN_TILE_DIM][IN_TILE_DIM];  
    __shared__ float s_next[IN_TILE_DIM][IN_TILE_DIM];
    
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    
    s_prev[ty][tx] = 0.0f;
    s_curr[ty][tx] = 0.0f;
    s_next[ty][tx] = 0.0f;
    
    if (z_start - 1 >= 0 && z_start - 1 < N && 
        global_y >= 0 && global_y < N && 
        global_x >= 0 && global_x < N) {
        s_prev[ty][tx] = in[(z_start - 1) * N * N + global_y * N + global_x];
    }
    
    if (z_start >= 0 && z_start < N && 
        global_y >= 0 && global_y < N && 
        global_x >= 0 && global_x < N) {
        s_curr[ty][tx] = in[z_start * N * N + global_y * N + global_x];
    }
    
    for (int z_offset = 0; z_offset < Z_COARSENING; z_offset++) {
        int global_z = z_start + z_offset;
        
        s_next[ty][tx] = 0.0f;
        if (global_z + 1 >= 0 && global_z + 1 < N && 
            global_y >= 0 && global_y < N && 
            global_x >= 0 && global_x < N) {
            s_next[ty][tx] = in[(global_z + 1) * N * N + global_y * N + global_x];
        }
        
        __syncthreads();
        
        if (global_z >= 1 && global_z < N - 1 &&     
            global_y >= 1 && global_y < N - 1 &&     
            global_x >= 1 && global_x < N - 1 &&     
            ty >= 1 && ty < IN_TILE_DIM - 1 &&            
            tx >= 1 && tx < IN_TILE_DIM - 1) {            
            
            float result = c0 * s_curr[ty][tx] +
                          c1 * s_curr[ty][tx - 1] +
                          c2 * s_curr[ty][tx + 1] +
                          c3 * s_curr[ty - 1][tx] +
                          c4 * s_curr[ty + 1][tx] +
                          c5 * s_prev[ty][tx] +
                          c6 * s_next[ty][tx]; 
            
            out[global_z * N * N + global_y * N + global_x] = result;
        }
        __syncthreads();
        
        s_prev[ty][tx] = s_curr[ty][tx];
        s_curr[ty][tx] = s_next[ty][tx];
    }
}
...    

dim3 block_size(IN_TILE_DIM, IN_TILE_DIM, 1);
dim3 grid_size(cdiv(N, OUT_TILE_DIM), cdiv(N, OUT_TILE_DIM), cdiv(N, OUT_TILE_DIM));

stencil_7point_coarsened<<<grid_size, block_size>>>(
    d_in, d_out, N, c0, c1, c2, c3, c4, c5, c6
);
```
**a. What is the size of the input tile (in number of elements) that the thread block loads throughout its lifetime?**

Each loaded tile is of size `IN_TILE_DIM x IN_TILE_DIM = 32x32= 1024` elements. Since each thread block processes 16 consecutive output planes, and we have two extra planes for halo cells (one before and one after), it will load `1024 x 18 = 18432` elements throughout its lifetime.

**b. What is the size of the output tile (in number of elements) that the thread block processes throughout its lifetime?**

Each thread will write to the output tile of size `OUT_TILE_DIM x OUT_TILE_DIM = 30 x 30 = 900` elements. Since each thread block processes 16 consecutive output planes, it will process `900 x 16 = 14400` elements throughout its lifetime.

**c. What is the floating point to global memory access ratio (in OP/B) of the kernel?**

We load `18432` elements; since this is a stencil, each element being `4 bytes`, we load a total of `18432 x 4 = 73728` bytes. 

For every output element, we do 7 multiplications and 6 additions—13 FLOPs in total. Since we have a total of `14400` output elements, we do `14400 x 13 = 187200` floating-point operations.

So we have `(14400 x 13) / (18432 x 4) = 187200/73728 = 2.54` OP/B.

*Note that this includes reads only; we also have writes that are not accounted for in this example. 

**d. How much shared memory (in bytes) is needed by each thread block if register tiling is not used, as in Fig. 8.10?**

We store three consecutive tiles to be stored; each tile is `IN_TILE_DIM x IN_TILE_DIM` and consists of 4-bit numbers, so we need `3 x IN_TILE_DIM x IN_TILE_DIM x 4 = 3 x 32 x 32 x 4 = 12288` bytes.

**e. How much shared memory (in bytes) is needed by each thread block if register tiling is used, as in Fig. 8.12?**

If the register tiling optimization is being used, we only need to store a single output tile, so we will need `IN_TILE_DIM x IN_TILE_DIM x 4 = 32 x 32 x 4 = 4096` bytes in shared memory. Note that this will come at increased demand on registers. 
