# Chapter 6

## Code

## Exercises

### Exercise 1

**Write a matrix multiplication kernel function that corresponds to the design illustrated in Fig. 6.4.**

Just the kernel:

```cpp
__global__ void TiledMatrixMulKernelColMajorOrder(float *M, float *N, float *P, int m, int n, int o)
{
    // We do corner turning: instead of accessing N row-by-row (which would cause non-coalesced memory accesses), 
    // we access it column-by-column to maintain coalescing. The shared memory tile then allows us to 
    // efficiently reorder these elements for the actual computation, avoiding the performance penalty 
    // of non-coalesced accesses in global memory
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int by = blockIdx.y;
    int bx = blockIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float PValue = 0;
    for (int ph = 0; ph < (n + TILE_WIDTH - 1) / TILE_WIDTH; ph++)
    {
        if (row < m && (ph * TILE_WIDTH + tx) < n)
            Mds[ty][tx] = M[row * n + ph * TILE_WIDTH + tx]; // row + phase + right row in a phase
        else
            Mds[ty][tx] = 0.0f;

        if ((ph * TILE_WIDTH + ty) < n && (col < o))
             Nds[ty][tx] = N[col * n + (ph * TILE_WIDTH + ty)]; //n is row len in the transposed, so we need to jump col * n + find the right phase and a correct thread in phase.
        else
            Nds[ty][tx] = 0.0f;

        __syncthreads(); // make sure everything is loaded to both tile matrices
        for (int k = 0; k < TILE_WIDTH; k++)
        {
            PValue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads(); // make sure we update this for every thread and we can start overwriting
    }
    if (row < m && col < o)
        P[row * o + col] = PValue;
}
```

The full code can be found in [excercise1.cu](code/excercise1.cu)

### Exercise 2

**For tiled matrix multiplication, of the possible range of values for BLOCK_SIZE, for what values of BLOCK_SIZE will the kernel completely avoid uncoalesced accesses to global memory? (You need to consider only square blocks.)**

Coalesced access requires that all of the threads in the warp access the neighboring cells. Warp is 32 neighbouring threads. If the block size is `<32`, then the warp will span multiple rows and access the data from multiple consecutive rows. In this case, some of the calls to the global memory will be uncoalesced. So the answer is that `BLOCK_SIZES` should be multiples of `32`, so `32` or `64`, etc. In practice we are usually bound by the shared memory size, so going beyond `64` is unlikely. 

### Exercise 3

**Consider the following CUDA kernel:**

```cpp
01  **global** void foo_kernel(float* a, float* b, float* c, float* d, float* e) {
02  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
03  **shared** float a_s[256];
04  **shared** float bc_s[4*256];
05  a_s[threadIdx.x] = a[i];
06  for(unsigned int j = 0; j < 4; ++j) {
07      bc_s[j*256 + threadIdx.x] = b[j*blockDim.x*gridDim.x + i] + c[i*4 + j];
08  }
09  __syncthreads();
10 d[i + 8] = a_s[threadIdx.x];
11 e[i*8] = bc_s[threadIdx.x*4];
12 }
```


**For each of the following memory accesses, specify whether they are coalesced or uncoalesced or coalescing is not applicable:**

- **a. The access to array a of line 05**
`a[blockIdx.x*blockDim.x + threadIdx.x]` the neighbouring threads within a block will access the neighbouring memory cells (subsequent `threadIdx.x`) - so coalesced access. 

- **b. The access to array a_s of line 05**
`a_s` is a shared memory, so it does not require coalescing. 

- **c. The access to array b of line 07**
`b[j*blockDim.x*gridDim.x + blockIdx.x*blockDim.x + threadIdx.x;]`For neighbouring threads within a block, only the `threadIdx.x` changes, so all threads in the block access the neighbouring memory cells—coalesced.

- **d. The access to array c of line 07**
`c[(blockIdx.x*blockDim.x + threadIdx.x)*4 + j]` - jump of 4 cells between the neighboring cells—uncoalesced access. 

- **e. The access to array bc_s of line 07**
`bc_s` is a shared memory, therefore the concept of coalescing does not apply here. 

- **f. The access to array a_s of line 10**
`a_s` is a shared memory, therefore the concept of coalescing does not apply here. 

- **g. The access to array d of line 10**
`d[blockIdx.x*blockDim.x + threadIdx.x + 8]` - based on `threadIdx.x`, neighbouring threads in the warp access neighbouring cells—coalesced memory access.

- **h. The access to array bc_s of line 11**
`bc_s[threadIdx.x*4]` - `bc_s` is a shared memory, therefore the concept of coalescing does not apply here. 

- **i. The access to array e of line 11**
`e[(blockIdx.x*blockDim.x + threadIdx.x)*8]` - for neighbouring threads, it does a jump by 8 cells—no coalescing.

### Exercise 4

**What is the floating point to global memory access ratio (in OP/B) of each of the following matrix-matrix multiplication kernels?**

Let's assume we have a matrix `M` of size `(m, n)` and a matrix N of size `(n, o)`. We assume `float32`, so 4 bytes, as a data type. 

**a. The simple kernel described in Chapter 3, Multidimensional Grids and Data, without any optimizations applied.**

In the simple kernel, we, for each element of the result matrix, we:
- Loaded an entire row of the input matrix `M`, so `n` memory loads
- Load an entire row of the input matrix N, so `n` memory loads. 
- Multiply each of the elements from the fow so `n` operations.
- Add all of the results to each other to get the final number, so `n-1` or approximately `n` operations. 

In total `2n` memory loads and `2n` operations, so `4bytes` so `2n operations / (4bytes x 2n)=0.25 operations/B`.

**b. The kernel described in Chapter 5, Memory Architecture and Data Locality, with shared memory tiling applied using a tile size of 32 × 32.**

Let's assume we have a matrix `M` of size `(m, n)` and a matrix N of size `(n, o)`. We assume `float32`, so 4 bytes, as a data type. 
In the kernel with tiling, we still have a single thread for each of the elements of the output matrix, but now we have somehow reduced the amount of work to be done. Now we:
- Each thread loads only `n/TILE_SIZE` or `n/32` values from the global memory from matrix `M`; the other consecutive 31 values are loaded by the consecutive threads in the warp. Note that the `TILE_SIZE` is exactly the size of a warp, so all of the memory access is coalesced. 
- Same for the col (though the coalesced access works slightly differently), so also `n/32` global memory accesses. 
- The number of multiplications and additions remains the same as before, so `n` and `n-1` so `~2n` operations. 

So `2n operations / (4 bytes * 2n / 32) = 8 operations/B` - so much much better (32x better to be precise). 


**c. The kernel described in this chapter with shared memory tiling applied using a tile size of 32 × 32 and thread coarsening applied using a coarsening factor of 4.**

In the kernel with tiling and coarsing, we get some further improvement for memory loading. 

- For the first matrix (`M`), we do `n/32` loads from the global memory as before. Previously we had to repeat this for every single block. Previously the tile was only used in a single block and had to be reaccessed in all other blocks. Now we are reusing this tile `coarsening factor` times, or `4` times in this case. Decreasing the memory access to `n/32/4` = `n/128`.
- For the second matrix (`N`), everything stays as it was, so `n/32` mem accesses. 
- The number of multiplications and additions remains the same as before, so `n` and `n-1` so `~2n` operations. 

So overall, `2n operations / (4 bytes * (n/32 + n/128)) = 2n operations / (4B * (5n/128)) = 12.8 Operations/B`. Even better OP/B ratio than before.