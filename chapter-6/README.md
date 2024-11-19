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
Coalased, within a block we access the neighbouring threads (`+ threadIdx.x` in `l01`) so all accesses to the global memory will be coalesced.

- **b. The access to array a_s of line 05**
`a_s` is a shared memory so it does not require coalasing. 

- **c. The access to array b of line 07**


- **d. The access to array c of line 07**
- **e. The access to array bc_s of line 07**
- **f. The access to array a_s of line 10**
- **g. The access to array d of line 10**
- **h. The access to array bc_s of line 11**
- **i. The access to array e of line 11**


### Exercise 4

What is the floating point to global memory access ratio (in OP/B) of each of the following matrix-matrix multiplication kernels?
a. The simple kernel described in Chapter 3, Multidimensional Grids and Data, without any optimizations applied.
b. The kernel described in Chapter 5, Memory Architecture and Data Locality, with shared memory tiling applied using a tile size of 32 × 32.
c. The kernel described in this chapter with shared memory tiling applied using a tile size of 32 × 32 and thread coarsening applied using a coarsening factor of 4.