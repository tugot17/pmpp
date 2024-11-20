# Chapter 5

## Code

In this chapter we implemented the tiled matrix multiplication algorithm. To run the experiments just execute:

```bash
cd code
nvcc -o matrix_mul matrix_mul_benchmark.cu
./matrix_mul
```

We run a bunch of speed comparison benchmarks as well as safety check for numerical parity. 


```bash
Average time for matrixMulTiling: 53.6356 ms
Average time for matrixMul: 57.2734 ms
Outputs are approximately the same
```


## Exercises

### Exercise 1

**Consider matrix addition. Can one use shared memory to reduce the global memory bandwidth consumption? Hint: Analyze the elements that are accessed by each thread and see whether there is any commonality between threads.**

In matrix addition, there is no reuse of values between threads. Thread `(0, 0)` is loading the value `M[0, 0]` and `N[0, 0]`, thread `(124, 12)` is loading the value `M[124, 12]` and `N[124, 12]`, etc. Since there is no reusage of values across the threads in the block, we can't use shared memory to reduce the memory bandwidth consumption.

### Exercise 2

**Draw the equivalent of Fig. 5.7 for a 8 × 8 matrix multiplication with 2 × 2 tiling and 4 × 4 tiling. Verify that the reduction in global memory bandwidth is indeed proportional to the dimension size of the tiles.**

![alt text](exercise_2a.jpeg)

![alt text](exercise_2b.jpeg)


### Exercise 3

**What type of incorrect execution behavior can happen if one forgot to use one or both __syncthreads() in the kernel of Fig. 5.9?**

If we forget the first `__syncthreads()`, we would read from the Mds and Nds matrices when not necessarily all of the correct data is already there. This could result in adding incorrect values, coming, e.g., from uninitialized memory, to the Pvalue variable.

IF we forget the second `__syncthreads()`, we are at risk of starting to overwrite the values Mds and Nds when we are still reading from them. This can result in errors as we load something we did not intend to. 

If we forget both, we will end up with the combination of both, aka loading and using potentially corrupted values.  

### Exercise 4

**Assuming that capacity is not an issue for registers or shared memory, give one important reason why it would be valuable to use shared memory instead of registers to hold values fetched from global memory? Explain your answer.**

The one upside of using shared memory over storing values in registers is that shared memory can be used by ANY thread in the block. This means that we can load the value once from global memory, store it in shared memory, and all of the threads in this block will be able to use this value. In contrast to this, registers are only accessible "per thread," meaning other threads don't have access to values stored in the registers of some particular thread. 

### Exercise 5

**For our tiled matrix-matrix multiplication kernel, if we use a 32 × 32 tile, what is the reduction of memory bandwidth usage for input matrices M and N?**

Let's assume `M` is of size `m x n` and `N` is of size `n x o`. 

Let's think step by step. For row 0 of matrix `P`, we used to load the entire row 0 of matrix M for every single value in this row (`n` loads). Now we load the values once from global memory and put them in the first row of the tile 'Mds.' We just reduced the loads from global memory from `n` to `n/32`, so the reduction is `32x`. The same thing is repeated for every single row in this matrix; e.g., assuming the `M` and `N` matrices are both '32x32,' we would go from `32` loads to `1` load. This is repeated for every row of `M` and for every column of `N`.

### Exercise 6

**Assume that a CUDA kernel is launched with 1000 thread blocks, each of which has 512 threads. If a variable is declared as a local variable in the kernel, how many versions of the variable will be created through the lifetime of the execution of the kernel?**

There are `1000 x 512 = 512,000` threads in the grid. The variable will be created and stored for every single one of them, meaning it will be created `512,000` times. 

### Exercise 7

**In the previous question, if a variable is declared as a shared memory variable, how many versions of the variable will be created through the lifetime of the execution of the kernel?**

The variables declared as a shred memory variable are created once per block, meaning the variable will be created `1000` times. 

### Exercise 8

**Consider performing a matrix multiplication of two input matrices with dimensions N × N. How many times is each element in the input matrices requested from global memory when:**

**a. There is no tiling?**

Let’s go step by step. In the resulting matrix `P`, for every one of `n` values in the first row, we need to load the entire first row of the input matrix M. So each element from the first row of `M` is loaded `N` times. This is repeated for every row, and the same is happening for columns of `N`. Hence, in the non-tiled case, each input element is loaded N times.

**b. Tiles of size T × T are used?**

For the tiled case, we do a single load from global memory, store the value in the global memory, and reuse it in T subsequence threads in the block (we assume that the block is the size of the tile). Hence in the tiled version, each element is loaded N/T times, or T times less than in the non-tiled version. 


### Exercise 9

**A kernel performs 36 floating-point operations and seven 32-bit global memory accesses per thread. For each of the following device properties, indicate whether this kernel is compute-bound or memory-bound.**

**a. Peak FLOPS=200 GFLOPS, peak memory bandwidth=100 GB/second**

Our device can do at most 200 GFlops, or 2*10^2 * 10^9 operations per second. As per kernel specification, we have 36 floating-point operations in our kernel, meaning that our kernel can be executed 200/36 * 10^9 = 5.55 * 10^9 times within a second. 

Our memory can provide 100*10^9 bytes in a second. Assuming that our kernel needs to access 7 32-bit (4-byte) values from the global memory, this would mean we can execute it 100 * 10^9 / (7*4) = 3.57 * 10^9 times. Based on these limits, we see that the memory limit is much more severe, meaning that the kernel is memory-bound in this case.


**b. Peak FLOPS=300 GFLOPS, peak memory bandwidth=250 GB/second**

Our device can do at most 300 GFlops, or 2*10^2 * 10^9 operations per second. As per kernel specification, we have 36 floating-point operations in our kernel, meaning that our kernel can be executed 300/36 * 10^9 = 8.33 * 10^9 times within a second. 

Our memory can provide 10 * 10^9 bytes in a second, assuming that our kernel needs to access seven 32-bit values from the global memory. This would mean we can execute it 200 * 10^9 / (7 * 4) = 7.14 * 10^9 times.
Based on these limits, we see that the compute limit is much more severe, meaning that the kernel is compute bound in this case.

### Exercise 10

**To manipulate tiles, a new CUDA programmer has written a device kernel that will transpose each tile in a matrix. The tiles are of size BLOCK_WIDTH by BLOCK_WIDTH, and each of the dimensions of matrix A is known to be a multiple of BLOCK_WIDTH. The kernel invocation and code are shown below. BLOCK_WIDTH is known at compile time and could be set anywhere from 1 to 20.**

```cpp
1  dim3 blockDim(BLOCK_WIDTH,BLOCK_WIDTH);
2  dim3 gridDim(A_width/blockDim.x,A_height/blockDim.y);
3  BlockTranspose<<<gridDim, blockDim>>>(A, A_width, A_height);

4  __global__ void
5  BlockTranspose(float* A_elements, int A_width, int A_height)
6  {
7      __shared__ float blockA[BLOCK_WIDTH][BLOCK_WIDTH];

8      int baseIdx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
9      baseIdx += (blockIdx.y * BLOCK_SIZE + threadIdx.y) * A_width;

10     blockA[threadIdx.y][threadIdx.x] = A_elements[baseIdx];

11     A_elements[baseIdx] = blockA[threadIdx.x][threadIdx.y];
12 }
```

**a. Out of the possible range of values for BLOCK_SIZE, for what values of BLOCK_SIZE will this kernel function execute correctly on the device?**

It will work correctly only for `TILE_SIZE=1`, as it is a trivial case where the kernel doesn't actually do any work whatsoever. It works with only a single thread in a block, meaning there is no need to synchronize anything. 

**b. If the code does not execute correctly for all BLOCK_SIZE values, what is the root cause of this incorrect execution behavior? Suggest a fix to the code to make it work for all BLOCK_SIZE values.**

The root cause is we lack the `__syncthreads` after line 10. If we don't sync the threads at this point, we don't have any control over what is stored in the `blockA`, so while in some threads we might already have the correct values, ready to be used in line 11, some other threads might still have the old, not yet updated values, leading to errors. 

The upaded codebase:

```cpp
1  dim3 blockDim(BLOCK_WIDTH,BLOCK_WIDTH);
2  dim3 gridDim(A_width/blockDim.x,A_height/blockDim.y);
3  BlockTranspose<<<gridDim, blockDim>>>(A, A_width, A_height);

4  __global__ void
5  BlockTranspose(float* A_elements, int A_width, int A_height)
6  {
7      __shared__ float blockA[BLOCK_WIDTH][BLOCK_WIDTH];

8      int baseIdx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
9      baseIdx += (blockIdx.y * BLOCK_SIZE + threadIdx.y) * A_width;

10     blockA[threadIdx.y][threadIdx.x] = A_elements[baseIdx];
11     __syncthreads();

12     A_elements[baseIdx] = blockA[threadIdx.x][threadIdx.y];
13 }
```

### Exercise 11

**Consider the following CUDA kernel and the corresponding host function that calls it:**

```cpp
1  __global__ void foo_kernel(float* a, float* b) {
2      unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
3      float x[4];
4      __shared__ float y_s;
5      __shared__ float b_s[128];
6      for(unsigned int j = 0; j < 4; ++j) {
7          x[j] = a[j*blockDim.x*gridDim.x + i];
8      }
9      if(threadIdx.x == 0) {
10         y_s = 7.4f;
11     }
12     b_s[threadIdx.x] = b[i];
13     __syncthreads();
14     b[i] = 2.5f*x[0] + 3.7f*x[1] + 6.3f*x[2] + 8.5f*x[3]
15             + y_s*b_s[threadIdx.x] + b_s[(threadIdx.x + 3)%128];
16 }
17 void foo(int* a_d, int* b_d) {
18     unsigned int N = 1024;
19     foo_kernel <<< (N + 128 - 1)/128, 128 >>>(a_d, b_d);
20 }
```

**a. How many versions of the variable i are there?**

There is one copy of variable `i` for each thread in the grid. There are `(N + 128 - 1)/128` = `1151/128 = 8` blocks in the grid, each block with `128` threads, `8x128=1024` threads in total, so `1024` versions of variable `i`.

**b. How many versions of the array x[] are there?**

As above, there is one copy of the array `x[]` for each thread in the grid, so `1024` versions of the array `x[]`.

**c. How many versions of the variable y_s are there?**

`y_s` is the variable stored in the shared memory. There is one copy of a variable per block in the grid. Since we have 128 blocks in the grid (see a), therefore we have `128` versions of the variable `y_s`.

**d. How many versions of the array b_s[] are there?**

Same as in c, 128 blocks, so `128` versions of `b_s` stored in the shared memory.

**e. What is the amount of shared memory used per block (in bytes)?**

To estimate the amount of shared memory used per block, we need to identify all `__shared__` variables in the kernel and multiply them by their corresponding data type size, accounting for the array dimensions. In case of the `foo` kernel, lines 4 and 5 are `float y_s` and `float b_s[128]`. `y_s` is a float, so it is `4 bytes` per block; `b_s` is an array of `128` floats, so `128 x 4 = 512`, and `4 + 512 = 516` bytes of shared memory per block in total. 

**f. What is the floating-point to global memory access ratio of the kernel (in OP/B)?**

To calculate the floating-point to global memory access ratio, we need to:
- Identify all reads from global memory operations in the kernel.
Calculate the number of floating-point operations that include the data loaded from the global memory. Note we mention here loading, not writing to the memory. 

Let's start with identifying all loading operations. 

1. Line 7: 4 load operations from `a`.
2. Line 12: 1 load operation from `b`.
3. 14: 4 `add` operations (`2.5f*x[0] + 3.7f*x[1] + 6.3f*x[2] + 8.5f*x[3]`), 1 multiply operation (`y_s*b_s[threadIdx.x]`), and two more add operations `y_s*b_s[threadIdx.x] + b_s[(threadIdx.x + 3)%128]`. 7 operations, including the variable loaded from the global memory in total. 

So the ratio is 7 operations per 5 loads. Since each of the loaded variables is a float—4 bytes of memory. We have 7 operations and `5x4 bytes = 20 bytes` of memory loaded, so `7 operations/20 bytes` or `0.7 operations/byte`.

### Exercise 12

**Consider a GPU with the following hardware limits: 2048 threads/SM, 32 blocks/SM, 64K (65,536) registers/SM, and 96 KB of shared memory/SM. For each of the following kernel characteristics, specify whether the kernel can achieve full occupancy. If not, specify the limiting factor.**

*We assume here that the problem description is not correct, and the 4 KB of shared memory is per block, not per SM. 

**a. The kernel uses 64 threads/block, 27 registers/thread, and 4 KB of shared memory/SM.**

The SM supports up to 32 blocks per SM, each block running `64` threads. This brings us up to `32x64=2048` threads in total. Each thread uses 27 registers, so `2048 x 27 = 55296` registers are used—less than our upper bound of 64k registers. The kernel is using 4 KB of shared memory per block; this would bring us to a total of `32x4=128` KB of shared memory. Since we have 96KB total, we won't be able to run so many blocks. At most we will be able to only run 24 blocks, achieving `24x64/2048=0.75=75%` occupancy with the shared memory being the limiting factor. 

**b. The kernel uses 256 threads/block, 31 registers/thread, and 8 KB of shared memory/SM.**

The kernel is using the 256 threads per block, meaning we can have up to `2048/256=8` blocks max. With this configuration, we run `8x256=2048` threads in total. Each thread will use 64 registers, bringing us to the total of `2048x31=63488` registers in total, slightly below our register upper bound. The kernel is using 8 KB per block, and since we have 8 blocks, we will be using `8 x 8 KB = 64 KB` of memory total, considerably below our memory limit. This means that we can run 2048 threads and that we will achieve a 100% occupancy rate. 