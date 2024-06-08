# Chapter 4

## Code

In this chapter we have very little actual code. We just query the CUDA device properties. 

Just run the [Makefile](code/Makefile)

```bash
cd code
```

```bash
make
```

You should see something like this:

```
Detected 2 CUDA capable device(s)

Device 0: "NVIDIA GeForce RTX 4090"
  Major revision number:         8
  Minor revision number:         9
  Total amount of global memory: 23.65 GB
  Number of multiprocessors:     128
  Total amount of constant memory: 65536 bytes
  Total amount of shared memory per block: 49152 bytes
  Total number of registers available per block: 65536
  Warp size:                     32
  Maximum number of threads per block: 1024
  Maximum sizes of each dimension of a block: 1024 x 1024 x 64
  Maximum sizes of each dimension of a grid: 2147483647 x 65535 x 65535
  Clock rate:                    2.57 GHz
  Memory clock rate:             10501 MHz
  Memory bus width:              384-bit
  L2 cache size:                 75497472 bytes

Device 1: "NVIDIA GeForce RTX 4090"
  Major revision number:         8
  Minor revision number:         9
  Total amount of global memory: 23.65 GB
  Number of multiprocessors:     128
  Total amount of constant memory: 65536 bytes
  Total amount of shared memory per block: 49152 bytes
  Total number of registers available per block: 65536
  Warp size:                     32
  Maximum number of threads per block: 1024
  Maximum sizes of each dimension of a block: 1024 x 1024 x 64
  Maximum sizes of each dimension of a grid: 2147483647 x 65535 x 65535
  Clock rate:                    2.57 GHz
  Memory clock rate:             10501 MHz
  Memory bus width:              384-bit
  L2 cache size:                 75497472 bytes
```

## Exercises

### Exercise 1

**Consider the following CUDA kernel and the corresponding host function that calls it:**
```cpp
01 __global__ void foo_kernel(int* a, int* b) {
02     unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
03     if(threadIdx.x < 40 || threadIdx.x >= 104) {
04         b[i] = a[i] + 1;
05     }
06     if(i%2 == 0) {
07         a[i] = b[i]*2;
08     }
09     for(unsigned int j = 0; j < 5 - (i%3); ++j) {
10         b[i] += j;
11     }
12 }
13 void foo(int* a_d, int* b_d) {
14     unsigned int N = 1024;
15     foo_kernel <<< (N + 128 - 1)/128, 128 >>>(a_d, b_d);
16 }
```

**a. What is the number of warps per block?**

Each warp is 32 threads; there are 128 threads in each block (second argument in `<<...>>`), so there are `128/8=4` warps in each block. 

**b. What is the number of warps in the grid?**

There are `(N + 128 - 1)/128 = (1024+128-1) = 8` blocks in total, each block having 4 warps (see a); therefore, there are 32 warps in the grid. 

**c. For the statement on line 04:**

**i. How many warps in the grid are active?**

The first warp (warp 0) is executing threads 0 to 31; all of them are running (`threadIdx.x < 40'), so the warp is active.

The second warp (warp 1) is executing threads 32–63. Since some of them are active (`threadIdx.x < 40`) because of the CUDA single-instruction-muliple-threads, all of the threads in the warp will run; some will just be inactive. 

The third warp (warp 2) is executing threads 64–95, and since none of them satisfy the if condition (`threadIdx.x < 40 || threadIdx.x >= 104'), all of the threads are skipped, so the warp is inactive. 

The fourth warp (warp 3) is executing threads 96 to 128, so, similar to the second warp, some of the threads (`threadIdx.x >= 104`) are active, meaning the warp will run.

So 3 warps per block are active, bringing it to a total of `3x8=24` warps in the grid. 

**ii. How many warps in the grid are divergent?**

As above, warp 1 and warp 3 are divergent (only some of the threads in the warp are active), so there are a total of two divergent warps per block, or `8x2=16` warps in total.

**iii. What is the SIMD efficiency (in %) of warp 0 of block 0?**

There are 32 threads in the warp (0 to 31), and all of them are being executed, so the efficiency is `32/32 = 100%`.

**iv. What is the SIMD efficiency (in %) of warp 1 of block 0?**

Warp 1 is covering threads `[32,63]`. Threads `32-39` are being executed, while threads `40-63` are not (`threadIdx.x < 40`). So the SIMD efficiency is `8/32=0,25=25%`

**v. What is the SIMD efficiency (in %) of warp 3 of block 0?**

Warp 3 is covering threads 96–127. Threads `96-103` are inactive, while threads 104-127 are being executed. So the SIMD efficiency is `24/32=0.75=75%`

**d. For the statement on line 07:**

  **i. How many warps in the grid are active?**

  Every second thread in the grid is active, meaning, half of the threads in every single warp will be active, meaning all 32 warps are active. 

  **ii. How many warps in the grid are divergent?**

All of the warps in the grid are divergent. So there are 32 divergent warps. 

  **iii. What is the SIMD efficiency (in %) of warp 0 of block 0?**

Half of the threads in a warp are active so the SIMD efficiency is `16/32=0.5=50%`

**e. For the loop on line 09:**

`i` ranges from 0 to 1023 (`8 x 128=1024` threads in the grid) . `i%3` can have 3 values `{0, 1, 2}`.  342 zeros, 341 ones and 341 twos. 

  ```cu
09     for(unsigned int j = 0; j < 5 - (i%3); ++j) 
10         b[i] += j
11     }
  ```

So line 10 will be executed 5 (5-0) times in 342 cases, 4 (5-1) times in 341 cases, and 3 (5-2) times in 341 cases. 

  **i. How many iterations have no divergence?**
  
  3 iterations will have no divergence (all 1024 threads will execute them). 

  **ii. How many iterations have divergence?**
  
  4 iterations and 5 iterations will have thread divergence. 
  

### Exercise 2
**For a vector addition, assume that the vector length is 2000, each thread calculates one output element, and the thread block size is 512 threads. How many threads will be in the grid?**

The minimum number of blocks of size 512 to cover 2000 elements is 4 - 2048 threads in total. 

### Exercise 3
**For the previous question, how many warps do you expect to have divergence due to the boundary check on vector length?**

There will be `2048 / 32 = 64` warps in total. The warp covering threads `2015-2047` will be inactive (aka skipped). The warp covering threeads `1984-2015` will be divergent. The threads 1984-1999 will have data to process, while the threads `2000-2015` will not. 

### Exercise 4
**Consider a hypothetical block with 8 threads executing a section of code before reaching a barrier. The threads require the following amount of time (in microseconds) to execute the sections: 2.0, 2.3, 3.0, 2.8, 2.4, 1.9, 2.6, and 2.9; they spend the rest of their time waiting for the barrier. What percentage of the threads’ total execution time is spent waiting for the barrier?**

To answer this, we need to identify the thread spending most of the time in the section; in this case, it is `3.0 ms`. Then we need to calculate the delta between the time it takes to execute the section for every thread. In this case, it is `(3.0 - 2.0) + (3.0 - 2.3) + (3.0 - 3.0) + (3.0 - 2.8) + (3.0 - 2.4) + (3.0 - 1.9) + (3.0 - 2.6) = 1.0 + 0.7 + 0.0 + 0.2 + 0.6 + 1.1 + 0.4 = 4.0`. Last but not least, we need to divide the delta by the total time spent on the execution: `8x3.0=24.0`ms. In this case, it will be `4.0/24.0=0.16=16%`, so 16% of the execution time is spent waiting for the barrier. 


### Exercise 5
**A CUDA programmer says that if they launch a kernel with only 32 threads in each block, they can leave out the `__syncthreads()` instruction wherever barrier synchronization is needed. Do you think this is a good idea? Explain.**

Since all of the threads are processed within the same warp, they are executed in lock-step in the SIMD (Single Instruction, Multiple Data) manner. This means they are inherently synchronized at the instruction level, so in theory, this might be a good enough prevention mechanism, allowing us to omit explicit usage of the `__syncthreads()` instruction. However, in practice, e.g., when memory is involved, it might not necessarily be true. While the instructions will be executed one by one, the underlying hardware, e.g., read/write to the memory, might in practice be subject to delays (due to various hardware limitations). Hence, in practice, it might be safer to use `__syncthreads()`. Moreover, Nvidia does not guarantee that in the future the warp size will remain at 32; in order to build a future-proof codebase, it is usually better to use `__syncthreads()`.


### Exercise 6
**If a CUDA device’s SM can take up to 1536 threads and up to 4 thread blocks, which of the following block configurations would result in the most number of threads in the SM?**
   - **a. 128 threads per block**
   - **b. 256 threads per block**
   - **c. 512 threads per block**
   - **d. 1024 threads per block**

To determine which config will result in the maximum number of threads, we need to consider two limitations:

- SM supports up to 4 blocks
- SM supports up to 1535 threads

To calculate the number of threads each config will yield, we need to:

- calculate the number of blocks given the thread size, accounting for the total number of threads supported by an SM, and figure out if it is larger than the allowed one (4)
- multiply the `min(4, number_of_thread_blocks) x the_block_size`.

Let's do this for our configurations:

- **a.** `min(4, 1536/128) * 128 = min(4, 12) x 128 = 4 x 128 = 512`
- **b.** `min(4, 1536/256) * 256 = min(4, 6) x 256 = 4 x 256 = 1024`
- **c.** `min(4, 1536/512) * 512 = min(4, 3) x 512 = 3 x 512 = 1536`
- **d.** `min(4, 1536/1024) * 1024 = min(4, 1) x 1024 = 1 x 1024 = 1024`

So the anseer is **c**. 


### Exercise 7
**Assume a device that allows up to 64 blocks per SM and 2048 threads per SM. Indicate which of the following assignments per SM are possible. In the cases in which it is possible, indicate the occupancy level.**

**a. 8 blocks with 128 threads each**  
**b. 16 blocks with 64 threads each**  
**c. 32 blocks with 32 threads each**  
**d. 64 blocks with 32 threads each**  
**e. 32 blocks with 64 threads each**  

To answer this, we need to multiply the number of blocks by the block size and verify if this is below the number of threads supported by SM. Normally, we would also need to verify if the proposed number of blocks in `<=64` allowed blocks, but in this case, all of the configurations fulfill this criteria. To calculate the occupancy, we will divide the number of threads for the configuration by the number of threads supported by the SM (`2048` in this case).

- **a.** `8 x 128 = 1024`. `1024` is `<=2048`, so it is possible. The occupacy in this case is `1024/2048 = 0.5=50%`
- **b.** `16 x 64 = 1024`. `1024` is `<=2048`, so it is possible. The occupacy in this case is `1024/2048 = 0.5=50%`
- **c.** `32 x 32 = 1024`. `1024` is `<=2048`, so it is possible. The occupacy in this case is `1024/2048 = 0.5=50%`
- **d.** `64 x 32 = 2048`. `2048` is `<=2048`, so it is possible. The occupacy in this case is `2048/2048 = 1.0=100%`
- **e.** `32 x 64 = 2048`. `2048` is `<=2048`, so it is possible. The occupacy in this case is `2048/2048 = 1.0=100%`

### Exercise 8
**Consider a GPU with the following hardware limits: 2048 threads per SM, 32 blocks per SM, and 64K (65,536) registers per SM. For each of the following kernel characteristics, specify whether the kernel can achieve full occupancy. If not, specify the limiting factor.**


The full occupacy is achieved if the SM can utilize all 2048 threads. We have two limiting factors: 
1.  The SM supports up to 32 blocks
2.  The SM supports up to 65,536.

We need to operate under these constraints. 

**a. The kernel uses 128 threads per block and 30 registers per thread.**

Maximum number of blocks based on the block size: `2048 / 128 = 16`. `16` is below the hardware limit of `32` blocks, so this constraint is satisfied. 

Number of registers per block: `128 x 30 = 3,840` registers per block. Maximum number of blocks based on the registers per block:  `65,536 / 3,840 = 17`. 17 is more than the limit impossed on us by the block size. 

So the total number of threads for this setting is: `16 x 128 = 2048` threads, so we get full utilization: `(2048 / 2048 = 100%)`.

**b. The kernel uses 32 threads per block and 29 registers per thread.**
Maximum number of blocks based on the block size: `2048 / 32 = 64`. 64 is above the hardware limit of 32 blocks per SM the constraint is not satisfied, and we can't operate those mamy blocks. 

Number of registers per block: `32 x 29 = 928`. Maximum number of blocks based on the registers per block:  `65,536 / 928 = 70`. The SM limitation is 32 blocks per SM; we can't operate that many.

So the total number of threads for this setting is `32 x 32 =1024` threads, so we get 50% `1024/2048=0.5=50%` utilization.

The limiting factor is the number of blocks supported by the GPU.

**c. The kernel uses 256 threads per block and 34 registers per thread.**
Maximum number of blocks based on the block size: `2048 / 256 = 8`. 8 is below the hardware limit of `32` blocks, so this constraint is satisfied. mamy blocks.

Number of registers per block: `256 x 34 = 8,704`. Maximum number of blocks based on the registers per block:  `65,536 / 8,704 = 7`. So 7 is the new number of blocks. 

The total number of threads will be `256 x 7 = 1,792` threads. So we have `1,792 / 2048 = 0,87 = 87%`. So we don't get the full occupancy. 

The limiting factor is the register limit. 

### Exercise 9
**A student mentions that they were able to multiply two 1024 × 1024 matrices using a matrix multiplication kernel with 32 × 32 thread blocks. The student is using a CUDA device that allows up to 512 threads per block and up to 8 blocks per SM. The student further mentions that each thread in a thread block calculates one element of the result matrix. What would be your reaction and why?**


There are `1024 x 1024 = 1.048.576` elements in the matrix. The student is using a grid with `32 x 32 = 1024` blocks, each block supporting `512 threads`. In total, there will be only `1024 x 512 = 524.288` threads in the grid. Students claim that each thread is calculating only one element of the result matrix, which is not possible based on the above configuration. The number of SM blocks per SM is irrelevant to this problem.
