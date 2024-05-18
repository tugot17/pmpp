# Chapter two
Solutions to the Programming Massively Parallel Processors book

## Code

In chapter two create just a basic vector multiplication kernel. The kernel itself can be found in [vecMul.cu](code/vecMul.cu)

### C
To run the C code just run the [Makefile](code/Makefile)

```bash
cd code
```

```bash
make
```

### Python

Set up the python env and install the required packages (`torch` and `Ninja`).

```bash
pip install -r requirements.txt
```

and run the python script

```
python vecMul.py
```

## Exercises

### Exercise 1
**Question:** If we want to use each thread in a grid to calculate one output element of a vector addition, what would be the expression for mapping the thread/block indices to the data index (i)?

**Choices:**
- A. `i = threadIdx.x + threadIdx.y;`
- B. `i = blockIdx.x + threadIdx.x;`
- C. `i = blockIdx.x * blockDim.x + threadIdx.x;`
- D. `i = blockIdx.x * threadIdx.x;`


**Solution:**
  
**C** it would be `i = blockIdx.x * blockDim.x + threadIdx.x` We need all three of these, we need `blockIdx.x` to identify the block and the `blockDim.x` to identify how big each block is. Each of the blocks has the same length (e.g. 256). So assuming we want to run the kernel for the 128th element of block in block one we would assign i to `1 * 256 + 128 = 384`


### Exercise 2
**Question:** Assume that we want to use each thread to calculate two adjacent elements of a vector addition. What would be the expression for mapping the thread/block indices to the data index (i) of the first element to be processed by a thread?

**Choices:**
- A. `i = blockIdx.x * blockDim.x + threadIdx.x * 2;`
- B. `i = blockIdx.x * threadIdx.x * 2;`
- C. `i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;`
- D. `i = blockIdx.x * blockDim.x * 2 + threadIdx.x;`

**Solution:**

**C** We want each kernel to process two adjacent elements. E.g. (0, 1), (1, 2) , ..., (1024, 1025). To do this, we need an equation that will skip every second element. The one that works here is `i = (blockIdx.x * blockDim.x + threadIdx.x) * 2` for `blockIdx=0` and `threadIdx.x=0` the first element is 0, for `threadIdx.x=1` it is 2 for `blockIdx=4` and `threadIdx.x=0` it is 1024 (assuming blocks of 256). Note that for `threadIdx.x > 128` it will jump to the next block automatically. 
  

### Exercise 3
**Question:** We want to use each thread to calculate two elements of a vector addition. Each thread block processes 2 * blockDim.x consecutive elements that form two sections. All threads in each block will process a section first, each processing one element. They will then all move to the next section, each processing one element. What would be the correct expression for the data index (i)? processing one element. Assume that variable i should be the index for the first element to be processed by a thread. What would be the expression for mapping the thread/block indices to data index of the first element?

**Choices:**
- A. `i=blockIdx.x*blockDim.x + threadIdx.x +2;`
- B. `i=blockIdx.x*threadIdx.x*2;`
- C. `i=(blockIdx.x*blockDim.x + threadIdx.x)*2;`
- D. `i=blockIdx.x*blockDim.x*2 + threadIdx.x;`

**Solution:**

**D** We want each block to process $ 2 \times BLOCKSIZE$ elements. Assuming the block size is 256, this would mean that the block 0 will process elements from 0 to 511 . 0 to 255 from the first section and 256 to 511 in section two. So the example indices processed by a thread will be $(0, 256), (1, 257) \dots (255, 511)$. For the block 1 will need to skip the 256 to 511. To do so we need to double the portion of the thread id indicated by the block size, hence the `i=blockIdx.x*blockDim.x*2 + threadIdx.x`


### Exercise 4

**Question:** For a vector addition, assume that the vector length is 8000, each thread calculates one output element, and the thread block size is 1024 threads. The programmer configures the kernel call to have a minimum number of thread blocks to cover all output elements. How many threads will be in the grid?



**Choices:**

- A. 8000

- B. 8196

- C. 8192

- D. 8200



**Solution:**

**C**. Each block has 1024 threads, we need to have 8 thread blocks to process 8000 th elements. Meaning there will be `8 * 1024 = 8192` threads.  


### Exercise 5

**Question:** If we want to allocate an array of v integer elements in the CUDA device global memory, what would be an appropriate expression for the second argument of the `cudaMalloc` call?

**Choices:**

- A. n

- B. v

- C. n * sizeof(int)

- D. v * sizeof(int)



**Solution:**

**D**. cudaMalloc function takes two arguments, address of a pointer and the size in bytes. Since we want to allocate `v x int_size` bytes the answer D makes sense. 



### Exercise 6

**Question:** If we want to allocate an array of n floating-point elements and have a floating-point pointer variable A_d to point to the allocated memory, what would be an appropriate expression for the first argument of the `cudaMalloc` call?



**Choices:**

- A. n

- B. (void*) A_d

- C. *A_d

- D. (void**) &A_d



**Solution:**

** D **. The first argument of the `cudaMalloc` function is a point3r to a pointer. Since `A_d` is a pointer the address `&A_d` is an address of a pointer. `cudaMallock` takes type void as a first argument, therefore we need to cast the pointer to the right type


### Exercise 7

**Question:** If we want to copy 3000 bytes of data from host array A_h (A_h is a pointer to element 0 of the source array) to device array A_d (A_d is a pointer to element 0 of the destination array), what would be an appropriate API call for this data copy in CUDA?



**Choices:**

- A. cudaMemcpy(3000, A_h, A_d, cudaMemcpyHostToDevice);

- B. cudaMemcpy(A_h, A_d, 3000, cudaMemcpyDeviceToHost);

- C. cudaMemcpy(A_d, A_h, 3000, cudaMemcpyHostToDevice);

- D. cudaMemcpy(3000, A_d, A_h, cudaMemcpyHostToDevice);



**Solution:**

**C** `A_d` is the destination, `A_h` is the source, the size is 3000 and the direction of transfer is from `host` to `device`. Therefore the answer is C.


### Exercise 8

**Question:** How would one declare a variable err that can appropriately receive the returned value of a CUDA API call?



**Choices:**

- A. int err;

- B. cudaError err;

- C. cudaError_t err;

- D. cudaSuccess_t err;



**Solution:**

**C** The correct type of cuda errors is cudaError_t so the answer is C. 


### Exercise 9

Consider the following CUDA kernel and the corresponding host function
that calls it:

```c
01 __global__ void foo_kernel(float* a, float* b, unsigned int N) {
02     unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
03     
04     if (i < N) {
05         b[i] = 2.7f * a[i] - 4.3f;
06     }
07 }
08 
09 void foo(float* a_d, float* b_d) {
10     unsigned int N = 200000;
11     foo_kernel<<<(N + 128 - 1) / 128, 128>>>(a_d, b_d, N);
12 }
```

**a. What is the number of threads per block?**
As indicated by the second argument of the kernel launch parameters (the thing in `<< >>>`) it is `128`.

**b. What is the number of threads in the grid?**
The number of threads lanched is equal to `number_of_blocks * block_size`. The `number_of_blocks` is inducated by the first argument of kernel launch parameters in our case `(N + 128 - 1) / 128` -> `(200000 + 128 - 1) // 128 = 1563`. The number of threads will be therefore `1563*128 = 200064`


**c. What is the number of blocks in the grid?**
As above `1563`

**d. What is the number of threads that execute the code on line 02?**
As above `1563*128 = 200064`

**e. What is the number of threads that execute the code on line 04?**
Here we have an if statement limiting the execution to only the memory allocated in `cudaMalloc`, in this case `N=200000` so the last 64 threads will not be used.

### Exercise 10:
**Question:** A new summer intern was frustrated with CUDA. He has been complaining that CUDA is very tedious. He had to declare many functions that he plans to execute on both the host and the device twice, once as a host function and once as a device function. What is your response?

**Solution:** You can just use the cuda function with `__host__` and `__device__` function type qualifiers. Than the cuda compiler will compile both a host and device versions of the function. No need to duplicate your code. 
