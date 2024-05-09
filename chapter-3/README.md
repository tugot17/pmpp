# Exercises

### Exercise 1
In this chapter we implemented a matrix multiplication kernel that has each thread produce one output matrix element. In this question, you will implement different matrix-matrix multiplication kernels and compare them.

**a.** Write a kernel that has each thread produce one output matrix row. Fill in the execution configuration parameters for the design.
<details>
Placeholder
</details>


**b.** Write a kernel that has each thread produce one output matrix column. Fill in the execution configuration parameters for the design.
<details>
Placeholder
</details>

**c.** Analyze the pros and cons of each of the two kernel designs.
<details>
Placeholder
</details>

### Exercise 2
A matrix-vector multiplication takes an input matrix B and a vector C and produces one output vector A. Each element of the output vector A is the dot product of one row of the input matrix B and C, that is, $ \(A[i] = \sum_{j} B[i][j] * C[j]\) $. For simplicity we will handle only square matrices whose elements are single-precision floating-point numbers. Write a matrix-vector multiplication kernel and the host stub function that can be called with four parameters: pointer to the output matrix, pointer to the input matrix, pointer to the input vector, and the number of elements in each dimension. Use one thread to calculate an output vector element.



### Exercise 3
Consider the following CUDA kernel and the corresponding host function that calls it:

```c
01 __global__ void foo_kernel(float* a, float* b, unsigned int M, unsigned int N) {
02     unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
03     unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
04     if (row < M && col < N) {
05         b[row*N + col] = a[row*N + col]*2.1f + 4.8f;
06     }
07 }
08 void foo(float* a, float* b_d) {
09     unsigned int N = 150;
10     unsigned int M = 300;
11     dim3 bd(16, 32);
12     dim3 gd((N - 1) / 32 + 1, (M - 1) / 32 + 1);
13     foo_kernel <<< grid, blockDim >>> (b_d, a, M, N);
14 }
```

**a. What is the number of threads per block?**

The number of threads per block can be infered from the variable `gd` (gridDim). It is `((N - 1) / 32 + 1, (M - 1) / 32 + 1)`, where `N=1500` and `M=300`. Hence `((150 - 1) / 32 + 1, (300 - 1) / 32 + 1)` -> `(149 / 32 + 1, 299 / 32 + 1)` -> `(149 / 32 + 1, 299 / 32 + 1)` -> (the integer division) `(149 / 32 + 1, 299 / 32 + 1)` -> `4 + 1, 9+1` -> `5, 10`, so the number of threads per block is `5 x 10 = 50`

**b. What is the number of threads in the grid?**

To answer this we need to figure out the number of blocks and multiply it by the number of threads per block (470, see **a**). The number of blocks can be inferred from variable `bd` (blockDim). `(32 x 16) x 50 = 25,600`, so the total number of threads is `25,600`. 

**c What is the number of blocks in the grid?**

As indicated by variable `bd` it is `32 x 16 = 512`.

**d. What is the number of threads that execute the code on line 05?**
To answer this we need to consider the total number of threads (`25,600`), and exclude the threads that will not be executed cause of line 04 `if (row < M && col < N) {`. 

To calculate the max row number `row = blockIdx.y * blockDim.y + threadIdx.y;` `threadIdx.y`,

`blockIdx.y` goes from 0 to 15 (see `bd.y`), 
`blockDim.y` is `(N - 1) / 32 + 1 = 5` (see **a**)
`threadIdx.y` goes from 0 to 5 (see above). 

So the max is `15 x 5 + 5` = `80`, `M= 300` so all of the blocks here can be processed. 

`blockIdx.x` goes from 0 to 31 (see `bd.x`), 
`blockDim.x` is `(M - 1) / 32 + 1 = 19` (see **a**)
`threadIdx.x` goes from 0 to 19 (see above). 

So the max is `31 x 19 + 19 = 608`, `N = 150`, so i don't fucking know ...


As indicated by `gd` `blockDim.y` can go from 0 up to to `(N - 1) / 32 + 1 = 5` (see **a**), `blockIdx.y` go from 0 all the way up to `15` (see `bd.y`, remember that in cuda the order is inversed z, y, x). And the `threadIdx.y`

The max row, as indicated by variable `bd`, is `16`. 

### Exercise 4
Consider a 2D matrix with a width of 400 and a height of 500. The matrix is stored as a one-dimensional array. Specify the array index of the matrix element at row 20 and column 10:
- **a.** If the matrix is stored in row-major order.

In the row-major order the way the array is linearized using the formula `row x width + col`, so the index will be `20 x 400 + 10 = 8,010`.

- **b.** If the matrix is stored in column-major order.

In the column-major order the array is linearized using the formula `col x height + row`, so the index will be `10 x 500 + 20 = 5,020`

### Exercise 5
Consider a 3D tensor with a width of 400, a height of 500, and a depth of 300. The tensor is stored as a one-dimensional array in row-major order. Specify the array index of the tensor element at x = 10, y = 20, and z = 5.

The linearized index of of an element in a 3d tensor will be calculated using the foltmula `plane x width x height + row x width + col`, so the index will be `5 x 400 x 500 + 20 x 400 + 10 = 100,000,000 + 8,000 + 10 = 1.008.010`
