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
01 unsigned int N = 1500;
02 unsigned int M = 300;
03 __global__ void foo_kernel(float* a, float* b, unsigned int M, unsigned int N) {
04     unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
05     unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
06     if (row < M && col < N) {
07         b[row*N + col] = a[row*N + col]*2.1f + 4.8f;
08     }
09 }
10 void foo(float* a, float* b_d) {
11     unsigned int N = 1500;
12     unsigned int M = 300;
13     dim3 blockDim(32, 16);
14     dim3 grid((N - 1) / 32 + 1, (M - 1) / 32 + 1);
15     foo_kernel <<< grid, blockDim >>> (b_d, a, M, N);
16 }
```

**a. What is the number of threads per block?**

**b. What is the number of threads in the grid?**

### Exercise 4
Consider a 2D matrix with a width of 400 and a height of 500. The matrix is stored as a one-dimensional array. Specify the array index of the matrix element at row 20 and column 10:
- **a.** If the matrix is stored in row-major order.
- **b.** If the matrix is stored in column-major order.

### Exercise 5
Consider a 3D tensor with a width of 400, a height of 500, and a depth of 300. The tensor is stored as a one-dimensional array in row-major order. Specify the array index of the tensor element at x = 10, y = 20, and z = 5.
