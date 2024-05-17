# Chapter 3.

In chapter 3 we learned about multidimensional data grids, and we wrote the first more complicated kernels. 

## Code

For a sake of simplicity we provide the most of the code with a Python interface to interract with. You run the python script which under the hood uses the cuda kernel. We reimplemented the kenrels from the chapter as well as implemented kernels for the exercises one and two.

We implement:

- Matrix multiplication, with a kernel operating on the column level and the row level.
- Matrix vector multiplication kernel.
- Matrix multiplication kernel. 
- RGB to grayscale kernel. 
- Gaussian blur kernel.

For gaussian blur we provide a little Gradio app, so you can visualize the effect of the kernel. To use it run:

```bash
python gaussian_blur/gradio_visualization.py
```

## Exercises

### Exercise 1
In this chapter we implemented a matrix multiplication kernel that has each thread produce one output matrix element. In this question, you will implement different matrix-matrix multiplication kernels and compare them.

**a.** Write a kernel that has each thread produce one output matrix row. Fill in the execution configuration parameters for the design.
```cu
1.  __global__
2.  void matrixMulRowKernel(float* M, float* N, float* P, int size){
3.      int row = blockIdx.x * blockDim.x + threadIdx.x;
4.      if (row < size){
5.          //do this for every element in the row:
6.          for (int col=0; col<size; ++col){
7.              float sum = 0;
8.              for (int j=0; j<size; ++j){
9.                  sum += M[row * size + j] * N[j * size + col];
10.             }
11.             P[row * size + col] = sum;
12.         }
13.     }
14. }
```


**b.** Write a kernel that has each thread produce one output matrix column. Fill in the execution configuration parameters for the design.


**c.** Analyze the pros and cons of each of the two kernel designs.


### Exercise 2
A matrix-vector multiplication takes an input matrix B and a vector C and produces one output vector A. Each element of the output vector A is the dot product of one row of the input matrix B and C, that is, $ \(A[i] = \sum_{j} B[i][j] * C[j]\) $. For simplicity we will handle only square matrices whose elements are single-precision floating-point numbers. Write a matrix-vector multiplication kernel and the host stub function that can be called with four parameters: pointer to the output matrix, pointer to the input matrix, pointer to the input vector, and the number of elements in each dimension. Use one thread to calculate an output vector element.

Full solution can by found in [exercise_2](code/exercise_2)

```cu
1  __global__
2  void matrixVecMulKernel(float* B, float* c, float* result, int vector_size, int matrix_rows){
3      int i = blockIdx.x * blockDim.x + threadIdx.x;
4      if (i < matrix_rows){
5          float sum = 0;
6          for (int j=0; j < vector_size; ++j){
7              sum += B[i * vector_size + j] * c[j];
8          }
9          result[i] = sum;
10     }
11 }
```



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
12     dim3 gd((N - 1) / 32 + 1, (M - 1) / 32 + 1); //TODO
13     foo_kernel <<< grid, blockDim >>> (b_d, a, M, N);
14 }
```

**a. What is the number of threads per block?**

T

WRONG
<!-- The number of threads per block can be inferred from the variable `gd` (gridDim). It is `((N - 1) / 32 + 1, (M - 1) / 32 + 1)`, where `N=1500` and `M=300`. Hence `((150 - 1) / 32 + 1, (300 - 1) / 32 + 1)` -> `(149 / 32 + 1, 299 / 32 + 1)` -> `(149 / 32 + 1, 299 / 32 + 1)` -> (the integer division) `(149 / 32 + 1, 299 / 32 + 1)` -> `4 + 1, 9+1` -> `5, 10`, so the number of threads per block is `5 x 10 = 50` -->

**b. What is the number of threads in the grid?**
WRONG
<!-- To answer this we need to figure out the number of blocks and multiply it by the number of threads per block (470, see **a**). The number of blocks can be inferred from variable `bd` (blockDim). `(32 x 16) x 50 = 25,600`, so the total number of threads is `25,600`.  -->

**c What is the number of blocks in the grid?**

As indicated by variable `bd` it is `32 x 16 = 512`.

**d. What is the number of threads that execute the code on line 05?**

To answer this question we need to know: `M`, the max `row` possible, the max `N` and the max `col` possible.

- `M` is 300
- max `row` is `blockIdx.y * blockDim.y + threadIdx.y;`. `blockIdx.y` max is 9 (0 - 9, see **3a**), `blockDim.y` is 32 ergo the max `threadIdx.y` is 32 as well. So `9 x 32 + 31 = 319`     
- `N` is 150
- max `col` is `blockIdx.x * blockDim.x + threadIdx.x`. `blockIdx.x` max is 4 (0-4, see **3a**), `blockDim.x` is 16 ergo the max `threadIdx.x` is 16 as well. So `4 x 16 + 15 = 79`.

So the total number of threads executed will be `min(300, 319) x min(150, 79) = 300 x 96 = 23,700` - so less than the total number of threads. 

### Exercise 4
Consider a 2D matrix with a width of 400 and a height of 500. The matrix is stored as a one-dimensional array. Specify the array index of the matrix element at row 20 and column 10:
- **a.** If the matrix is stored in row-major order.

In the row-major order the way the array is linearized using the formula `row x width + col`, so the index will be `20 x 400 + 10 = 8,010`.

- **b.** If the matrix is stored in column-major order.

In the column-major order the array is linearized using the formula `col x height + row`, so the index will be `10 x 500 + 20 = 5,020`

### Exercise 5
Consider a 3D tensor with a width of 400, a height of 500, and a depth of 300. The tensor is stored as a one-dimensional array in row-major order. Specify the array index of the tensor element at x = 10, y = 20, and z = 5.

The linearized index of of an element in a 3d tensor will be calculated using the foltmula `plane x width x height + row x width + col`, so the index will be `5 x 400 x 500 + 20 x 400 + 10 = 100,000,000 + 8,000 + 10 = 1.008.010`
