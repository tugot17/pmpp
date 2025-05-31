# Chapter 20

## Code

We replicate the MPI stencil kernel described in the chapter. We provide the code for the kernels and also the host function. To run it, run:

```bash
make run
```
You should see something resembling

```logs
nvcc -o stencil_mpi stencil_mpi.cu -I/usr/lib/x86_64-linux-gnu/openmpi/include -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi
mpirun -np 3 ./stencil_mpi
Output computed for grid 48 x 48 x 40
First few output values: 23817476.000 -14936832.000 40859312.000 33115106.000 71862064.000 
```

## Exercises

### Exercise 1

**Assume that the 25-point stencil computation in this chapter is applied to a grid whose size is `64` grid points in the x dimension, `64` in the y dimension, and `2048` in the z dimension. The computation is divided across 17 MPI ranks, of which `16` ranks are compute processes and `1` rank is the data server process.**

**a. How many output grid point values are computed by each compute process?**

First things first, we need to decide along which axis we will divide the work. We have a grid of size `64 x 64 x 2048` so it makes the most sense to split across the `z` dimension. Since we have `16` processes, each process will get a grid of size `64 x 64 x 2048/16 = 64 x 64 x 128` plus halo cells. 

There will be 14 internal processes and 2 edge processes. 

A 25-point stencil extends 4 points in each dimension (8 total), meaning we will lose 4 points on both sides in each dimension. 

So each internal process will have `(64-8) x (64-8) x 128 = 401 408` output grid points. The edge processes will lose 4 due to halo cells, so they will have `(64-8) x (64-8) x (128-4) = 388 864` output grid points.

**b. How many halo grid points are needed:**
**i. By each internal compute process?**
4 halo cells along the `z` dimension for the left side, 4 halo cells along the `z` dimension for the right side, so the total of `(64 x 64 x 4) x 2 = 16384 x 2 = 32768` halo grid points. 

Note that as halo grid points, we use the entire grid.

> Note that the total number of slices in each device memory is four slices of left halo points (dashed white) plus four slices of left boundary points plus `dim_x x dim_y x (dim_z - 8)` internal points plus four slices of right boundary points plus four slices of right halo points (dashed white).

**ii. By each edge compute process?**

4 halo cells along the `z` dimension but only on one side, so the total of `64 x 64 x 4 = 16384` halo grid points. 

**c. How many boundary grid points are computed in stage 1 of Fig. 20.12:**
**i. By each internal compute process?**
Internal processes compute both left and right boundary slices. For each slice, we need to account for the 25-point stencil. All in all `((64 - 8) x (64 - 8) x 4) x 2 = 12544 x 2 = 25088` grid points will be calculated. 

**ii. By each edge compute process?**
Edge processes only compute one side boundary slice, so `(64 - 8) x (64 - 8) x 4 = 12544` grid points will be calculated. 

**d. How many internal grid points are computed in stage 2 of Fig. 20.12:**
**i. By each internal compute process?**
From earlier calculations (**a**) we know that there should be `56 × 56 × 128 = 401 408` computed points in total between stage 1 and stage 2. A simple subtraction would tell us that there should be `401408 - 25088 = 376320` internal grid points computed in stage 2. 

We can verify these results. We have a total of 128 z-slices; stage 1 computes 8 boundary slices (4 left and 4 right ones), so we are left with `128-8 = 120` slices, bringing us to the total of `(64 - 8) x (64 - 8) x 120 = 56 x 56 x 120 = 376320` computed grid points, so sthe samenumber as we calculated before. 

**ii. By each edge compute process?**
We can do similar calculations. We know that there should be a total of `388 864` grid points being processed and that stage 1 processes `12544`, meaning that there will be `388864 - 12544 = 376320` grid points being processed during stage 2. 

We can do the same verification as before, for along `z` afix we have 128 slices, but only 124 can be processed due to the boundary condition. Then we have 4 processed in stage 1, leaving us with `(64 - 8) x (64 - 8) x 120 = 56 x 56 x 120 = 376320` points being computed in stage 2. 

**e. How many bytes are sent in stage 2 of Fig. 20.12:**
**i. By each internal compute process?**

From the code we look at 

```cpp
/* Send data to left, get data from right */
MPI_Sendrecv(h_left_boundary, num_halo_points, MPI_FLOAT,
    left_neighbor, i, h_right_halo, num_halo_points,
    MPI_FLOAT, right_neighbor, i, MPI_COMM_WORLD, &status );

/* Send data to right, get data from left */
MPI_Sendrecv(h_right_boundary, num_halo_points, MPI_FLOAT,
    right_neighbor, i, h_left_halo, num_halo_points,
    MPI_FLOAT, left_neighbor, i, MPI_COMM_WORLD, &status );
```

Where

```cpp
unsigned int num_halo_points = 4 * dimx * dimy;
```

Meaning we send `(4 x 64 x 64 x 4 bytes) x 2 = 65536 x 2 = 131072 bytes` in total.

**ii. By each edge compute process?**

Just half of the above, so `65536` bytes. 

### Exercise 2

**If the MPI call `MPI_Send(ptr_a, 1000, MPI_FLOAT, 2000, 4, MPI_COMM_WORLD)` results in a data transfer of 4000 bytes, what is the size of each data element being sent?**

- **a. 1 byte**
- **b. 2 bytes**
- **c. 4 bytes**
- **d. 8 bytes**

`MPI_Send(buf, count, datatype, dest, tag, comm)`

- `buf` - starting address of send buffer (pointer)
- `count` - number of elements in the send buffer (nonnegative integer)
- `datatype` - datatype of each send buffer element (MPI_Datatype) - this is what we are interested in; we can check this in the MPI documentation, or we can infer it.

Other elements, `dest`, `tag`, and `comm` are not that interesting. 

We know that calling this line results in `4000 bytes` being transferred, and we know that we have `1000` elements. Based on this, we can infer that each element is `4000 / 1000 = 4 bytes` 

So `C` 4 bytes is the answer. 

### Exercise 3

**Which of the following statements is true?**
**a. MPI_Send() is blocking by default.**
This is kind of true, but not fully. `MPI_Send` buffers a small message, allowing the send to complete locally before the receiver actually receives the data. 

**b. MPI_Recv() is blocking by default.**
Yes, this is fundamentally true. The  `MPI_Recv()` is always blocking until a matching message is received. 

**c. MPI messages must be at least 128 bytes.**
Nope, you can send the message of whatever size you want (including 0 bytes).

**d. MPI processes can access the same variable through shared memory.**
Nope, each MPI process has its own memory - we call it a distributed memory model.

### Exercise 4

**Modify the example code to remove the calls to cudaMemcpyAsync() from the compute processes’ code by using GPU memory addresses on `MPI_Send` and `MPI_Recv`.**

```cpp
void compute_node_stencil(int dimx, int dimy, int dimz, int nreps ) {
    int np, pid;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    int server_process = np - 1;
    unsigned int num_points      = dimx * dimy * (dimz + 8);
    unsigned int num_bytes       = num_points * sizeof(float);
    unsigned int num_halo_points = 4 * dimx * dimy;
    unsigned int num_halo_bytes  = num_halo_points * sizeof(float);
    MPI_Status status;
    
    /* Allocate host memory */
    float *h_input  = (float *)malloc(num_bytes);
    /* Allocate device memory for input and output data */
    float *d_input = NULL;
    cudaMalloc((void **)&d_input,  num_bytes );
    float *rcv_address = h_input + ((0 == pid) ? num_halo_points : 0);
    MPI_Recv(rcv_address, num_points, MPI_FLOAT, server_process,
        MPI_ANY_TAG, MPI_COMM_WORLD, &status );
    cudaMemcpy(d_input, h_input, num_bytes, cudaMemcpyHostToDevice);

    float *h_output = NULL, *d_output = NULL;
    h_output = (float *)malloc(num_bytes);
    cudaMalloc((void **)&d_output, num_bytes );

    // REMOVED: Host pinned memory allocations for halo data
    // No longer need h_left_boundary, h_right_boundary, h_left_halo, h_right_halo

    /* Create streams used for stencil computation */
    cudaStream_t stream0, stream1;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);

    int left_neighbor  = (pid > 0)         ? (pid - 1) : MPI_PROC_NULL;
    int right_neighbor = (pid < np - 2)    ? (pid + 1) : MPI_PROC_NULL;

    /* Upload stencil coefficients */
    float dummy_coeff[5];
    upload_coefficients(dummy_coeff, 5);
    int left_halo_offset    = 0;
    int right_halo_offset   = dimx * dimy * (4 + dimz);
    int left_stage1_offset  = 0;
    int right_stage1_offset = dimx * dimy * (dimz - 4);
    int stage2_offset       = num_halo_points;
    MPI_Barrier( MPI_COMM_WORLD );
    
    for(int i=0; i < nreps; i++) {
        /* Compute boundary values needed by other nodes first */
        call_stencil_kernel(d_output + left_stage1_offset,
            d_input + left_stage1_offset, dimx, dimy, 12, stream0);
        call_stencil_kernel(d_output + right_stage1_offset,
            d_input + right_stage1_offset, dimx, dimy, 12, stream0);
        /* Compute the remaining points */
        call_stencil_kernel(d_output + stage2_offset, d_input + 
            stage2_offset, dimx, dimy, dimz, stream1);

        // REMOVED: cudaMemcpyAsync calls to copy data to host
        // REMOVED: cudaStreamSynchronize(stream0)

        /* CUDA-aware MPI: Direct GPU-to-GPU communication */
        /* Send data to left, get data from right */
        MPI_Sendrecv(d_output + num_halo_points, num_halo_points, MPI_FLOAT,
            left_neighbor, i, d_output + right_halo_offset, num_halo_points,
            MPI_FLOAT, right_neighbor, i, MPI_COMM_WORLD, &status );
        /* Send data to right, get data from left */
        MPI_Sendrecv(d_output + right_stage1_offset + num_halo_points, num_halo_points, MPI_FLOAT,
            right_neighbor, i, d_output + left_halo_offset, num_halo_points,
            MPI_FLOAT, left_neighbor, i, MPI_COMM_WORLD, &status );

        // REMOVED: cudaMemcpyAsync calls to copy data back to device

        cudaDeviceSynchronize();

        float *temp = d_output;
        d_output = d_input; d_input = temp;
    }

    /* Wait for previous communications */
    MPI_Barrier(MPI_COMM_WORLD);

    float *temp = d_output;
    d_output = d_input;
    d_input = temp;

    /* Send the output, skipping halo points */
    cudaMemcpy(h_output, d_output, num_bytes, cudaMemcpyDeviceToHost);
    float *send_address = h_output + num_halo_points;
    MPI_Send(send_address, dimx * dimy * dimz, MPI_FLOAT,
        server_process, DATA_COLLECT, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    /* Release resources */
    free(h_input); free(h_output);
    // REMOVED: cudaFreeHost calls for halo memory
    cudaFree( d_input ); cudaFree( d_output );
    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);
}
```

