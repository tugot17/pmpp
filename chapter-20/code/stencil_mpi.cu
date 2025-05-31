#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "mpi.h"

// Missing constant definitions
#define DATA_COLLECT 100

// Global stencil coefficients (example 5-point stencil)
__constant__ float coeff[5];

// Missing function implementations
void random_data(float* data, int dimx, int dimy, int dimz, float min_val, float max_val) {
    int num_points = dimx * dimy * dimz;
    for (int i = 0; i < num_points; i++) {
        data[i] = min_val + (max_val - min_val) * ((float)rand() / RAND_MAX);
    }
}

void store_output(float* output, int dimx, int dimy, int dimz) {
    // Simple implementation - could write to file
    printf("Output computed for grid %d x %d x %d\n", dimx, dimy, dimz);
    // Example: save first few values
    printf("First few output values: ");
    for (int i = 0; i < 5 && i < dimx * dimy * dimz; i++) {
        printf("%.3f ", output[i]);
    }
    printf("\n");
}

void upload_coefficients(float* host_coeff, int num_coeff) {
    // Example 5-point stencil coefficients
    float stencil_coeff[5] = {-4.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    cudaError_t err = cudaMemcpyToSymbol(coeff, stencil_coeff, num_coeff * sizeof(float));
    if (err != cudaSuccess) {
        printf("Failed to upload coefficients: %s\n", cudaGetErrorString(err));
        exit(1);
    }
}

// Simple 3D stencil kernel (5-point stencil in z-direction)
__global__ void stencil_kernel(float* output, float* input, int dimx, int dimy, int dimz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= dimx || j >= dimy || k >= dimz) {
        return;
    }
    if (k < 2 || k >= dimz - 2) {
        return;  // Skip boundary points
    }

    int idx = k * dimx * dimy + j * dimx + i;

    output[idx] = coeff[0] * input[idx] + coeff[1] * input[idx - dimx * dimy] + coeff[2] * input[idx + dimx * dimy] +
                  coeff[3] * input[idx - dimx * dimy * 2] + coeff[4] * input[idx + dimx * dimy * 2];
}

void call_stencil_kernel(float* d_output, float* d_input, int dimx, int dimy, int dimz, cudaStream_t stream) {
    dim3 blockSize(8, 8, 8);
    dim3 gridSize((dimx + blockSize.x - 1) / blockSize.x, (dimy + blockSize.y - 1) / blockSize.y,
                  (dimz + blockSize.z - 1) / blockSize.z);

    stencil_kernel<<<gridSize, blockSize, 0, stream>>>(d_output, d_input, dimx, dimy, dimz);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        exit(1);
    }
}

void data_server(int dimx, int dimy, int dimz, int nreps) {
    int np;
    /* Set MPI Communication Size */
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    unsigned int num_comp_nodes = np - 1, first_node = 0, last_node = np - 2;
    unsigned int num_points = dimx * dimy * dimz;
    unsigned int num_bytes = num_points * sizeof(float);
    float *input = 0, *output = 0;
    /* Allocate input data */
    input = (float*)malloc(num_bytes);
    output = (float*)malloc(num_bytes);
    if (input == NULL || output == NULL) {
        printf("server couldn't allocate memory\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    /* Initialize input data */
    random_data(input, dimx, dimy, dimz, 1, 10);
    /* Calculate number of shared points */
    int edge_num_points = dimx * dimy * ((dimz / num_comp_nodes) + 4);
    int int_num_points = dimx * dimy * ((dimz / num_comp_nodes) + 8);
    float* send_address = input;
    /* Send data to the first compute node */
    MPI_Send(send_address, edge_num_points, MPI_FLOAT, first_node, 0, MPI_COMM_WORLD);
    send_address += dimx * dimy * ((dimz / num_comp_nodes) - 4);
    /* Send data to "internal" compute nodes */
    for (int process = 1; process < last_node; process++) {
        MPI_Send(send_address, int_num_points, MPI_FLOAT, process, 0, MPI_COMM_WORLD);
        send_address += dimx * dimy * (dimz / num_comp_nodes);
    }
    /* Send data to the last compute node */
    MPI_Send(send_address, edge_num_points, MPI_FLOAT, last_node, 0, MPI_COMM_WORLD);
    /* Wait for nodes to compute */
    MPI_Barrier(MPI_COMM_WORLD);
    /* Collect output data */
    MPI_Status status;
    for (int process = 0; process < num_comp_nodes; process++) {
        MPI_Recv(output + process * num_points / num_comp_nodes, num_points / num_comp_nodes, MPI_FLOAT, process,
                 DATA_COLLECT, MPI_COMM_WORLD, &status);
    }

    /* Store output data */
    store_output(output, dimx, dimy, dimz);

    /* Wait for compute nodes to finish cleanup */
    MPI_Barrier(MPI_COMM_WORLD);

    /* Release resources */
    free(input);
    free(output);
}

void compute_node_stencil(int dimx, int dimy, int dimz, int nreps) {
    int np, pid;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    int server_process = np - 1;
    unsigned int num_points = dimx * dimy * (dimz + 8);
    unsigned int num_bytes = num_points * sizeof(float);
    unsigned int num_halo_points = 4 * dimx * dimy;
    unsigned int num_halo_bytes = num_halo_points * sizeof(float);
    MPI_Status status;

    /* Allocate host memory */
    float* h_input = (float*)malloc(num_bytes);
    if (h_input == NULL) {
        printf("Process %d: failed to allocate host memory\n", pid);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Allocate device memory for input and output data */
    float* d_input = NULL;
    cudaError_t err = cudaMalloc((void**)&d_input, num_bytes);
    if (err != cudaSuccess) {
        printf("Process %d: GPU memory allocation failed: %s\n", pid, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    float* rcv_address = h_input + ((0 == pid) ? num_halo_points : 0);
    MPI_Recv(rcv_address, num_points, MPI_FLOAT, server_process, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

    err = cudaMemcpy(d_input, h_input, num_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Process %d: failed to copy data to GPU: %s\n", pid, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    float *h_output = NULL, *d_output = NULL;
    h_output = (float*)malloc(num_bytes);
    if (h_output == NULL) {
        printf("Process %d: failed to allocate host output memory\n", pid);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    err = cudaMalloc((void**)&d_output, num_bytes);
    if (err != cudaSuccess) {
        printf("Process %d: GPU output memory allocation failed: %s\n", pid, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    float *h_left_boundary = NULL, *h_right_boundary = NULL;
    float *h_left_halo = NULL, *h_right_halo = NULL;
    /* Allocate host memory for halo data */
    err = cudaHostAlloc((void**)&h_left_boundary, num_halo_bytes, cudaHostAllocDefault);
    if (err != cudaSuccess) {
        printf("Process %d: failed to allocate pinned memory: %s\n", pid, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    cudaHostAlloc((void**)&h_right_boundary, num_halo_bytes, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_left_halo, num_halo_bytes, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_right_halo, num_halo_bytes, cudaHostAllocDefault);

    /* Create streams used for stencil computation */
    cudaStream_t stream0, stream1;
    err = cudaStreamCreate(&stream0);
    if (err != cudaSuccess) {
        printf("Process %d: failed to create stream0: %s\n", pid, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    err = cudaStreamCreate(&stream1);
    if (err != cudaSuccess) {
        printf("Process %d: failed to create stream1: %s\n", pid, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int left_neighbor = (pid > 0) ? (pid - 1) : MPI_PROC_NULL;
    int right_neighbor = (pid < np - 2) ? (pid + 1) : MPI_PROC_NULL;

    /* Upload stencil coefficients */
    float dummy_coeff[5];
    upload_coefficients(dummy_coeff, 5);

    int left_halo_offset = 0;
    int right_halo_offset = dimx * dimy * (4 + dimz);
    int left_stage1_offset = 0;
    int right_stage1_offset = dimx * dimy * (dimz - 4);
    int stage2_offset = num_halo_points;

    MPI_Barrier(MPI_COMM_WORLD);

    for (int i = 0; i < nreps; i++) {
        /* Compute boundary values needed by other nodes first */
        call_stencil_kernel(d_output + left_stage1_offset, d_input + left_stage1_offset, dimx, dimy, 12, stream0);
        call_stencil_kernel(d_output + right_stage1_offset, d_input + right_stage1_offset, dimx, dimy, 12, stream0);
        /* Compute the remaining points */
        call_stencil_kernel(d_output + stage2_offset, d_input + stage2_offset, dimx, dimy, dimz, stream1);
        /* Copy the data needed by other nodes to the host */
        cudaMemcpyAsync(h_left_boundary, d_output + num_halo_points, num_halo_bytes, cudaMemcpyDeviceToHost, stream0);
        cudaMemcpyAsync(h_right_boundary, d_output + right_stage1_offset + num_halo_points, num_halo_bytes,
                        cudaMemcpyDeviceToHost, stream0);
        cudaStreamSynchronize(stream0);
        /* Send data to left, get data from right */
        MPI_Sendrecv(h_left_boundary, num_halo_points, MPI_FLOAT, left_neighbor, i, h_right_halo, num_halo_points,
                     MPI_FLOAT, right_neighbor, i, MPI_COMM_WORLD, &status);
        /* Send data to right, get data from left */
        MPI_Sendrecv(h_right_boundary, num_halo_points, MPI_FLOAT, right_neighbor, i, h_left_halo, num_halo_points,
                     MPI_FLOAT, left_neighbor, i, MPI_COMM_WORLD, &status);

        cudaMemcpyAsync(d_output + left_halo_offset, h_left_halo, num_halo_bytes, cudaMemcpyHostToDevice, stream0);
        cudaMemcpyAsync(d_output + right_halo_offset, h_right_halo, num_halo_bytes, cudaMemcpyHostToDevice, stream0);
        cudaDeviceSynchronize();

        float* temp = d_output;
        d_output = d_input;
        d_input = temp;
    }

    // Swap buffers to get final result in d_input
    float* temp = d_output;
    d_output = d_input;
    d_input = temp;

    /* Send the output, skipping halo points */
    err = cudaMemcpy(h_output, d_output, num_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("Process %d: failed to copy results to host: %s\n", pid, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    float* send_address = h_output + num_halo_points;
    MPI_Send(send_address, dimx * dimy * dimz, MPI_FLOAT, server_process, DATA_COLLECT, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    /* Release resources */
    free(h_input);
    free(h_output);
    cudaFreeHost(h_left_boundary);
    cudaFreeHost(h_right_boundary);
    cudaFreeHost(h_left_halo);
    cudaFreeHost(h_right_halo);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);
}

int main(int argc, char* argv[]) {
    // Reasonable problem size for testing - you can increase these for production
    int pad = 0, dimx = 48 + pad, dimy = 48, dimz = 40, nreps = 10;
    int pid = -1, np = -1;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    if (np < 3) {
        if (0 == pid) {
            printf("Needed 3 or more processes.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    // Check CUDA device availability and set device (only for compute nodes)
    if (pid < np - 1) {
        int deviceCount;
        cudaError_t err = cudaGetDeviceCount(&deviceCount);
        if (err != cudaSuccess) {
            printf("Process %d: CUDA not available: %s\n", pid, cudaGetErrorString(err));
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Set device (simple round-robin for multi-GPU systems)
        int device = pid % deviceCount;
        err = cudaSetDevice(device);
        if (err != cudaSuccess) {
            printf("Process %d: failed to set CUDA device %d: %s\n", pid, device, cudaGetErrorString(err));
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    if (pid < np - 1) {
        compute_node_stencil(dimx, dimy, dimz / (np - 1), nreps);
    } else {
        data_server(dimx, dimy, dimz, nreps);
    }

    MPI_Finalize();
    return 0;
}
