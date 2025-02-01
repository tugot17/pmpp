// nvcc merge_bench.cu -o merge_bench

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>

#define cdiv(x, y) (((x) + (y)-1) / (y))
#define TILE_SIZE 256

#define CUDA_CHECK(call)                                                                                 \
    do {                                                                                                 \
        cudaError_t error = call;                                                                        \
        if (error != cudaSuccess) {                                                                      \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(EXIT_FAILURE);                                                                          \
        }                                                                                                \
    } while (0)

void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true);
void clear_l2() {
    static int l2_clear_size = 0;
    static unsigned char* gpu_scratch_l2_clear = NULL;
    if (!gpu_scratch_l2_clear) {
        CUDA_CHECK(cudaDeviceGetAttribute(&l2_clear_size, cudaDevAttrL2CacheSize, 0));
        l2_clear_size *= 2;  // extra padding, if desired
        CUDA_CHECK(cudaMalloc(&gpu_scratch_l2_clear, l2_clear_size));
    }
    CUDA_CHECK(cudaMemset(gpu_scratch_l2_clear, 0, l2_clear_size));
}

__host__ __device__ void merge_sequential(float* A, int m, float* B, int n, float* C) {
    int i = 0, j = 0, k = 0;
    while (i < m && j < n) {
        if (A[i] <= B[j]) {
            C[k++] = A[i++];
        } else {
            C[k++] = B[j++];
        }
    }
    while (i < m) {
        C[k++] = A[i++];
    }
    while (j < n) {
        C[k++] = B[j++];
    }
}

__host__ __device__ int co_rank(int k, float* A, int m, float* B, int n) {
    int i = min(k, m);
    int j = k - i;

    int i_low = max(0, k - n);
    int j_low = max(0, k - m);
    int delta;

    bool active = true;
    while (active) {
        // if i is too big, decrease it
        if (i > 0 && j < n && A[i - 1] > B[j]) {
            delta = cdiv(i - i_low, 2);
            j_low = j;
            i -= delta;
            j += delta;
        } else if (j > 0 && i < m && B[j - 1] >= A[i]) {  // if i is too small, increase it
            delta = cdiv(j - j_low, 2);
            i_low = i;
            i += delta;
            j -= delta;
        } else {
            active = false;
        }
    }
    return i;
}

__global__ void merge_sequential_kernel(float* A, int m, float* B, int n, float* C) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        merge_sequential(A, m, B, n, C);
    }
}

// A basic parallel merge kernel
__global__ void merge_basic_kernel(float* A, int m, float* B, int n, float* C) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Determine how many elements each thread will process.
    int elementsPerThread = cdiv((m + n), blockDim.x * gridDim.x);

    // Compute the output indices for this thread.
    int k_curr = tid * elementsPerThread;
    int k_next = min((tid + 1) * elementsPerThread, m + n);

    // Compute the corresponding coranks in A and B.
    int i_curr = co_rank(k_curr, A, m, B, n);
    int j_curr = k_curr - i_curr;
    int i_next = co_rank(k_next, A, m, B, n);
    int j_next = k_next - i_next;

    // Perform the sequential merge on the subarrays.
    merge_sequential(&A[i_curr], i_next - i_curr, &B[j_curr], j_next - j_curr, &C[k_curr]);
}

// A tiled merge kernel (unchanged)
__global__ void merge_tiled_kernel(float* A, int m, float* B, int n, float* C) {
    // Use shared memory for tiles from A and B.
    extern __shared__ float shareAB[];
    float* A_S = shareAB;              // first half for A tile
    float* B_S = shareAB + TILE_SIZE;  // second half for B tile

    int total = m + n;
    int chunk = cdiv(total, gridDim.x);
    int C_curr = blockIdx.x * chunk;
    int C_next = min((blockIdx.x + 1) * chunk, total);

    if (threadIdx.x == 0) {
        A_S[0] = (float)co_rank(C_curr, A, m, B, n);
        A_S[1] = (float)co_rank(C_next, A, m, B, n);
    }
    __syncthreads();

    int A_curr = (int)A_S[0];
    int A_next = (int)A_S[1];
    int B_curr = C_curr - A_curr;
    int B_next = C_next - A_next;

    int C_length = C_next - C_curr;
    int A_length = A_next - A_curr;
    int B_length = B_next - B_curr;

    int total_iteration = cdiv(C_length, TILE_SIZE);

    int C_completed = 0;
    int A_consumed = 0;
    int B_consumed = 0;
    int counter = 0;

    while (counter < total_iteration) {
        int A_remaining = A_length - A_consumed;
        int B_remaining = B_length - B_consumed;
        int A_tile = min(TILE_SIZE, A_remaining);
        int B_tile = min(TILE_SIZE, B_remaining);
        int tile_merged = min(TILE_SIZE, C_length - C_completed);

        for (int i = threadIdx.x; i < A_tile; i += blockDim.x) {
            A_S[i] = A[A_curr + A_consumed + i];
        }
        for (int i = threadIdx.x; i < B_tile; i += blockDim.x) {
            B_S[i] = B[B_curr + B_consumed + i];
        }
        __syncthreads();

        int thread_chunk = cdiv(tile_merged, blockDim.x);
        int c_tile_start = threadIdx.x * thread_chunk;
        int c_tile_end = min((threadIdx.x + 1) * thread_chunk, tile_merged);

        int a_tile_start = co_rank(c_tile_start, A_S, A_tile, B_S, B_tile);
        int b_tile_start = c_tile_start - a_tile_start;
        int a_tile_end = co_rank(c_tile_end, A_S, A_tile, B_S, B_tile);
        int b_tile_end = c_tile_end - a_tile_end;

        merge_sequential(A_S + a_tile_start, a_tile_end - a_tile_start, B_S + b_tile_start, b_tile_end - b_tile_start,
                         C + C_curr + C_completed + c_tile_start);
        __syncthreads();

        int consumed_from_A = co_rank(tile_merged, A_S, A_tile, B_S, B_tile);
        A_consumed += consumed_from_A;
        B_consumed += (tile_merged - consumed_from_A);
        C_completed += tile_merged;

        counter++;
        __syncthreads();
    }
}

void merge_sequential_gpu(float* d_A, int m, float* d_B, int n, float* d_C) {
    merge_sequential_kernel<<<1, 1>>>(d_A, m, d_B, n, d_C);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void simple_merge_parallel_gpu(float* d_A, int m, float* d_B, int n, float* d_C) {
    int block_dim = 256;
    dim3 dimBlock(block_dim);
    dim3 dimGrid(1);
    merge_basic_kernel<<<dimGrid, dimBlock>>>(d_A, m, d_B, n, d_C);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void merge_parallel_with_tiling_gpu(float* d_A, int m, float* d_B, int n, float* d_C) {
    int threadsPerBlock = 1024;
    int numBlocks = (m + n + threadsPerBlock - 1) / threadsPerBlock;
    numBlocks = min(numBlocks, 65535);
    dim3 dimBlock(threadsPerBlock);
    dim3 dimGrid(numBlocks);
    int sharedMemBytes = 2 * TILE_SIZE * sizeof(float);
    merge_tiled_kernel<<<dimGrid, dimBlock, sharedMemBytes>>>(d_A, m, d_B, n, d_C);
    CUDA_CHECK(cudaDeviceSynchronize());
}

float benchmark_merge(void (*merge_func)(float*, int, float*, int, float*), float* d_A, int m, float* d_B, int n,
                      float* d_C, int warmup, int reps) {
    for (int i = 0; i < warmup; ++i) {
        merge_func(d_A, m, d_B, n, d_C);
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float totalTime_ms = 0.0f;

    for (int i = 0; i < reps; ++i) {
        clear_l2();

        CUDA_CHECK(cudaEventRecord(start));
        merge_func(d_A, m, d_B, n, d_C);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float iterTime = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&iterTime, start, stop));
        totalTime_ms += iterTime;
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return totalTime_ms / reps;
}

float* createSortedArray(int length, float start, float step) {
    float* array = (float*)malloc(length * sizeof(float));
    for (int i = 0; i < length; i++) {
        array[i] = start + i * step;
    }
    return array;
}

bool allclose(float* a, float* b, int N, float rtol = 1e-5, float atol = 1e-8) {
    for (int i = 0; i < N; i++) {
        float allowed_error = atol + rtol * fabs(b[i]);
        if (fabs(a[i] - b[i]) > allowed_error) {
            printf("Arrays differ at index %d: %f != %f\n", i, a[i], b[i]);
            return false;
        }
    }
    return true;
}

int main() {
    const int m = 10283;
    const int n = 131131;
    const int total = m + n;

    // Allocate and initialize host arrays.
    float* h_A = createSortedArray(m, 1.0f, 0.3f);
    float* h_B = createSortedArray(n, 1.5f, 0.4f);
    float* h_C_ref = (float*)malloc(total * sizeof(float));  // for reference result

    // Compute reference result on host.
    merge_sequential(h_A, m, h_B, n, h_C_ref);

    // Allocate device memory once.
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, m * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_C, total * sizeof(float)));

    // Copy input arrays to device once.
    CUDA_CHECK(cudaMemcpy(d_A, h_A, m * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, n * sizeof(float), cudaMemcpyHostToDevice));

    const int warmup = 10;
    const int reps = 50;

    float t_seq = benchmark_merge(merge_sequential_gpu, d_A, m, d_B, n, d_C, warmup, reps);
    printf("GPU sequential merge (1 thread): %f ms\n", t_seq);

    float t_basic = benchmark_merge(simple_merge_parallel_gpu, d_A, m, d_B, n, d_C, warmup, reps);
    printf("Naive parallel merge: %f ms\n", t_basic);

    float t_tiled = benchmark_merge(merge_parallel_with_tiling_gpu, d_A, m, d_B, n, d_C, warmup, reps);
    printf("Tiled parallel merge: %f ms\n", t_tiled);

    float* h_C = (float*)malloc(total * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_C, d_C, total * sizeof(float), cudaMemcpyDeviceToHost));
    if (allclose(h_C, h_C_ref, total)) {
        printf("Result is correct!\n");
    } else {
        printf("Result is incorrect!\n");
    }

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
