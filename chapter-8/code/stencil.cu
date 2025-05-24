#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <iostream>
#include <vector>
#include <functional>

//some of our kernels have qubic requirement for the shared memory, other have squared, hence we introduce two block sizes
#define OUT_TILE_DIM_SMALL 8
#define IN_TILE_DIM_SMALL (OUT_TILE_DIM_SMALL+2)
#define OUT_TILE_DIM_BIG 30
#define IN_TILE_DIM_BIG (OUT_TILE_DIM_BIG+2)

int c0 = 0;
int c1 = 1;
int c2 = 1;
int c3 = 1;
int c4 = 1;
int c5 = 1;
int c6 = 1;

#define CUDA_CHECK(call)                                                                                 \
    do {                                                                                                 \
        cudaError_t error = call;                                                                        \
        if (error != cudaSuccess) {                                                                      \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(EXIT_FAILURE);                                                                          \
        }                                                                                                \
    } while (0)

#define gpuErrchk(ans) \
    { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) {
            exit(code);
        }
    }
}

void clear_l2() {
    static int l2_clear_size = 0;
    static unsigned char* gpu_scratch_l2_clear = NULL;
    if (!gpu_scratch_l2_clear) {
        cudaDeviceGetAttribute(&l2_clear_size, cudaDevAttrL2CacheSize, 0);
        l2_clear_size *= 2;
        gpuErrchk(cudaMalloc(&gpu_scratch_l2_clear, l2_clear_size));
    }
    gpuErrchk(cudaMemset(gpu_scratch_l2_clear, 0, l2_clear_size));
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

void stencil_3d_sequential(float* in, float* out, unsigned int N, 
                          int c0, int c1, int c2, int c3, int c4, int c5, int c6) {
    for (int i = 1; i < N - 1; i++) {
        for (int j = 1; j < N - 1; j++) {
            for (int k = 1; k < N - 1; k++) {
                out[i * N * N + j * N + k] =
                    c0 * in[i * N * N + j * N + k] +
                    c1 * in[i * N * N + j * N + (k-1)] +
                    c2 * in[i * N * N + j * N + (k+1)] +
                    c3 * in[i * N * N + (j-1) * N + k] +
                    c4 * in[i * N * N + (j+1) * N + k] +
                    c5 * in[(i-1) * N * N + j * N + k] +
                    c6 * in[(i+1) * N * N + j * N + k];
            }
        }
    }
}

__global__ void stencil_kernel(float* in, float* out, unsigned int N,
                              int c0, int c1, int c2, int c3, int c4, int c5, int c6) {
    unsigned int i = blockIdx.z*blockDim.z + threadIdx.z;
    unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
        out[i*N*N + j*N + k] = c0*in[i*N*N + j*N + k]
                             + c1*in[i*N*N + j*N + (k - 1)]
                             + c2*in[i*N*N + j*N + (k + 1)]
                             + c3*in[i*N*N + (j - 1)*N + k]
                             + c4*in[i*N*N + (j + 1)*N + k]
                             + c5*in[(i - 1)*N*N + j*N + k]
                             + c6*in[(i + 1)*N*N + j*N + k];
    }
}

void stencil_3d_parallel_basic(float* in, float* out, unsigned int N,
                              int c0, int c1, int c2, int c3, int c4, int c5, int c6){
    float *d_in, *d_out;
    cudaError_t error;

    error = cudaMalloc((void**)&d_in, N*N*N * sizeof(float));
    if (error != cudaSuccess) {
        std::cout << "cudaMalloc d_in failed: " << cudaGetErrorString(error) << std::endl;
        return;
    }

    error = cudaMalloc((void**)&d_out, N*N*N * sizeof(float));
    if (error != cudaSuccess) {
        std::cout << "cudaMalloc d_out failed: " << cudaGetErrorString(error) << std::endl;
        return;
    }

    error = cudaMemcpy(d_in, in, N * N * N * sizeof(float), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        std::cout << "cudaMemcpy to device failed: " << cudaGetErrorString(error) << std::endl;
        return;
    }

    dim3 dimBlock(OUT_TILE_DIM_SMALL, OUT_TILE_DIM_SMALL, OUT_TILE_DIM_SMALL);
    dim3 dimGrid(cdiv(N, dimBlock.x), cdiv(N, dimBlock.y), cdiv(N, dimBlock.z));

    stencil_kernel<<<dimGrid, dimBlock>>>(d_in, d_out, N, c0, c1, c2, c3, c4, c5, c6);

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cout << "Kernel launch failed: " << cudaGetErrorString(error) << std::endl;
    }
    cudaDeviceSynchronize();

    error = cudaMemcpy(out, d_out, N * N * N * sizeof(float), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        std::cout << "cudaMemcpy to host failed: " << cudaGetErrorString(error) << std::endl;
    }

    cudaFree(d_in);
    cudaFree(d_out);
}

__global__ void stencil_kernel_shared_memory(float* in, float* out, unsigned int N,
                                           int c0, int c1, int c2, int c3, int c4, int c5, int c6) {
    int i = blockIdx.z*OUT_TILE_DIM_SMALL+ threadIdx.z - 1;
    int j = blockIdx.y*OUT_TILE_DIM_SMALL+ threadIdx.y - 1;
    int k = blockIdx.x*OUT_TILE_DIM_SMALL+ threadIdx.x - 1;
    __shared__ float in_s[IN_TILE_DIM_SMALL][IN_TILE_DIM_SMALL][IN_TILE_DIM_SMALL];
    if(i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
        in_s[threadIdx.z][threadIdx.y][threadIdx.x] = in[i*N*N + j*N + k];
    }
    __syncthreads();
    if(i >= 1 && i < N-1 && j >= 1 && j < N-1 && k >= 1 && k < N-1) {
        if(threadIdx.z >= 1 && threadIdx.z < IN_TILE_DIM_SMALL-1 && threadIdx.y >= 1
           && threadIdx.y<IN_TILE_DIM_SMALL-1 && threadIdx.x>=1 && threadIdx.x<IN_TILE_DIM_SMALL-1) {
            out[i*N*N + j*N + k] = c0*in_s[threadIdx.z][threadIdx.y][threadIdx.x]
                                 + c1*in_s[threadIdx.z][threadIdx.y][threadIdx.x-1]
                                 + c2*in_s[threadIdx.z][threadIdx.y][threadIdx.x+1]
                                 + c3*in_s[threadIdx.z][threadIdx.y-1][threadIdx.x]
                                 + c4*in_s[threadIdx.z][threadIdx.y+1][threadIdx.x]
                                 + c5*in_s[threadIdx.z-1][threadIdx.y][threadIdx.x]
                                 + c6*in_s[threadIdx.z+1][threadIdx.y][threadIdx.x];
        }
    }
}

void stencil_3d_parallel_shared_memory(float* in, float* out, unsigned int N,
                                      int c0, int c1, int c2, int c3, int c4, int c5, int c6){
    float *d_in, *d_out;
    cudaError_t error;

    error = cudaMalloc((void**)&d_in, N*N*N * sizeof(float));
    if (error != cudaSuccess) {
        std::cout << "cudaMalloc d_in failed: " << cudaGetErrorString(error) << std::endl;
        return;
    }

    error = cudaMalloc((void**)&d_out, N*N*N * sizeof(float));
    if (error != cudaSuccess) {
        std::cout << "cudaMalloc d_out failed: " << cudaGetErrorString(error) << std::endl;
        return;
    }

    error = cudaMemcpy(d_in, in, N * N * N * sizeof(float), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        std::cout << "cudaMemcpy to device failed: " << cudaGetErrorString(error) << std::endl;
        return;
    }

    dim3 dimBlock(IN_TILE_DIM_SMALL, IN_TILE_DIM_SMALL, IN_TILE_DIM_SMALL);
    dim3 dimGrid(cdiv(N, OUT_TILE_DIM_SMALL), cdiv(N, OUT_TILE_DIM_SMALL), cdiv(N, OUT_TILE_DIM_SMALL));

    stencil_kernel_shared_memory<<<dimGrid, dimBlock>>>(d_in, d_out, N, c0, c1, c2, c3, c4, c5, c6);

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cout << "Kernel launch failed: " << cudaGetErrorString(error) << std::endl;
    }
    cudaDeviceSynchronize();

    error = cudaMemcpy(out, d_out, N * N * N * sizeof(float), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        std::cout << "cudaMemcpy to host failed: " << cudaGetErrorString(error) << std::endl;
    }

    cudaFree(d_in);
    cudaFree(d_out);
}

__global__ void stencil_kernel_thread_coarsening(float* in, float* out, unsigned int N,
                                                int c0, int c1, int c2, int c3, int c4, int c5, int c6) {
    int iStart = blockIdx.z*OUT_TILE_DIM_BIG;
    int j = blockIdx.y*OUT_TILE_DIM_BIG+ threadIdx.y - 1;
    int k = blockIdx.x*OUT_TILE_DIM_BIG+ threadIdx.x - 1;
    __shared__ float inPrev_s[IN_TILE_DIM_BIG][IN_TILE_DIM_BIG];
    __shared__ float inCurr_s[IN_TILE_DIM_BIG][IN_TILE_DIM_BIG];
    __shared__ float inNext_s[IN_TILE_DIM_BIG][IN_TILE_DIM_BIG];
    
    // Initialize shared memory
    inPrev_s[threadIdx.y][threadIdx.x] = 0.0f;
    inCurr_s[threadIdx.y][threadIdx.x] = 0.0f;
    inNext_s[threadIdx.y][threadIdx.x] = 0.0f;
    
    if(iStart-1 >= 0 && iStart-1 < N && j >= 0 && j < N && k >= 0 && k < N) {
        inPrev_s[threadIdx.y][threadIdx.x] = in[(iStart - 1)*N*N + j*N + k];
    }
    if(iStart >= 0 && iStart < N && j >= 0 && j < N && k >= 0 && k < N) {
        inCurr_s[threadIdx.y][threadIdx.x] = in[iStart*N*N + j*N + k];
    }
    for(int i = iStart; i < iStart + OUT_TILE_DIM_BIG; ++i) {
        inNext_s[threadIdx.y][threadIdx.x] = 0.0f;
        if(i + 1 >= 0 && i + 1 < N && j >= 0 && j < N && k >= 0 && k < N) {
            inNext_s[threadIdx.y][threadIdx.x] = in[(i + 1)*N*N + j*N + k];
        }
        __syncthreads();
        if(i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
            if(threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM_BIG - 1
               && threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM_BIG - 1) {
                out[i*N*N + j*N + k] = c0*inCurr_s[threadIdx.y][threadIdx.x]
                                     + c1*inCurr_s[threadIdx.y][threadIdx.x-1]
                                     + c2*inCurr_s[threadIdx.y][threadIdx.x+1]
                                     + c3*inCurr_s[threadIdx.y-1][threadIdx.x]
                                     + c4*inCurr_s[threadIdx.y+1][threadIdx.x]
                                     + c5*inPrev_s[threadIdx.y][threadIdx.x]
                                     + c6*inNext_s[threadIdx.y][threadIdx.x];
            }
        }
        __syncthreads();
        inPrev_s[threadIdx.y][threadIdx.x] = inCurr_s[threadIdx.y][threadIdx.x];
        inCurr_s[threadIdx.y][threadIdx.x] = inNext_s[threadIdx.y][threadIdx.x];
    }
}

void stencil_3d_parallel_thread_coarsening(float* in, float* out, unsigned int N,
                                          int c0, int c1, int c2, int c3, int c4, int c5, int c6){
    float *d_in, *d_out;
    cudaError_t error;

    error = cudaMalloc((void**)&d_in, N*N*N * sizeof(float));
    if (error != cudaSuccess) {
        std::cout << "cudaMalloc d_in failed: " << cudaGetErrorString(error) << std::endl;
        return;
    }

    error = cudaMalloc((void**)&d_out, N*N*N * sizeof(float));
    if (error != cudaSuccess) {
        std::cout << "cudaMalloc d_out failed: " << cudaGetErrorString(error) << std::endl;
        return;
    }

    error = cudaMemcpy(d_in, in, N * N * N * sizeof(float), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        std::cout << "cudaMemcpy to device failed: " << cudaGetErrorString(error) << std::endl;
        return;
    }

    dim3 dimBlock(IN_TILE_DIM_BIG, IN_TILE_DIM_BIG, 1);
    dim3 dimGrid(cdiv(N, OUT_TILE_DIM_BIG), cdiv(N, OUT_TILE_DIM_BIG), cdiv(N, OUT_TILE_DIM_BIG));

    stencil_kernel_thread_coarsening<<<dimGrid, dimBlock>>>(d_in, d_out, N, c0, c1, c2, c3, c4, c5, c6);

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cout << "Kernel launch failed: " << cudaGetErrorString(error) << std::endl;
    }
    cudaDeviceSynchronize();

    error = cudaMemcpy(out, d_out, N * N * N * sizeof(float), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        std::cout << "cudaMemcpy to host failed: " << cudaGetErrorString(error) << std::endl;
    }

    cudaFree(d_in);
    cudaFree(d_out);
}

__global__ void stencil_kernel_register_tiling(float* in, float* out, unsigned int N,
                                              int c0, int c1, int c2, int c3, int c4, int c5, int c6) {
   int iStart = blockIdx.z*OUT_TILE_DIM_SMALL;
   int j = blockIdx.y*OUT_TILE_DIM_SMALL+ threadIdx.y - 1;
   int k = blockIdx.x*OUT_TILE_DIM_SMALL+ threadIdx.x - 1;
   float inPrev;
   __shared__ float inCurr_s[IN_TILE_DIM_SMALL][IN_TILE_DIM_SMALL];
   float inCurr;
   float inNext;
   if(iStart-1 >= 0 && iStart-1 < N && j >= 0 && j < N && k >= 0 && k < N) {
       inPrev = in[(iStart - 1)*N*N + j*N + k];
   }
   
   if(iStart >= 0 && iStart < N && j >= 0 && j < N && k >= 0 && k < N) {
       inCurr = in[iStart*N*N + j*N + k];
       inCurr_s[threadIdx.y][threadIdx.x] = inCurr;
   }
   
   for(int i = iStart; i < iStart + OUT_TILE_DIM_SMALL; ++i) {
       if(i + 1 >= 0 && i + 1 < N && j >= 0 && j < N && k >= 0 && k < N) {
           inNext = in[(i + 1)*N*N + j*N + k];
       }
       
       __syncthreads();
       if(i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
           if(threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM_SMALL - 1
              && threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM_SMALL - 1) {
               out[i*N*N + j*N + k] = c0*inCurr
                                    + c1*inCurr_s[threadIdx.y][threadIdx.x-1]
                                    + c2*inCurr_s[threadIdx.y][threadIdx.x+1]
                                    + c3*inCurr_s[threadIdx.y+1][threadIdx.x]
                                    + c4*inCurr_s[threadIdx.y-1][threadIdx.x]
                                    + c5*inPrev
                                    + c6*inNext;
           }
       }
       __syncthreads();
       inPrev = inCurr;
       inCurr = inNext;
       inCurr_s[threadIdx.y][threadIdx.x] = inNext;
   }
}

void stencil_3d_parallel_register_tiling(float* in, float* out, unsigned int N,
                                        int c0, int c1, int c2, int c3, int c4, int c5, int c6){
    float *d_in, *d_out;
    cudaError_t error;

    error = cudaMalloc((void**)&d_in, N*N*N * sizeof(float));
    if (error != cudaSuccess) {
        std::cout << "cudaMalloc d_in failed: " << cudaGetErrorString(error) << std::endl;
        return;
    }

    error = cudaMalloc((void**)&d_out, N*N*N * sizeof(float));
    if (error != cudaSuccess) {
        std::cout << "cudaMalloc d_out failed: " << cudaGetErrorString(error) << std::endl;
        return;
    }

    error = cudaMemcpy(d_in, in, N * N * N * sizeof(float), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        std::cout << "cudaMemcpy to device failed: " << cudaGetErrorString(error) << std::endl;
        return;
    }

    dim3 dimBlock(IN_TILE_DIM_SMALL, IN_TILE_DIM_SMALL, 1);
    dim3 dimGrid(cdiv(N, OUT_TILE_DIM_SMALL), cdiv(N, OUT_TILE_DIM_SMALL), cdiv(N, OUT_TILE_DIM_SMALL));

    stencil_kernel_register_tiling<<<dimGrid, dimBlock>>>(d_in, d_out, N, c0, c1, c2, c3, c4, c5, c6);

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cout << "Kernel launch failed: " << cudaGetErrorString(error) << std::endl;
    }
    cudaDeviceSynchronize();

    error = cudaMemcpy(out, d_out, N * N * N * sizeof(float), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        std::cout << "cudaMemcpy to host failed: " << cudaGetErrorString(error) << std::endl;
    }

    cudaFree(d_in);
    cudaFree(d_out);
}

bool arrays_allclose(float* a, float* b, unsigned int size, 
                     float rtol = 1e-5f, float atol = 1e-8f) {
    for (unsigned int i = 0; i < size; i++) {
        float diff = fabs(a[i] - b[i]);
        float tolerance = atol + rtol * fmax(fabs(a[i]), fabs(b[i]));
        
        if (diff > tolerance) {
            printf("Mismatch at index %u: a[%u] = %f, b[%u] = %f, diff = %f, tolerance = %f\n", 
                   i, i, a[i], i, b[i], diff, tolerance);
            return false;
        }
    }
    return true;
}

float* generate_random_3d_data(unsigned int N, unsigned int seed = 42) {
    float* data = (float*)malloc(N * N * N * sizeof(float));
    if (data == NULL) {
        printf("Memory allocation failed!\n");
        return NULL;
    }

    srand(seed);
    for (unsigned int i = 0; i < N * N * N; i++) {
        data[i] = ((float)rand() / RAND_MAX) * 100.0f;  // Random values between 0-100
    }
    return data;
}

// Benchmark function for stencil operations
float benchmark_stencil(void (*func)(float*, float*, unsigned int, int, int, int, int, int, int, int), 
                       float* in, float* out, unsigned int N,
                       int c0, int c1, int c2, int c3, int c4, int c5, int c6,
                       int warmup = 5, int reps = 20) {
    unsigned int total_size = N * N * N;
    
    // Warmup runs
    for (int i = 0; i < warmup; ++i) {
        memset(out, 0, total_size * sizeof(float));
        func(in, out, N, c0, c1, c2, c3, c4, c5, c6);
    }

    cudaEvent_t iterStart, iterStop;
    cudaEventCreate(&iterStart);
    cudaEventCreate(&iterStop);

    float totalTime_ms = 0.0f;

    for (int i = 0; i < reps; ++i) {
        clear_l2();
        memset(out, 0, total_size * sizeof(float));
        
        cudaEventRecord(iterStart);
        func(in, out, N, c0, c1, c2, c3, c4, c5, c6);
        cudaEventRecord(iterStop);
        cudaEventSynchronize(iterStop);

        float iterTime = 0.0f;
        cudaEventElapsedTime(&iterTime, iterStart, iterStop);
        totalTime_ms += iterTime;
    }

    cudaEventDestroy(iterStart);
    cudaEventDestroy(iterStop);

    return totalTime_ms / reps;
}

// Special benchmark function for sequential (CPU) implementation
float benchmark_stencil_sequential(void (*func)(float*, float*, unsigned int, int, int, int, int, int, int, int), 
                                  float* in, float* out, unsigned int N,
                                  int c0, int c1, int c2, int c3, int c4, int c5, int c6,
                                  int warmup = 2, int reps = 5) {
    unsigned int total_size = N * N * N;
    
    // Warmup runs
    for (int i = 0; i < warmup; ++i) {
        memset(out, 0, total_size * sizeof(float));
        func(in, out, N, c0, c1, c2, c3, c4, c5, c6);
    }

    struct timespec start, end;
    double totalTime_ms = 0.0;

    for (int i = 0; i < reps; ++i) {
        memset(out, 0, total_size * sizeof(float));
        
        clock_gettime(CLOCK_MONOTONIC, &start);
        func(in, out, N, c0, c1, c2, c3, c4, c5, c6);
        clock_gettime(CLOCK_MONOTONIC, &end);

        double iterTime = (end.tv_sec - start.tv_sec) * 1000.0 + 
                         (end.tv_nsec - start.tv_nsec) / 1000000.0;
        totalTime_ms += iterTime;
    }

    return totalTime_ms / reps;
}

struct BenchmarkResult {
    const char* name;
    float time_ms;
    float* output;
};

int main(int argc, char const* argv[]) {
    // Use different sizes for testing
    std::vector<unsigned int> test_sizes = {32, 64, 128};
    
    for (unsigned int N : test_sizes) {
        printf("\n================================================================================\n");
        printf("Benchmarking 3D Stencil Operations - Grid Size: %dx%dx%d\n", N, N, N);
        printf("================================================================================\n");
        
        unsigned int total_size = N * N * N;
        
        // Allocate memory for input and outputs
        float* in = generate_random_3d_data(N);
        if (in == NULL) {
            printf("Failed to generate input data\n");
            continue;
        }
        
        // Allocate separate output arrays for each implementation
        float* out_sequential = (float*)malloc(total_size * sizeof(float));
        float* out_basic = (float*)malloc(total_size * sizeof(float));
        float* out_shared = (float*)malloc(total_size * sizeof(float));
        float* out_coarsening = (float*)malloc(total_size * sizeof(float));
        float* out_register = (float*)malloc(total_size * sizeof(float));
        
        if (!out_sequential || !out_basic || !out_shared || !out_coarsening || !out_register) {
            printf("Memory allocation failed!\n");
            free(in);
            continue;
        }

        printf("Configuration:\n");
        printf("Grid size: %dx%dx%d\n", N, N, N);
        printf("Total elements: %u\n", total_size);
        printf("Memory per array: %.2f MB\n", (total_size * sizeof(float)) / (1024.0f * 1024.0f));
        printf("OUT_TILE_DIM: %d, IN_TILE_DIM: %d\n\n", OUT_TILE_DIM_SMALL, OUT_TILE_DIM_SMALL);

        std::vector<BenchmarkResult> results;
        
        float sequential_time = benchmark_stencil_sequential(stencil_3d_sequential, in, out_sequential, N, 
                                                           c0, c1, c2, c3, c4, c5, c6);
        results.push_back({"Sequential", sequential_time, out_sequential});
        
        float basic_time = benchmark_stencil(stencil_3d_parallel_basic, in, out_basic, N, 
                                           c0, c1, c2, c3, c4, c5, c6);
        results.push_back({"Parallel Basic", basic_time, out_basic});
        
        float shared_time = benchmark_stencil(stencil_3d_parallel_shared_memory, in, out_shared, N, 
                                            c0, c1, c2, c3, c4, c5, c6);
        results.push_back({"Shared Memory", shared_time, out_shared});
        
        float coarsening_time = benchmark_stencil(stencil_3d_parallel_thread_coarsening, in, out_coarsening, N, 
                                                c0, c1, c2, c3, c4, c5, c6);
        results.push_back({"Thread Coarsening", coarsening_time, out_coarsening});
        
        float register_time = benchmark_stencil(stencil_3d_parallel_register_tiling, in, out_register, N, 
                                              c0, c1, c2, c3, c4, c5, c6);
        results.push_back({"Register Tiling", register_time, out_register});

        printf("\nResults:\n");
        printf("Implementation           | Time (ms) | Speedup vs Sequential | Speedup vs Basic\n");
        printf("-------------------------|-----------|----------------------|------------------\n");
        
        for (const auto& result : results) {
            float speedup_vs_seq = sequential_time / result.time_ms;
            float speedup_vs_basic = basic_time / result.time_ms;
            printf("%-23s | %8.3f  | %19.2fx | %15.2fx\n", 
                   result.name, result.time_ms, speedup_vs_seq, speedup_vs_basic);
        }

        // Verify correctness - compare all results against sequential
        printf("\nCorrectness Verification:\n");
        bool all_correct = true;
        
        for (size_t i = 1; i < results.size(); i++) {
            bool correct = arrays_allclose(out_sequential, results[i].output, total_size);
            printf("%s vs Sequential: %s\n", results[i].name, correct ? "✓ PASS" : "✗ FAIL");
            if (!correct) all_correct = false;
        }
        
        printf("\nOverall correctness: %s\n", all_correct ? "✓ All implementations correct" : "✗ Some implementations incorrect");

        free(in);
        free(out_sequential);
        free(out_basic);
        free(out_shared);
        free(out_coarsening);
        free(out_register);
    }

    return 0;
}