#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <iostream>
#include <iomanip>
#include <cmath>

#define SECTION_SIZE 4

inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void hierarchical_kogge_stone_domino_exclusive_inplace(float* X, float* scan_value, int* flags, int* blockCounter, unsigned int N) {
    extern __shared__ float buffer[];
    __shared__ unsigned int bid_s;
    __shared__ float previous_sum;
    const unsigned int tid = threadIdx.x;

    if (tid == 0) {
        bid_s = atomicAdd(blockCounter, 1);
    }
    __syncthreads();
    const unsigned int bid = bid_s;
    const unsigned int gid = bid * blockDim.x + tid;

    // Phase 1: Load and local scan (unchanged)
    if (gid < N) {
        buffer[tid] = X[gid];
    } else {
        buffer[tid] = 0.0f;
    }
    __syncthreads();

    // Kogge-Stone scan within block (unchanged)
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        float temp = buffer[tid];
        if (tid >= stride) {
            temp += buffer[tid - stride];
        }
        __syncthreads();
        buffer[tid] = temp;
    }

    float exclusive_value;
    if (tid == 0) {
        exclusive_value = 0.0f;
    } else {
        exclusive_value = buffer[tid - 1];
    }

    // Store block's total sum
    const float local_sum = buffer[blockDim.x - 1];
    
    // Phase 2: Inter-block sum propagation
    if (tid == 0) {
        // Store this block's sum
        scan_value[bid] = local_sum;
        __threadfence();
        atomicAdd(&flags[bid], 1);
        
        if (bid > 0) {
            // Wait for all previous blocks
            for (int prev_bid = 0; prev_bid < bid; prev_bid++) {
                while (atomicAdd(&flags[prev_bid], 0) == 0) { }
            }
            
            // Accumulate all previous blocks' sums
            previous_sum = 0.0f;
            for (int prev_bid = 0; prev_bid < bid; prev_bid++) {
                previous_sum += scan_value[prev_bid];
            }
        } else {
            previous_sum = 0.0f;
        }
    }
    __syncthreads();

    // Phase 3: Write final result
    if (gid < N) {
        X[gid] = exclusive_value + previous_sum;
    }
}

void hierarchical_exclusive_scan_inplace(float* X, unsigned int N) {
    float *d_X, *d_scan_value;
    int *d_flags, *d_blockCounter;
    const unsigned int block_size = SECTION_SIZE;
    const unsigned int num_blocks = cdiv(N, block_size);

    // Allocate device memory
    gpuErrchk(cudaMalloc((void**)&d_X, N * sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&d_scan_value, (num_blocks + 1) * sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&d_flags, (num_blocks + 1) * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&d_blockCounter, sizeof(int)));

    // Initialize device memory
    gpuErrchk(cudaMemset(d_flags, 0, (num_blocks + 1) * sizeof(int)));
    gpuErrchk(cudaMemset(d_blockCounter, 0, sizeof(int)));
    gpuErrchk(cudaMemcpy(d_X, X, N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    hierarchical_kogge_stone_domino_exclusive_inplace<<<num_blocks, block_size, block_size * sizeof(float)>>>(
        d_X, d_scan_value, d_flags, d_blockCounter, N);

    // Copy results back and cleanup
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(X, d_X, N * sizeof(float), cudaMemcpyDeviceToHost));

    gpuErrchk(cudaFree(d_X));
    gpuErrchk(cudaFree(d_scan_value));
    gpuErrchk(cudaFree(d_flags));
    gpuErrchk(cudaFree(d_blockCounter));
}

int main() {
    const unsigned int N = 1024;
    
    // Initialize input data
    float* h_data = new float[N];
    float* h_data_copy = new float[N];  // For Thrust comparison
    
    for(unsigned int i = 0; i < N; i++) {
        h_data[i] = 1.0f;
        h_data_copy[i] = 1.0f;
    }
    
    // Run custom implementation in-place
    hierarchical_exclusive_scan_inplace(h_data, N);
    
    // Run Thrust implementation for comparison
    thrust::device_vector<float> d_input(h_data_copy, h_data_copy + N);
    thrust::device_vector<float> d_output(N);
    
    thrust::exclusive_scan(
        d_input.begin(),
        d_input.end(),
        d_output.begin(),
        0.0f
    );
    
    thrust::host_vector<float> thrust_output = d_output;
    
    // Compare results
    bool match = true;
    const float epsilon = 1e-5f;
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\nComparison of results:\n";
    std::cout << "Index\tCustom\t\tThrust\t\tDiff\n";
    std::cout << "----------------------------------------\n";
    
    for(unsigned int i = 0; i < N; i++) {
        float diff = std::abs(h_data[i] - thrust_output[i]);
        if(diff > epsilon) {
            match = false;
        }
        if(i < 10 || !match) {
            std::cout << i << "\t" 
                     << h_data[i] << "\t\t"
                     << thrust_output[i] << "\t\t"
                     << diff << "\n";
        }
    }
    
    if(match) {
        std::cout << "\nResults match within tolerance! ✓\n";
    } else {
        std::cout << "\nResults differ! ✗\n";
    }
    
    delete[] h_data;
    delete[] h_data_copy;
    
    return 0;
}