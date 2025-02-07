#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <iostream>
#include <iomanip>
#include <cmath>

#define SECTION_SIZE 4

// Utility function for division rounding up
inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

// Error checking macro
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Custom implementation
__global__ void hierarchical_kogge_stone_domino_exclusive(float* X, float* Y, float* scan_value, int* flags, int* blockCounter,
    unsigned int N) {
    extern __shared__ float buffer[];
    __shared__ unsigned int bid_s;
    __shared__ float previous_sum;
    const unsigned int tid = threadIdx.x;

    // DEADLOCK PREVENTION: Dynamic block index assignment
    if (tid == 0) {
        bid_s = atomicAdd(blockCounter, 1);
    }
    __syncthreads();
    const unsigned int bid = bid_s;
    const unsigned int gid = bid * blockDim.x + tid;

    // Phase 1: Local block scan using Kogge-Stone
    if (gid < N) {
        buffer[tid] = X[gid];
    } else {
        buffer[tid] = 0.0f;
    }

    // Store original value for later use
    float original_value = buffer[tid];
    
    // Kogge-Stone scan within block
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        float temp = buffer[tid];
        if (tid >= stride) {
            temp += buffer[tid - stride];
        }
        __syncthreads();
        buffer[tid] = temp;
    }

    // Convert to exclusive scan by shifting right
    __syncthreads();
    float exclusive_value;
    if (tid == 0) {
        exclusive_value = 0.0f;
    } else {
        exclusive_value = buffer[tid - 1];
    }

    // Store local result
    if (gid < N) {
        Y[gid] = exclusive_value;
    }

    // Get local sum for this block
    const float local_sum = buffer[blockDim.x - 1];

    // Phase 2: Inter-block sum propagation
    if (tid == 0) {
        if (bid > 0) {
            // Wait for previous block's flag
            while (atomicAdd(&flags[bid], 0) == 0) { }
            
            // Get sum from previous block
            previous_sum = scan_value[bid];
            
            // Add local sum and propagate
            const float total_sum = previous_sum + local_sum;
            scan_value[bid + 1] = total_sum;
            
            // Ensure scan_value is visible
            __threadfence();
            
            // Signal next block
            atomicAdd(&flags[bid + 1], 1);
        } else {
            // First block just propagates its sum
            scan_value[1] = local_sum;
            __threadfence();
            atomicAdd(&flags[1], 1);
        }
    }
    __syncthreads();

    // Phase 3: Add previous block's sum to local results
    if (bid > 0 && gid < N) {
        Y[gid] += previous_sum;
    }
}

void hierarchical_exclusive_scan_with_domino_like_sync(float* X, float* Y, unsigned int N) {
    float *d_X, *d_Y, *d_scan_value;
    int *d_flags, *d_blockCounter;
    const unsigned int block_size = SECTION_SIZE;
    const unsigned int num_blocks = cdiv(N, block_size);

    gpuErrchk(cudaMalloc((void**)&d_X, N * sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&d_Y, N * sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&d_scan_value, (num_blocks + 1) * sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&d_flags, (num_blocks + 1) * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&d_blockCounter, sizeof(int)));

    gpuErrchk(cudaMemset(d_flags, 0, (num_blocks + 1) * sizeof(int)));
    gpuErrchk(cudaMemset(d_blockCounter, 0, sizeof(int)));
    gpuErrchk(cudaMemcpy(d_X, X, N * sizeof(float), cudaMemcpyHostToDevice));

    hierarchical_kogge_stone_domino_exclusive<<<num_blocks, block_size, block_size * sizeof(float)>>>(
        d_X, d_Y, d_scan_value, d_flags, d_blockCounter, N);

    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(Y, d_Y, N * sizeof(float), cudaMemcpyDeviceToHost));

    gpuErrchk(cudaFree(d_X));
    gpuErrchk(cudaFree(d_Y));
    gpuErrchk(cudaFree(d_scan_value));
    gpuErrchk(cudaFree(d_flags));
    gpuErrchk(cudaFree(d_blockCounter));
}

bool compareResults(float* custom_result, thrust::host_vector<float>& thrust_result, unsigned int N) {
    const float epsilon = 1e-5f;  // Tolerance for floating point comparison
    bool match = true;
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\nComparison of results:\n";
    std::cout << "Index\tCustom\t\tThrust\t\tDiff\n";
    std::cout << "----------------------------------------\n";
    
    for(unsigned int i = 0; i < N; i++) {
        float diff = std::abs(custom_result[i] - thrust_result[i]);
        if(diff > epsilon) {
            match = false;
        }
        if(i < 10 || !match) {  // Print first 10 elements or if there's a mismatch
            std::cout << i << "\t" 
                     << custom_result[i] << "\t\t"
                     << thrust_result[i] << "\t\t"
                     << diff << "\n";
        }
    }
    
    return match;
}

int main() {
    const unsigned int N = 1024;  // Test size
    
    // Initialize input data
    float* h_input = new float[N];
    float* h_custom_output = new float[N];
    
    // Fill with test data
    for(unsigned int i = 0; i < N; i++) {
        h_input[i] = 1.0f;  // Using 1.0 for easy verification
    }
    
    // Run custom implementation
    hierarchical_exclusive_scan_with_domino_like_sync(h_input, h_custom_output, N);
    
    // Run Thrust implementation
    thrust::host_vector<float> thrust_input(N);
    for(unsigned int i = 0; i < N; i++) {
        thrust_input[i] = h_input[i];
    }
    
    thrust::device_vector<float> d_input = thrust_input;
    thrust::device_vector<float> d_output(N);
    
    thrust::exclusive_scan(
        d_input.begin(),
        d_input.end(),
        d_output.begin(),
        0.0f  // Initial value
    );
    
    thrust::host_vector<float> thrust_output = d_output;
    
    // Compare results
    bool results_match = compareResults(h_custom_output, thrust_output, N);
    
    if(results_match) {
        std::cout << "\nResults match within tolerance! ✓\n";
    } else {
        std::cout << "\nResults differ! ✗\n";
    }
    
    // Cleanup
    delete[] h_input;
    delete[] h_custom_output;
    
    return 0;
}