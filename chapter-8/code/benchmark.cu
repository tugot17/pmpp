#include <math.h>
#include <string.h>
#include <time.h>

#include <functional>
#include <iostream>
#include <vector>

#include "stencil.h"

struct BenchmarkResult {
    const char* name;
    float time_ms;
    float* output;
};

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

bool arrays_allclose(float* a, float* b, unsigned int size, float rtol = 1e-5f, float atol = 1e-8f) {
    for (unsigned int i = 0; i < size; i++) {
        float diff = fabs(a[i] - b[i]);
        float tolerance = atol + rtol * fmax(fabs(a[i]), fabs(b[i]));

        if (diff > tolerance) {
            printf("Mismatch at index %u: a[%u] = %f, b[%u] = %f, diff = %f, tolerance = %f\n", i, i, a[i], i, b[i],
                   diff, tolerance);
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
float benchmark_stencil(void (*func)(float*, float*, unsigned int, int, int, int, int, int, int, int), float* in,
                        float* out, unsigned int N, int c0, int c1, int c2, int c3, int c4, int c5, int c6,
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
                                   float* in, float* out, unsigned int N, int c0, int c1, int c2, int c3, int c4,
                                   int c5, int c6, int warmup = 2, int reps = 5) {
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

        double iterTime = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_nsec - start.tv_nsec) / 1000000.0;
        totalTime_ms += iterTime;
    }

    return totalTime_ms / reps;
}

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

        float sequential_time =
            benchmark_stencil_sequential(stencil_3d_sequential, in, out_sequential, N, c0, c1, c2, c3, c4, c5, c6);
        results.push_back({"Sequential", sequential_time, out_sequential});

        float basic_time = benchmark_stencil(stencil_3d_parallel_basic, in, out_basic, N, c0, c1, c2, c3, c4, c5, c6);
        results.push_back({"Parallel Basic", basic_time, out_basic});

        float shared_time =
            benchmark_stencil(stencil_3d_parallel_shared_memory, in, out_shared, N, c0, c1, c2, c3, c4, c5, c6);
        results.push_back({"Shared Memory", shared_time, out_shared});

        float coarsening_time =
            benchmark_stencil(stencil_3d_parallel_thread_coarsening, in, out_coarsening, N, c0, c1, c2, c3, c4, c5, c6);
        results.push_back({"Thread Coarsening", coarsening_time, out_coarsening});

        float register_time =
            benchmark_stencil(stencil_3d_parallel_register_tiling, in, out_register, N, c0, c1, c2, c3, c4, c5, c6);
        results.push_back({"Register Tiling", register_time, out_register});

        printf("\nResults:\n");
        printf("Implementation           | Time (ms) | Speedup vs Sequential | Speedup vs Basic\n");
        printf("-------------------------|-----------|----------------------|------------------\n");

        for (const auto& result : results) {
            float speedup_vs_seq = sequential_time / result.time_ms;
            float speedup_vs_basic = basic_time / result.time_ms;
            printf("%-23s | %8.3f  | %19.2fx | %15.2fx\n", result.name, result.time_ms, speedup_vs_seq,
                   speedup_vs_basic);
        }

        // Verify correctness - compare all results against sequential
        printf("\nCorrectness Verification:\n");
        bool all_correct = true;

        for (size_t i = 1; i < results.size(); i++) {
            bool correct = arrays_allclose(out_sequential, results[i].output, total_size);
            printf("%s vs Sequential: %s\n", results[i].name, correct ? "✓ PASS" : "✗ FAIL");
            if (!correct) {
                all_correct = false;
            }
        }

        printf("\nOverall correctness: %s\n",
               all_correct ? "✓ All implementations correct" : "✗ Some implementations incorrect");

        free(in);
        free(out_sequential);
        free(out_basic);
        free(out_shared);
        free(out_coarsening);
        free(out_register);
    }

    return 0;
}
