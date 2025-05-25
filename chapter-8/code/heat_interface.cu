#include <cstring>

#include "stencil.h"

// C interface for Python to call
extern "C" {

// Function to perform one heat equation timestep
void heat_step_cuda(float* data, unsigned int N, float alpha, float dt, float dx) {
    // Calculate stencil coefficients for heat equation
    // 3D heat equation: ∂u/∂t = α(∂²u/∂x² + ∂²u/∂y² + ∂²u/∂z²)
    // Finite difference discretization with dt time step and dx spatial step

    float r = alpha * dt / (dx * dx);  // Diffusion number

    // For stability, we need r <= 1/6 for 3D
    if (r > 1.0f / 6.0f) {
        printf("Warning: Diffusion number r = %f > 1/6, simulation may be unstable!\n", r);
    }

    // Stencil coefficients for heat equation
    int c0_heat = (int)((1.0f - 6.0f * r) * 1000.0f);  // Scale for integer arithmetic
    int c1_heat = (int)(r * 1000.0f);
    int c2_heat = (int)(r * 1000.0f);
    int c3_heat = (int)(r * 1000.0f);
    int c4_heat = (int)(r * 1000.0f);
    int c5_heat = (int)(r * 1000.0f);
    int c6_heat = (int)(r * 1000.0f);

    // Allocate output array
    float* output = (float*)malloc(N * N * N * sizeof(float));
    if (!output) {
        printf("Memory allocation failed!\n");
        return;
    }

    // Apply stencil using register tiling implementation
    stencil_3d_parallel_register_tiling(data, output, N, c0_heat, c1_heat, c2_heat, c3_heat, c4_heat, c5_heat, c6_heat);

    // Copy result back to input array (scale back from integer arithmetic)
    unsigned int total_size = N * N * N;
    for (unsigned int i = 0; i < total_size; i++) {
        data[i] = output[i] / 1000.0f;
    }

    free(output);
}

// Alternative version using floating point coefficients directly
void heat_step_cuda_float(float* data, unsigned int N, float alpha, float dt, float dx) {
    float r = alpha * dt / (dx * dx);

    if (r > 1.0f / 6.0f) {
        printf("Warning: Diffusion number r = %f > 1/6, simulation may be unstable!\n", r);
    }

    // Use small integer approximation to work with existing integer interface
    // Scale coefficients by 10000 for better precision
    int scale = 10000;
    int c0_heat = (int)((1.0f - 6.0f * r) * scale);
    int c1_heat = (int)(r * scale);
    int c2_heat = (int)(r * scale);
    int c3_heat = (int)(r * scale);
    int c4_heat = (int)(r * scale);
    int c5_heat = (int)(r * scale);
    int c6_heat = (int)(r * scale);

    float* output = (float*)malloc(N * N * N * sizeof(float));
    if (!output) {
        printf("Memory allocation failed!\n");
        return;
    }

    stencil_3d_parallel_register_tiling(data, output, N, c0_heat, c1_heat, c2_heat, c3_heat, c4_heat, c5_heat, c6_heat);

    // Copy result back and scale
    unsigned int total_size = N * N * N;
    for (unsigned int i = 0; i < total_size; i++) {
        data[i] = output[i] / scale;
    }

    free(output);
}

// Initialize CUDA (call once at startup)
void init_cuda() {
    cudaError_t error = cudaSetDevice(0);
    if (error != cudaSuccess) {
        printf("CUDA initialization failed: %s\n", cudaGetErrorString(error));
    }
}

// Cleanup CUDA (call at shutdown)
void cleanup_cuda() {
    cudaDeviceReset();
}

}  // extern "C"
