#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int c0 = 0;
int c1 = 1;
int c2 = 1;
int c3 = 1;
int c4 = 1;
int c5 = 1;
int c6 = 1;

void stencil_3d_sequential(float* in, float* out, unsigned int N) {
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

void print_3d_slice(float* data, int N, int slice_i) {
    printf("Slice i=%d:\n", slice_i);
    for (int j = 0; j < N; j++) {
        for (int k = 0; k < N; k++) {
            printf("%6.1f ", data[slice_i * N * N + j * N + k]);
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    // Test with a small 4x4x4 grid
    unsigned int N = 4;
    int total_size = N * N * N;
    
    // Allocate memory
    float* in = (float*)malloc(total_size * sizeof(float));
    float* out = (float*)malloc(total_size * sizeof(float));
    
    // Initialize input data with simple pattern
    // Set all to 0 first
    memset(in, 0, total_size * sizeof(float));
    memset(out, 0, total_size * sizeof(float));
    
    // Put a "hot spot" in the center
    in[1 * N * N + 1 * N + 1] = 10.0f;  // Center point
    in[1 * N * N + 1 * N + 2] = 5.0f;   // Adjacent points
    in[1 * N * N + 2 * N + 1] = 5.0f;
    in[2 * N * N + 1 * N + 1] = 5.0f;
    
    printf("Input data:\n");
    for (int i = 0; i < N; i++) {
        print_3d_slice(in, N, i);
    }
    
    // Run the stencil
    stencil_3d_sequential(in, out, N);
    
    printf("Output data:\n");
    for (int i = 0; i < N; i++) {
        print_3d_slice(out, N, i);
    }
    
    return 0;
}