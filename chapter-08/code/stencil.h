#ifndef CHAPTER_08_CODE_STENCIL_H_
#define CHAPTER_08_CODE_STENCIL_H_

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// some of our kernels have qubic requirement for the shared memory, other have squared, hence we introduce two block
// sizes
#define OUT_TILE_DIM_SMALL 8
#define IN_TILE_DIM_SMALL (OUT_TILE_DIM_SMALL + 2)
#define OUT_TILE_DIM_BIG 30
#define IN_TILE_DIM_BIG (OUT_TILE_DIM_BIG + 2)

extern int c0, c1, c2, c3, c4, c5, c6;

// CUDA error checking macros
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

// Utility functions
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) {
            exit(code);
        }
    }
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

// Stencil implementations
void stencil_3d_sequential(float* in, float* out, unsigned int N, int c0, int c1, int c2, int c3, int c4, int c5,
                           int c6);

void stencil_3d_parallel_basic(float* in, float* out, unsigned int N, int c0, int c1, int c2, int c3, int c4, int c5,
                               int c6);

void stencil_3d_parallel_shared_memory(float* in, float* out, unsigned int N, int c0, int c1, int c2, int c3, int c4,
                                       int c5, int c6);

void stencil_3d_parallel_thread_coarsening(float* in, float* out, unsigned int N, int c0, int c1, int c2, int c3,
                                           int c4, int c5, int c6);

void stencil_3d_parallel_register_tiling(float* in, float* out, unsigned int N, int c0, int c1, int c2, int c3, int c4,
                                         int c5, int c6);

#endif  // CHAPTER_08_CODE_STENCIL_H_
