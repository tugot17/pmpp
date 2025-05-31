#include <cuda.h>
#include <stdio.h>
#define MAX_TESS_POINTS 32

// Structure compatible with Python ctypes (flat float arrays)
struct BezierLine {
    float CP[6];                           // Control points: [x0,y0,x1,y1,x2,y2]
    float vertexPos[MAX_TESS_POINTS * 2];  // Vertex positions: [x0,y0,x1,y1,...]
    int nVertices;                         // Number of tessellated vertices
};

__device__ float computeCurvature(float* cp) {
    float dx = cp[4] - cp[0];  // cp[2].x - cp[0].x
    float dy = cp[5] - cp[1];  // cp[2].y - cp[0].y
    float line_length = sqrtf(dx * dx + dy * dy);
    if (line_length < 0.001f) {
        return 0.0f;
    }
    float cross =
        fabsf((cp[2] - cp[0]) * dy - (cp[3] - cp[1]) * dx);  // (cp[1].x - cp[0].x)*dy - (cp[1].y - cp[0].y)*dx
    return cross / line_length;
}

// Child kernel for dynamic parallelism - writes directly to main structure
__global__ void computeBezierLine_child_direct(int lidx, BezierLine* bLines, int nTessPoints) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < nTessPoints) {
        float u = (float)idx / (float)(nTessPoints - 1);
        float omu = 1.0f - u;
        float B3u[3];
        B3u[0] = omu * omu;
        B3u[1] = 2.0f * u * omu;
        B3u[2] = u * u;

        float pos_x = 0.0f, pos_y = 0.0f;
        for (int i = 0; i < 3; i++) {
            pos_x += B3u[i] * bLines[lidx].CP[i * 2];      // CP[i].x
            pos_y += B3u[i] * bLines[lidx].CP[i * 2 + 1];  // CP[i].y
        }
        bLines[lidx].vertexPos[idx * 2] = pos_x;      // Store x
        bLines[lidx].vertexPos[idx * 2 + 1] = pos_y;  // Store y
    }
}

// Parent kernel using dynamic parallelism
__global__ void computeBezierLines_dynamic(BezierLine* bLines, int nLines) {
    int lidx = threadIdx.x + blockDim.x * blockIdx.x;
    if (lidx < nLines) {
        // Compute curvature and determine tessellation points
        float curvature = computeCurvature(bLines[lidx].CP);
        int nTessPoints = min(max((int)(curvature * 16.0f), 4), MAX_TESS_POINTS);
        bLines[lidx].nVertices = nTessPoints;

        // Launch child kernel to compute tessellation directly into main structure
        int childBlocks = (nTessPoints + 31) / 32;
        computeBezierLine_child_direct<<<childBlocks, 32>>>(lidx, bLines, nTessPoints);

        // Child kernels are automatically synchronized when parent kernel completes
    }
}

// Original static version for comparison (updated to use flat arrays)
__global__ void computeBezierLines_static(BezierLine* bLines, int nLines) {
    int bidx = blockIdx.x;
    if (bidx < nLines) {
        float curvature = computeCurvature(bLines[bidx].CP);
        int nTessPoints = min(max((int)(curvature * 16.0f), 4), MAX_TESS_POINTS);
        bLines[bidx].nVertices = nTessPoints;

        for (int inc = 0; inc < nTessPoints; inc += blockDim.x) {
            int idx = inc + threadIdx.x;
            if (idx < nTessPoints) {
                float u = (float)idx / (float)(nTessPoints - 1);
                float omu = 1.0f - u;
                float B3u[3];
                B3u[0] = omu * omu;
                B3u[1] = 2.0f * u * omu;
                B3u[2] = u * u;

                float pos_x = 0.0f, pos_y = 0.0f;
                for (int i = 0; i < 3; i++) {
                    pos_x += B3u[i] * bLines[bidx].CP[i * 2];      // CP[i].x
                    pos_y += B3u[i] * bLines[bidx].CP[i * 2 + 1];  // CP[i].y
                }
                bLines[bidx].vertexPos[idx * 2] = pos_x;      // Store x
                bLines[bidx].vertexPos[idx * 2 + 1] = pos_y;  // Store y
            }
        }
    }
}

// Host wrapper functions
extern "C" {

// Original static version (same as before)
int tessellate_bezier_curves(BezierLine* lines, int num_lines) {
    BezierLine* d_lines;
    size_t size = num_lines * sizeof(BezierLine);

    if (cudaMalloc(&d_lines, size) != cudaSuccess) {
        return -1;
    }
    if (cudaMemcpy(d_lines, lines, size, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d_lines);
        return -2;
    }

    computeBezierLines_static<<<num_lines, 32>>>(d_lines, num_lines);
    cudaDeviceSynchronize();

    if (cudaMemcpy(lines, d_lines, size, cudaMemcpyDeviceToHost) != cudaSuccess) {
        cudaFree(d_lines);
        return -3;
    }

    cudaFree(d_lines);
    return 0;
}

// Dynamic parallelism version (Python compatible)
int tessellate_bezier_curves_dynamic(BezierLine* lines, int num_lines) {
    BezierLine* d_lines;
    size_t size = num_lines * sizeof(BezierLine);

    if (cudaMalloc(&d_lines, size) != cudaSuccess) {
        return -1;
    }
    if (cudaMemcpy(d_lines, lines, size, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d_lines);
        return -2;
    }

    // Launch parent kernel with dynamic parallelism
    int blocks = (num_lines + 31) / 32;
    computeBezierLines_dynamic<<<blocks, 32>>>(d_lines, num_lines);
    cudaDeviceSynchronize();

    if (cudaMemcpy(lines, d_lines, size, cudaMemcpyDeviceToHost) != cudaSuccess) {
        cudaFree(d_lines);
        return -3;
    }

    cudaFree(d_lines);
    return 0;
}
}
