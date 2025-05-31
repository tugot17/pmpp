#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define MAX_TESS_POINTS 32

// Single unified structure - no separate C/CUDA versions needed
struct BezierCurve {
    float cp[6];                    // Control points: [x0,y0,x1,y1,x2,y2]
    float vertices[MAX_TESS_POINTS*2]; // Tessellated vertices: [x0,y0,x1,y1,...]
    int num_vertices;               // Number of tessellated vertices
};

// Simple curvature calculation
__device__ float calculate_curvature(float* cp) {
    // Distance from middle control point to line between endpoints
    float dx = cp[4] - cp[0];  // x2 - x0
    float dy = cp[5] - cp[1];  // y2 - y0
    float line_length = sqrtf(dx*dx + dy*dy);
    
    if (line_length < 0.001f) return 0.0f;
    
    // Distance from control point to line
    float cross = fabsf((cp[2]-cp[0])*dy - (cp[3]-cp[1])*dx);
    return cross / line_length;
}

// One thread per curve - much simpler!
__global__ void tessellate_curves(BezierCurve* curves, int num_curves) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_curves) return;
    
    BezierCurve* curve = &curves[idx];
    
    // Calculate tessellation level based on curvature
    float curvature = calculate_curvature(curve->cp);
    int num_points = fminf(fmaxf((int)(curvature * 16.0f), 4), MAX_TESS_POINTS);
    curve->num_vertices = num_points;
    
    // Tessellate the curve
    for (int i = 0; i < num_points; i++) {
        float t = (float)i / (float)(num_points - 1);
        float t2 = t * t;
        float mt = 1.0f - t;
        float mt2 = mt * mt;
        
        // Quadratic Bezier formula
        float x = mt2 * curve->cp[0] + 2*mt*t * curve->cp[2] + t2 * curve->cp[4];
        float y = mt2 * curve->cp[1] + 2*mt*t * curve->cp[3] + t2 * curve->cp[5];
        
        curve->vertices[i*2] = x;
        curve->vertices[i*2+1] = y;
    }
}

// Simplified host interface
extern "C" {
    int tessellate_bezier_curves(BezierCurve* curves, int num_curves) {
        BezierCurve* d_curves;
        size_t size = num_curves * sizeof(BezierCurve);
        
        // Allocate and copy to device
        if (cudaMalloc(&d_curves, size) != cudaSuccess) return -1;
        if (cudaMemcpy(d_curves, curves, size, cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(d_curves);
            return -2;
        }
        
        // Launch kernel: one thread per curve
        int threads = min(256, num_curves);
        int blocks = (num_curves + threads - 1) / threads;
        tessellate_curves<<<blocks, threads>>>(d_curves, num_curves);
        
        // Copy results back
        if (cudaMemcpy(curves, d_curves, size, cudaMemcpyDeviceToHost) != cudaSuccess) {
            cudaFree(d_curves);
            return -3;
        }
        
        cudaFree(d_curves);
        return 0;
    }
}