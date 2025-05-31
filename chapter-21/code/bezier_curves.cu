#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "helper_math.h"  // Provides float2 operators (+, -, *, /, length, dot, etc.)

#define MAX_TESS_POINTS 32

// A structure containing all parameters needed to tessellate a Bezier line
struct BezierLine {
   float2 CP[3];                    //Control points for the line
   float2 vertexPos[MAX_TESS_POINTS]; //Vertex position array to tessellate into
   int nVertices;                   //Number of tessellated vertices
};

// C-compatible structure for Python interface
extern "C" {
    struct BezierLineC {
        float CP[6];        // 3 control points * 2 coordinates (x,y)
        float vertexPos[64]; // MAX_TESS_POINTS * 2 coordinates
        int nVertices;
    };
}

__device__ float computeCurvature(BezierLine* bLine) {
    float2 P0 = bLine->CP[0];  // Start point
    float2 P1 = bLine->CP[1];  // Control point  
    float2 P2 = bLine->CP[2];  // End point
    
    // Distance from control point to the straight line (chord) P0-P2
    float2 chord = P2 - P0;
    float2 toControl = P1 - P0;
    
    // Project toControl onto chord, then find perpendicular distance
    float chordLength = length(chord);
    if(chordLength < 0.001f) return 0.0f; // Degenerate case
    
    float2 chordNorm = chord / chordLength;
    float projection = dot(toControl, chordNorm);
    float2 perpendicular = toControl - projection * chordNorm;
    
    return length(perpendicular);
}

__global__ void computeBezierLines(BezierLine *bLines, int nLines) {
   int bidx = blockIdx.x;
   if(bidx < nLines){
       //Compute the curvature of the line
       float curvature = computeCurvature(&bLines[bidx]);

       //From the curvature, compute the number of tessellation points
       int nTessPoints = min(max((int)(curvature*16.0f),4),32);
       bLines[bidx].nVertices = nTessPoints;

       //Loop through vertices to be tessellated, incrementing by blockDim.x
       for(int inc = 0; inc < nTessPoints; inc += blockDim.x){
           int idx = inc + threadIdx.x;  //Compute a unique index for this point
           if(idx < nTessPoints){
               float u = (float)idx/(float)(nTessPoints-1);  //Compute u from idx
               float omu = 1.0f - u;    //pre-compute one minus u
               float B3u[3]; //Compute quadratic Bezier coefficients
               B3u[0] = omu*omu;
               B3u[1] = 2.0f*u*omu;
               B3u[2] = u*u;
               float2 position = make_float2(0.0f, 0.0f);  //Set position to zero
               for(int i = 0; i < 3; i++){
                   //Add the contribution of the i'th control point to position
                   position = position + B3u[i] * bLines[bidx].CP[i];
               }
               //Assign value of vertex position to the correct array element
               bLines[bidx].vertexPos[idx] = position;
           }
       }
   }
}

// Helper function to convert between C and CUDA structures
void convertCToCuda(BezierLineC* c_lines, BezierLine* cuda_lines, int nLines) {
    for(int i = 0; i < nLines; i++) {
        cuda_lines[i].CP[0] = make_float2(c_lines[i].CP[0], c_lines[i].CP[1]);
        cuda_lines[i].CP[1] = make_float2(c_lines[i].CP[2], c_lines[i].CP[3]);
        cuda_lines[i].CP[2] = make_float2(c_lines[i].CP[4], c_lines[i].CP[5]);
        cuda_lines[i].nVertices = 0;
    }
}

void convertCudaToC(BezierLine* cuda_lines, BezierLineC* c_lines, int nLines) {
    for(int i = 0; i < nLines; i++) {
        c_lines[i].CP[0] = cuda_lines[i].CP[0].x;
        c_lines[i].CP[1] = cuda_lines[i].CP[0].y;
        c_lines[i].CP[2] = cuda_lines[i].CP[1].x;
        c_lines[i].CP[3] = cuda_lines[i].CP[1].y;
        c_lines[i].CP[4] = cuda_lines[i].CP[2].x;
        c_lines[i].CP[5] = cuda_lines[i].CP[2].y;
        c_lines[i].nVertices = cuda_lines[i].nVertices;
        
        // Copy tessellated vertices
        for(int j = 0; j < cuda_lines[i].nVertices && j < MAX_TESS_POINTS; j++) {
            c_lines[i].vertexPos[j*2] = cuda_lines[i].vertexPos[j].x;
            c_lines[i].vertexPos[j*2+1] = cuda_lines[i].vertexPos[j].y;
        }
    }
}

// Host wrapper functions with C linkage for Python
extern "C" {

int tessellate_bezier_curves(BezierLineC* lines, int nLines) {
    // Allocate host memory for CUDA structures
    BezierLine* h_bLines = (BezierLine*)malloc(nLines * sizeof(BezierLine));
    if(!h_bLines) return -1;
    
    // Convert C structures to CUDA structures
    convertCToCuda(lines, h_bLines, nLines);
    
    // Allocate device memory
    BezierLine* d_bLines;
    cudaError_t err = cudaMalloc(&d_bLines, nLines * sizeof(BezierLine));
    if(err != cudaSuccess) {
        free(h_bLines);
        return -2;
    }
    
    // Copy data to device
    err = cudaMemcpy(d_bLines, h_bLines, nLines * sizeof(BezierLine), cudaMemcpyHostToDevice);
    if(err != cudaSuccess) {
        cudaFree(d_bLines);
        free(h_bLines);
        return -3;
    }
    
    // Calculate grid dimensions
    int blockSize = 256;
    int gridSize = min(nLines, 65535); // Max grid size limit
    
    // Launch kernel
    computeBezierLines<<<gridSize, blockSize>>>(d_bLines, nLines);
    
    // Wait for kernel to complete
    err = cudaDeviceSynchronize();
    if(err != cudaSuccess) {
        cudaFree(d_bLines);
        free(h_bLines);
        return -4;
    }
    
    // Copy results back to host
    err = cudaMemcpy(h_bLines, d_bLines, nLines * sizeof(BezierLine), cudaMemcpyDeviceToHost);
    if(err != cudaSuccess) {
        cudaFree(d_bLines);
        free(h_bLines);
        return -5;
    }
    
    // Convert back to C structures
    convertCudaToC(h_bLines, lines, nLines);
    
    // Cleanup
    cudaFree(d_bLines);
    free(h_bLines);
    
    return 0; // Success
}

void print_cuda_error() {
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}

int get_cuda_device_count() {
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if(err != cudaSuccess) return -1;
    return deviceCount;
}

} // extern "C"

// Test main function
int main() {
    printf("Testing Bezier curve tessellation...\n");
    
    // Check CUDA device
    int deviceCount = get_cuda_device_count();
    printf("CUDA devices available: %d\n", deviceCount);
    if(deviceCount == 0) {
        printf("No CUDA devices found!\n");
        return -1;
    }
    
    // Create test data
    int nLines = 3;
    BezierLineC* lines = (BezierLineC*)malloc(nLines * sizeof(BezierLineC));
    
    // Define some interesting Bezier curves
    // Curve 1: Simple arc
    lines[0].CP[0] = 0.0f; lines[0].CP[1] = 0.0f;  // P0
    lines[0].CP[2] = 0.5f; lines[0].CP[3] = 1.0f;  // P1 (control)
    lines[0].CP[4] = 1.0f; lines[0].CP[5] = 0.0f;  // P2
    
    // Curve 2: S-curve
    lines[1].CP[0] = 0.0f; lines[1].CP[1] = 1.0f;
    lines[1].CP[2] = 0.8f; lines[1].CP[3] = 1.2f;
    lines[1].CP[4] = 1.0f; lines[1].CP[5] = 2.0f;
    
    // Curve 3: Sharp turn
    lines[2].CP[0] = 1.0f; lines[2].CP[1] = 2.0f;
    lines[2].CP[2] = 1.5f; lines[2].CP[3] = 1.0f;
    lines[2].CP[4] = 2.0f; lines[2].CP[5] = 2.0f;
    
    // Tessellate the curves
    int result = tessellate_bezier_curves(lines, nLines);
    if(result == 0) {
        printf("Tessellation successful!\n");
        for(int i = 0; i < nLines; i++) {
            printf("Curve %d: %d vertices\n", i, lines[i].nVertices);
        }
    } else {
        printf("Tessellation failed with error code: %d\n", result);
        print_cuda_error();
    }
    
    free(lines);
    return result;
}