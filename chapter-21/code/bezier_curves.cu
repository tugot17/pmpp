#include <stdio.h>
#include <cuda.h>
#define MAX_TESS_POINTS 32

// A structure containing all parameters needed to tessellate a Bezier line
struct BezierLine {
   float2 CP[3];                    //Control points for the line
   float2 vertexPos[MAX_TESS_POINTS]; //Vertex position array to tessellate into
   int nVertices;                   //Number of tessellated vertices
};

__device__ float computeCurvature(float2 *cp) {
    float dx = cp[2].x - cp[0].x;
    float dy = cp[2].y - cp[0].y; 
    float line_length = sqrtf(dx*dx + dy*dy);
    if (line_length < 0.001f) return 0.0f;
    
    float cross = fabsf((cp[1].x - cp[0].x)*dy - (cp[1].y - cp[0].y)*dx);
    return cross / line_length;
}

__global__ void computeBezierLines(BezierLine *bLines, int nLines) {
   int bidx = blockIdx.x;
   if(bidx < nLines){
       //Compute the curvature of the line
       float curvature = computeCurvature(bLines[bidx].CP);
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
               float2 position = {0,0};  //Set position to zero
               for(int i = 0; i < 3; i++){
                   //Add the contribution of the i'th control point to position
                   position.x += B3u[i] * bLines[bidx].CP[i].x;
                   position.y += B3u[i] * bLines[bidx].CP[i].y;
               }
               //Assign value of vertex position to the correct array element
               bLines[bidx].vertexPos[idx] = position;
           }
       }
   }
}

// Host wrapper function
extern "C" {
int tessellate_bezier_lines(BezierLine* lines, int num_lines) {
    BezierLine* d_lines;
    size_t size = num_lines * sizeof(BezierLine);
    
    // Allocate and copy to device
    if (cudaMalloc(&d_lines, size) != cudaSuccess) return -1;
    if (cudaMemcpy(d_lines, lines, size, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d_lines);
        return -2;
    }
    
    // Launch kernel: one block per line, multiple threads per block
    int threads = 32; // Adjust based on your needs
    computeBezierLines<<<num_lines, threads>>>(d_lines, num_lines);
    
    // Wait for kernel to complete
    cudaDeviceSynchronize();
    
    // Copy results back to host
    if (cudaMemcpy(lines, d_lines, size, cudaMemcpyDeviceToHost) != cudaSuccess) {
        cudaFree(d_lines);
        return -3;
    }
    
    cudaFree(d_lines);
    return 0;
}
}