#include <stdio.h>
#include <cuda.h>

#define MAX_TESS_POINTS 32

// A structure containing all parameters needed to tessellate a Bezier line
struct BezierLine {
   float2 CP[3];                    //Control points for the line
   float2 vertexPos[MAX_TESS_POINTS]; //Vertex position array to tessellate into
   int nVertices;                   //Number of tessellated vertices
};

__global__ void computeBezierLines(BezierLine *bLines, int nLines) {
   int bidx = blockIdx.x;
   if(bidx < nLines){
       //Compute the curvature of the line
       float curvature = computeCurvature(bLines);

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
                   position = position + B3u[i] * bLines[bidx].CP[i];
               }
               //Assign value of vertex position to the correct array element
               bLines[bidx].vertexPos[idx] = position;
           }
       }
   }
}

struct BezierLine {
   float2 CP[3];        //Control points for the line
   float2 *vertexPos;   //Vertex position array to tessellate into
   int nVertices;       //Number of tessellated vertices
};

__global__ void computeBezierLines_parent(BezierLine *bLines, int nLines) {
   //Compute a unique index for each Bezier line
   int lidx = threadIdx.x + blockDim.x*blockIdx.x;
   if(lidx < nLines){
       //Compute the curvature of the line
       float curvature = computeCurvature(bLines);
       
       //From the curvature, compute the number of tessellation points
       bLines[lidx].nVertices = min(max((int)(curvature*16.0f),4),MAX_TESS_POINTS);
       cudaMalloc((void**)&bLines[lidx].vertexPos,
                  bLines[lidx].nVertices*sizeof(float2));
       
       //Call the child kernel to compute the tessellated points for each line
       computeBezierLine_child<<<(bLines[lidx].nVertices/32.0f), 32>>>
           (lidx, bLines, bLines[lidx].nVertices);
   }
}

__global__ void computeBezierLine_child(int lidx, BezierLine* bLines,
                                       int nTessPoints) {
   int idx = threadIdx.x + blockDim.x*blockIdx.x; //Compute idx unique to this vertex
   if(idx < nTessPoints){
       float u = (float)idx/(float)(nTessPoints-1);  //Compute u from idx
       float omu = 1.0f - u;  //Pre-compute one minus u
       float B3u[3];          //Compute quadratic Bezier coefficients
       B3u[0] = omu*omu;
       B3u[1] = 2.0f*u*omu;
       B3u[2] = u*u;
       float2 position = {0,0};  //Set position to zero
       for(int i = 0; i < 3; i++) {
           //Add the contribution of the i'th control point to position
           position = position + B3u[i] * bLines[lidx].CP[i];
       }
       //Assign the value of the vertex position to the correct array element
       bLines[lidx].vertexPos[idx] = position;
   }
}

__global__ void freeVertexMem(BezierLine *bLines, int nLines) {
   //Compute a unique index for each Bezier line
   int lidx = threadIdx.x + blockDim.x*blockIdx.x;
   if(lidx < nLines)
       cudaFree(bLines[lidx].vertexPos);  //Free the vertex memory for this line
}