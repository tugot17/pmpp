// nvcc merge.cu -o merge

#include <stdio.h>
#include <stdlib.h>

#define cdiv(x, y) (((x) + (y)-1) / (y))
#define BLOCK_DIM 4

#define CUDA_CHECK(call)                                                                                 \
    do {                                                                                                 \
        cudaError_t error = call;                                                                        \
        if (error != cudaSuccess) {                                                                      \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(EXIT_FAILURE);                                                                          \
        }                                                                                                \
    } while (0)


float* createSortedArray(int length, float start, float step){
    float* array = (float*)malloc(length * sizeof(float));
    for(unsigned int i =0; i < length; i++){
        array[i] = start + i * step;
    }
    return array;
}

void printArray(float* array, int length) {
    for (int i = 0; i < length; i++) {
        printf("%.1f ", array[i]);
    }
    printf("\n");
}

__host__ __device__ int co_rank(int k, float* A, int m, float* B, int n){
    int i = min(k, m);
    int j = k-i;

    int i_low = max(0, k-n);
    int j_low = max(0, k-m);
    int delta;

    bool active = true;
    while (active)
    {   
        //i too big
        if(i >0 && j < n && A[i-1] > B[j]){
            delta = cdiv(i - i_low, 2);
            j_low = j;
            
            i -= delta;
            j += delta;

        }
        //i too small
        else if (j > 0 && i < m && B[j-1] >= A[i]){
            delta = cdiv(j - j_low, 2);
            i_low = i;
            i += delta;
            j -= delta;
        }

        //the condition A[i-j] <= B[j] and A[i] > B[j-1] satisfied
        else{
            active = false;
        }
    }

    return i;
}

__host__ __device__ void merge_sequential(float* A, int m, float* B, int n, float *C){
    int i = 0; //index into C
    int j = 0; //index into C
    int k = 0; //index into C

    //triage between values of A and B
    while (i<m && j<n)
    {
        if (A[i] <= B[j]){
            C[k++] = A[i++];
        }
        else{
            C[k++] = B[j++];
        }
    }
    
    // Done with A handle remaining B
    while (j < n)
    {
        C[k++] = B[j++];
    }

    // Done with A handle remaining A
    while (i < m)
    {
        C[k++] = A[i++];
    }
}

__global__ void merge_basic_kernel(float *A, int m, float* B, int n, float *C){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    //how many elements in the resulting array C to be processed by a single thread
    int elementsPerThread = cdiv((m+n), blockDim.x * gridDim.x);
    
    //output start and end indices
    int k_curr = tid * elementsPerThread;
    int k_next = min((tid+1) * elementsPerThread, m+n);

    //corank for begenning of a subarray processed by the thread
    int i_curr = co_rank(k_curr, A, m, B, n);
    int j_curr = k_curr - i_curr;

    //corank for end of a subarray processed by the thread
    int i_next = co_rank(k_next, A, m, B, n);
    int j_next = k_next - i_next;

    //execute the sequential merge on the two subarrays
    merge_sequential(&A[i_curr], i_next-i_curr, &B[j_curr], j_next-j_curr, &C[k_curr]);
}

void simple_merge_parallel(float* A, int m, float* B, int n, float *C){
    float* d_A;
    float* d_B;
    float* d_C;

    dim3 dimBlock(BLOCK_DIM);  // for now we stick to a single section executed within a single block
    dim3 dimGrid(1);

    CUDA_CHECK(cudaMalloc((void**)&d_A, m * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_C, (m+n) * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, A, m * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, n * sizeof(float), cudaMemcpyHostToDevice));


    merge_basic_kernel<<<dimGrid, dimBlock>>>(d_A, m, d_B, n, d_C);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(C, d_C, (m+n) * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

int main()
{
    const int m = 5;
    const int n = 5;
    const int k = m + n;
    
    // float* array1 = createSortedArray(m, 1.0f, 0.3f);
    float A[] = {1.0f, 1.3f, 1.6f, 1.9f, 2.2f};
    printf("Array 1: ");
    printArray(A, m);

    // float* array2 = createSortedArray(n, 1.5f, 0.4f);
    float B[] = {1.5f, 1.9f, 2.3f, 2.7f, 3.1f};
    printf("Array 2: ");
    printArray(B, n);

    float C[k];
    float C_2[k];
    
    merge_sequential(A, m, B, n, C);
    printf("Array C sequential: ");
    printArray(C, k);

    simple_merge_parallel(A, m, B, n, C_2);
    printf("Array C parallel: ");
    printArray(C_2, k);

    // printf("\nCo-ranks in Array C:\n");
    // for (int i = 0; i < k; i++) {
    //     int current = C[i];
    //     int rank = co_rank(i + 1, A, m, B, n);
    //     printf("Element: %d, Co-rank: %d\n", i, rank);
    // }

    return 0;

    // free(A);
    // free(B);
}
