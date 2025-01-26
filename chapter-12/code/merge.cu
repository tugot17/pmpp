// nvcc merge.cu -o merge

#include <stdio.h>
#include <stdlib.h>

#define cdiv(x, y) (((x) + (y)-1) / (y))
#define TILE_SIZE 128

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

    int block_dim = 256;
    dim3 dimBlock(block_dim);  // for now we stick to a single section executed within a single block
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


__global__ void merge_tiled_kernel(float *A, int m, float* B, int n, float *C){
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Block 0 starting: m=%d, n=%d\n", m, n);
    }


    extern __shared__ float shareAB[];
    
    float *A_S = &shareAB[0]; //1st part for A_S
    float *B_S = &shareAB[TILE_SIZE]; //2nd part for B_S

    int C_curr = blockIdx.x * cdiv((m+n), gridDim.x); //here the C subarray starts
    int C_next = min((blockIdx.x+1) * cdiv((m+n), gridDim.x), (m+n));//here the C subarray ends

    //use thread 0 to calculate the co-rank for first and last element of the subarray C
    //make it block level visible so all of the thrads have access to this
    if (threadIdx.x == 0){
        A_S[0] = co_rank(C_curr, A, m, B, n);
        A_S[1] = co_rank(C_next, A, m, B, n);
    }
    __syncthreads();//this ensures all of the threads have access to these values

    int A_curr = A_S[0];
    int A_next = A_S[1];
    int B_curr = C_curr - A_curr;
    int B_next = C_next - A_next;
    __syncthreads();
    
    int counter = 0; 
    int C_length = C_next - C_curr;
    int A_length = A_next - A_curr;
    int B_length = B_next - B_curr;
    int total_iteration = cdiv(C_length, TILE_SIZE);
    int C_completed = 0;
    int A_consumed = 0;
    int B_consumed = 0;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Block 0: A_curr=%d, A_next=%d, B_curr=%d, B_next=%d\n", 
               A_curr, A_next, B_curr, B_next);
        printf("C_length=%d, A_length=%d, B_length=%d\n", 
               C_length, A_length, B_length);
    }


    while (counter < total_iteration)
    {
        //levergae all thereads in the block to load the data from global memory in a coalesced manner
        for (unsigned i = 0; i < TILE_SIZE; i += blockDim.x){
            if (i + threadIdx.x < A_length - A_consumed){
                A_S[i + threadIdx.x] = A[A_curr + A_consumed + i + threadIdx.x];
            }

            if (i + threadIdx.x < B_length - B_consumed){
                B_S[i + threadIdx.x] = B[B_curr + B_consumed + i + threadIdx.x];
            }
        }
        __syncthreads();

        int c_curr = threadIdx.x * cdiv(TILE_SIZE, blockDim.x);
        int c_next = min((threadIdx.x + 1) * cdiv(TILE_SIZE, blockDim.x), TILE_SIZE);

        c_curr = min(c_curr, C_length - C_completed);
        c_next = min(c_next, C_length - C_completed);

        //find co-rank for c_curr and c_next
        int a_curr = co_rank(c_curr, A_S, min(TILE_SIZE, A_length-A_consumed), B_S, min(TILE_SIZE, B_length-B_consumed));
        int b_curr = c_curr - a_curr;
        
        int a_next = co_rank(c_next, A_S, min(TILE_SIZE, A_length-A_consumed), B_S, min(TILE_SIZE, B_length-B_consumed));
        int b_next = c_next - a_next;

        if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("Thread 0: c_curr=%d, c_next=%d, a_curr=%d, a_next=%d, b_curr=%d, b_next=%d\n",
                c_curr, c_next, a_curr, a_next, b_curr, b_next);
            printf("Writing to C at index: %d\n", C_curr + C_completed + c_curr);
        }

        //every thread calls the sequential merge function on its subarrays
        merge_sequential(A_S+a_curr, a_next-a_curr, B_S+b_curr, b_next-b_curr, C+C_curr+C_completed+c_curr);

        if (counter == 0 && threadIdx.x == 0 && blockIdx.x == 0) {
            printf("First tile: A_consumed=%d, B_consumed=%d, C_completed=%d\n",
                A_consumed, B_consumed, C_completed);
            // Print first few elements of shared memory
            printf("A_S[0-4]: %.1f %.1f %.1f %.1f %.1f\n", 
                A_S[0], A_S[1], A_S[2], A_S[3], A_S[4]);
            printf("B_S[0-4]: %.1f %.1f %.1f %.1f %.1f\n", 
                B_S[0], B_S[1], B_S[2], B_S[3], B_S[4]);
        }

        counter++; 
        C_completed += TILE_SIZE;
        A_consumed += co_rank(TILE_SIZE, A_S, TILE_SIZE, B_S, TILE_SIZE);
        B_consumed = C_completed - A_consumed;
        __syncthreads();
    }
}

void merge_parallel_with_tiling(float* A, int m, float* B, int n, float *C){
    float* d_A;
    float* d_B;
    float* d_C;

    CUDA_CHECK(cudaMalloc((void**)&d_A, m * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_C, (m+n) * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, A, m * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, n * sizeof(float), cudaMemcpyHostToDevice));

    int threadsPerBlock = 256; // Standard warp-aligned value
    int numBlocks = (m + n + threadsPerBlock - 1) / threadsPerBlock; // Ceil division
    numBlocks = min(numBlocks, 65535); // Maximum grid size limit

    dim3 dimBlock(threadsPerBlock);
    dim3 dimGrid(numBlocks);

    merge_tiled_kernel<<<dimGrid, dimBlock, 2 * TILE_SIZE * sizeof(float)>>>(d_A, m, d_B, n, d_C);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(C, d_C, (m+n) * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

bool allclose(float* a, float* b, int N, float rtol = 1e-5, float atol = 1e-8) {
    for (int i = 0; i < N; i++) {
        float allowed_error = atol + rtol * fabs(b[i]);
        if (fabs(a[i] - b[i]) > allowed_error) {
            printf("Arrays differ at index %d: %f != %f (allowed error: %f)\n", i, a[i], b[i], allowed_error);
            printf("Values around error point:\n");
            int start = (i > 5) ? i - 5 : 0;
            int end = (i + 5 < N) ? i + 5 : N - 1;
            printf("Index\tSequential\tParallel\n");
            for (int j = start; j <= end; j++) {
                printf("%d\t%.1f\t\t%.1f\n", j, a[j], b[j]);
            }
            return false;
        }
    }
    return true;
}

int main()
{
    const int m = 37;
    const int n = 38;
    const int k = m + n;
    
    float* A = createSortedArray(m, 1.0f, 0.3f);
    // float A[] = {1.0f, 1.3f, 1.6f, 1.9f, 2.2f};
    // printf("Array 1: ");
    // printArray(A, m);

    float* B = createSortedArray(n, 1.5f, 0.4f);
    // float B[] = {1.5f, 1.9f, 2.3f, 2.7f, 3.1f};
    // printf("Array 2: ");
    // printArray(B, n);

    float C[k];
    float C_2[k];
    
    merge_sequential(A, m, B, n, C);
    // printf("Array C sequential: ");
    // printArray(C, k);

    merge_parallel_with_tiling(A, m, B, n, C_2);
    // printf("Array C parallel: ");
    // printArray(C_2, k);

    printf("\nComparing results:\n");
    bool equal = allclose(C, C_2, k);
    printf("Arrays are %s\n", equal ? "equal" : "different");

    // printf("\nCo-ranks in Array C:\n");
    // for (int i = 0; i < k; i++) {
    //     int current = C[i];
    //     int rank = co_rank(i + 1, A, m, B, n);
    //     printf("Element: %d, Co-rank: %d\n", i, rank);
    // }

    return 0;

    free(A);
    free(B);
}
