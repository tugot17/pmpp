// nvcc merge.cu -o merge

#include <stdio.h>
#include <stdlib.h>

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

void merge_sequential(float* A, int m, float* B, int n, float *C){
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
    while (i < n)
    {
        C[k++] = A[i++];
    }
}

// int co_rank(int k, int* A, int m, int* B, int n){

// }

int main()
{
    const int n1 = 5;
    const int n2 = 5;
    const int n3 = n1 + n2;
    
    // float* array1 = createSortedArray(n1, 1.0f, 0.3f);
    float A[] = {1.0f, 1.3f, 1.6f, 1.9f, 2.2f};
    printf("Array 1: ");
    printArray(A, n1);

    // float* array2 = createSortedArray(n2, 1.5f, 0.4f);
    float B[] = {1.5f, 1.9f, 2.3f, 2.7f, 3.1f};
    printf("Array 2: ");
    printArray(B, n2);

    float C[n3];
    merge_sequential(A, n1, B, n2, C);

    printf("Array C: ");
    printArray(C, n3);

    // free(A);
    // free(B);
}
