// nvcc merge.cu -o merge

#include <stdio.h>
#include <stdlib.h>

#define cdiv(x, y) (((x) + (y)-1) / (y))

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

int co_rank(int k, float* A, int m, float* B, int n){
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
    merge_sequential(A, m, B, n, C);

    printf("Array C: ");
    printArray(C, k);

    printf("\nCo-ranks in Array C:\n");
    for (int i = 0; i < k; i++) {
        int current = C[i];
        int rank = co_rank(i + 1, A, m, B, n);
        printf("Element: %d, Co-rank: %d\n", i, rank);
    }

    return 0;

    // free(A);
    // free(B);
}
