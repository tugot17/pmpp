#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include "gpu_radix_sort.h"
// Three-kernel implementation
__global__ void extractBitsKernel(unsigned int* input, unsigned int* bits, unsigned int N, unsigned int iter) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        unsigned int key = input[tid];
        bits[tid] = (key >> iter) & 1;
    }
}

__global__ void scatterKernel(unsigned int* input, unsigned int* output, unsigned int* scannedBits, unsigned int N,
                              unsigned int iter, unsigned int totalOnes) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        unsigned int key = input[tid];
        unsigned int bit = (key >> iter) & 1;
        unsigned int numOnesBefore = scannedBits[tid];
        unsigned int dst = (bit == 0) ? tid - numOnesBefore : N - totalOnes + numOnesBefore;
        output[dst] = key;
    }
}

void gpuRadixSortThreeKernels(unsigned int* d_input, int N) {
    unsigned int *d_output, *d_bits;
    CUDA_CHECK(cudaMalloc((void**)&d_output, N * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc((void**)&d_bits, N * sizeof(unsigned int)));

    const int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (unsigned int iter = 0; iter < NUM_BITS; iter++) {
        extractBitsKernel<<<numBlocks, BLOCK_SIZE>>>(d_input, d_bits, N, iter);
        CUDA_CHECK(cudaDeviceSynchronize());

        unsigned int lastBit;
        CUDA_CHECK(cudaMemcpy(&lastBit, d_bits + (N - 1), sizeof(unsigned int), cudaMemcpyDeviceToHost));

        thrust::device_ptr<unsigned int> d_bits_ptr(d_bits);
        thrust::exclusive_scan(d_bits_ptr, d_bits_ptr + N, d_bits_ptr);
        CUDA_CHECK(cudaDeviceSynchronize());

        unsigned int scanned_last;
        CUDA_CHECK(cudaMemcpy(&scanned_last, d_bits + (N - 1), sizeof(unsigned int), cudaMemcpyDeviceToHost));

        unsigned int totalOnes = scanned_last + lastBit;

        scatterKernel<<<numBlocks, BLOCK_SIZE>>>(d_input, d_output, d_bits, N, iter, totalOnes);
        CUDA_CHECK(cudaDeviceSynchronize());

        unsigned int* temp = d_input;
        d_input = d_output;
        d_output = temp;
    }

    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_bits));
}

// Single kernel implementation
__device__ void hierarchical_kogge_stone_scan(float* X, float* scan_value, int* flags, unsigned int N) {
    extern __shared__ float buffer[];
    __shared__ float previous_sum;
    const unsigned int tid = threadIdx.x;
    const unsigned int bid = blockIdx.x;
    const unsigned int gid = bid * blockDim.x + tid;

    if (gid < N) {
        buffer[tid] = X[gid];
    } else {
        buffer[tid] = 0.0f;
    }
    __syncthreads();

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        float temp = buffer[tid];
        if (tid >= stride) {
            temp += buffer[tid - stride];
        }
        __syncthreads();
        buffer[tid] = temp;
        __syncthreads();
    }

    float exclusive_value = (tid == 0) ? 0.0f : buffer[tid - 1];
    const float local_sum = buffer[blockDim.x - 1];

    if (tid == 0) {
        if (bid > 0) {
            while (atomicAdd(&flags[bid], 0) == 0) {
            }
            previous_sum = scan_value[bid];
            scan_value[bid + 1] = previous_sum + local_sum;
            __threadfence();
            atomicAdd(&flags[bid + 1], 1);
        } else {
            scan_value[1] = local_sum;
            __threadfence();
            atomicAdd(&flags[1], 1);
        }
    }
    __syncthreads();

    if (gid < N) {
        X[gid] = exclusive_value + (bid > 0 ? previous_sum : 0.0f);
    }
}

__global__ void radix_sort_iter(unsigned int* input, unsigned int* output, float* bits_float, float* scan_value,
                                int* flags, unsigned int N, unsigned int iter) {
    const unsigned int tid = threadIdx.x;
    const unsigned int bid = blockIdx.x;
    const unsigned int i = bid * blockDim.x + tid;

    if (i < N) {
        unsigned int key = input[i];
        bits_float[i] = (float)((key >> iter) & 1);
    }
    __syncthreads();

    __shared__ bool isLastBlock;
    if (tid == 0) {
        isLastBlock = (bid == ((N + blockDim.x - 1) / blockDim.x) - 1);
    }
    __syncthreads();

    if (isLastBlock && tid == ((N - 1) % blockDim.x)) {
        bits_float[N] = 0;
        __threadfence();
        atomicAdd(&flags[gridDim.x], 1);
    }

    hierarchical_kogge_stone_scan(bits_float, scan_value, flags, N);

    if (isLastBlock && tid == ((N - 1) % blockDim.x)) {
        bits_float[N] = bits_float[N - 1] + ((input[N - 1] >> iter) & 1);
        __threadfence();
        atomicAdd(&flags[gridDim.x + 1], 1);
    }

    if (tid == 0) {
        while (atomicAdd(&flags[gridDim.x + 1], 0) == 0) {
        }
    }
    __syncthreads();

    if (i < N) {
        unsigned int key = input[i];
        unsigned int bit = (key >> iter) & 1;
        float numOnesBefore = bits_float[i];
        float numOnesTotal = bits_float[N];

        unsigned int dst =
            (bit == 0) ? i - (unsigned int)numOnesBefore : N - (unsigned int)numOnesTotal + (unsigned int)numOnesBefore;

        output[dst] = key;
    }
}

void gpuRadixSortSingleKernel(unsigned int* d_input, int N) {
    assert(N <= MAX_INPUT_SIZE &&
           "Input size above limit leads to potential deadlock due to grid-level synchronization issues.");

    unsigned int* d_output;
    float *d_bits_float, *d_scan_value;
    int* d_flags;
    const int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    CUDA_CHECK(cudaMalloc((void**)&d_output, N * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc((void**)&d_bits_float, (N + 1) * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_scan_value, (numBlocks + 1) * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_flags, (numBlocks + 3) * sizeof(int)));

    for (unsigned int iter = 0; iter < NUM_BITS; iter++) {
        CUDA_CHECK(cudaMemset(d_flags, 0, (numBlocks + 3) * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_scan_value, 0, (numBlocks + 1) * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_bits_float + N, 0, sizeof(float)));

        radix_sort_iter<<<numBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_input, d_output, d_bits_float,
                                                                               d_scan_value, d_flags, N, iter);
        CUDA_CHECK(cudaDeviceSynchronize());

        unsigned int* temp = d_input;
        d_input = d_output;
        d_output = temp;
    }

    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_bits_float));
    CUDA_CHECK(cudaFree(d_scan_value));
    CUDA_CHECK(cudaFree(d_flags));
}

/*
Each block load its section of d_input, computes the bits outstanding and puts that into the shared memory.
Than we perform an exclusive scan on the data in the shared memory (using the Belloch algorithm).
The per thread result is saved into d_localScan - aka we store here how many 1s proceeds the current element.
We also save the each block's total ones into the `d_blockOneCount[blockIdx.x]` - to be used in the later phase.
We also write out the bit value into d_bits global memory array so it can be later consumed by the scatter kernel.
*/
__global__ void localScanKernel(unsigned int* d_input, unsigned int* d_localScan, unsigned int* d_blockOneCount, int N,
                                unsigned int iter) {
    extern __shared__ unsigned int s_bits_scan[];  // shared memory for the bits scan
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    unsigned int bit_val = 0;
    if (gid < N) {
        unsigned int key = d_input[gid];
        bit_val = (key >> iter) & 1;
    } else {
        bit_val = 0;
    }

    s_bits_scan[tid] = bit_val;
    __syncthreads();

    // Up-sweep phase (reduce)
    for (unsigned int offset = 1; offset < blockDim.x; offset *= 2) {
        int index = (tid + 1) * offset * 2 - 1;
        if (index < blockDim.x) {
            s_bits_scan[index] += s_bits_scan[index - offset];
        }
        __syncthreads();
    }

    // save the block's total ones (the last element) and set it to zero for exclusive scan
    if (tid == 0) {
        d_blockOneCount[blockIdx.x] = s_bits_scan[blockDim.x - 1];
        s_bits_scan[blockDim.x - 1] = 0;
    }
    __syncthreads();

    // Down sweep phase
    for (unsigned int offset = blockDim.x / 2; offset >= 1; offset /= 2) {
        int index = (tid + 1) * offset * 2 - 1;
        if (index < blockDim.x) {
            unsigned int t = s_bits_scan[index - offset];
            s_bits_scan[index - offset] = s_bits_scan[index];
            s_bits_scan[index] += t;
        }
        __syncthreads();
        // Prevent underflow of offset
        if (offset == 1) {
            break;
        }
    }

    // Write the result from shared memory to global memory.
    if (gid < N) {
        d_localScan[gid] = s_bits_scan[tid];
    }
}

/*
Each block uses its local scan results (in d_localScan) along with the
pre-computed global offsets for zeros and (d_blockZeroOffsets) and ones
(d_blockOneOffsets) to compute for each element a destination index.
For an element with bit==0 the local index is (tid-localPrefix: by how many ones to move it left)
and for bit==1 it is just localPrefix
The final destination is computed by adding the per-block offset
*/
__global__ void scatterKernelCoalesced(unsigned int* d_input, unsigned int* d_output, unsigned int* d_localScan,
                                       unsigned int* d_blockZeroOffsets, unsigned int* d_blockOneOffsets,
                                       unsigned int totalZeros, int N, unsigned int iter) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    if (gid < N) {
        unsigned int key = d_input[gid];
        unsigned int bit = (key >> iter) & 1;
        unsigned int local_prefix = d_localScan[gid];  // how many ones proceeds the current element
        // we define dest, so that the memory save is done in a coaleased manner
        unsigned int dest;
        if (bit == 0) {
            dest = d_blockZeroOffsets[blockIdx.x] + tid -
                   local_prefix;  // how many zeros in blocks before + local tid - number of ones proceeding it
        } else {
            dest = totalZeros + d_blockOneOffsets[blockIdx.x] +
                   local_prefix;  // how many zeros in total, plus how many ones in blocks before + number of ones
                                  // preceding current element
        }
        d_output[dest] = key;
    }
}

void gpuRadixSortWithMemoryCoalescing(unsigned int* d_input, int N) {
    // Allocate extra arrays on device:
    unsigned int *d_output, *d_bits, *d_localScan, *d_blockOneCount;
    // For per–block offsets (there will be at most MAX_BLOCKS blocks)
    unsigned int *d_blockZeroOffsets, *d_blockOneOffsets;
    CUDA_CHECK(cudaMalloc((void**)&d_output, N * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc((void**)&d_bits, N * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc((void**)&d_localScan, N * sizeof(unsigned int)));
    // One value per block (the grid size is computed below)
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    CUDA_CHECK(cudaMalloc((void**)&d_blockOneCount, numBlocks * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc((void**)&d_blockZeroOffsets, numBlocks * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc((void**)&d_blockOneOffsets, numBlocks * sizeof(unsigned int)));

    // Allocate temporary host arrays for block totals and offsets.
    unsigned int* h_blockOneCount = (unsigned int*)malloc(numBlocks * sizeof(unsigned int));
    unsigned int* h_blockZeroCount = (unsigned int*)malloc(numBlocks * sizeof(unsigned int));
    unsigned int* h_blockZeroOffsets = (unsigned int*)malloc(numBlocks * sizeof(unsigned int));
    unsigned int* h_blockOneOffsets = (unsigned int*)malloc(numBlocks * sizeof(unsigned int));
    if (!h_blockOneCount || !h_blockZeroCount || !h_blockZeroOffsets || !h_blockOneOffsets) {
        fprintf(stderr, "Failed to allocate host arrays for block totals.\n");
        exit(EXIT_FAILURE);
    }

    const int numBits = 32;
    unsigned int totalZeros = 0;

    for (unsigned int iter = 0; iter < numBits; iter++) {
        localScanKernel<<<numBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(unsigned int)>>>(d_input, d_localScan,
                                                                                      d_blockOneCount, N, iter);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy per–block ones count to host and compute block zeros counts.
        CUDA_CHECK(
            cudaMemcpy(h_blockOneCount, d_blockOneCount, numBlocks * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        for (int i = 0; i < numBlocks; i++) {
            // For all blocks except perhaps the last, there are BLOCK_SIZE elements; for the last block, it might be
            // less.
            int blockSize = ((i == numBlocks - 1) && (N % BLOCK_SIZE != 0)) ? (N % BLOCK_SIZE) : BLOCK_SIZE;
            h_blockZeroCount[i] = blockSize - h_blockOneCount[i];
        }
        // Compute exclusive scan (prefix sums) on the block zeros and ones counts.
        h_blockZeroOffsets[0] = 0;
        h_blockOneOffsets[0] = 0;
        for (int i = 1; i < numBlocks; i++) {
            h_blockZeroOffsets[i] = h_blockZeroOffsets[i - 1] + h_blockZeroCount[i - 1];
            h_blockOneOffsets[i] = h_blockOneOffsets[i - 1] + h_blockOneCount[i - 1];
        }
        // Total zeros is the sum of all block zeros.
        totalZeros = h_blockZeroOffsets[numBlocks - 1] + h_blockZeroCount[numBlocks - 1];

        // Copy the computed offsets back to the device.
        CUDA_CHECK(cudaMemcpy(d_blockZeroOffsets, h_blockZeroOffsets, numBlocks * sizeof(unsigned int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(
            cudaMemcpy(d_blockOneOffsets, h_blockOneOffsets, numBlocks * sizeof(unsigned int), cudaMemcpyHostToDevice));

        scatterKernelCoalesced<<<numBlocks, BLOCK_SIZE>>>(d_input, d_output, d_localScan, d_blockZeroOffsets,
                                                          d_blockOneOffsets, totalZeros, N, iter);
        CUDA_CHECK(cudaDeviceSynchronize());

        unsigned int* temp = d_input;
        d_input = d_output;
        d_output = temp;
    }

    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_bits));
    CUDA_CHECK(cudaFree(d_localScan));
    CUDA_CHECK(cudaFree(d_blockOneCount));
    CUDA_CHECK(cudaFree(d_blockZeroOffsets));
    CUDA_CHECK(cudaFree(d_blockOneOffsets));

    free(h_blockOneCount);
    free(h_blockZeroCount);
    free(h_blockZeroOffsets);
    free(h_blockOneOffsets);
}
/*
For each key in d_input we extract the digit. In shared memory we build a histogram (s_hist) for the block.
We also store the keys digit in the latter part of the shared memory (s_digits). Then, each thread computes the local
offset, aka how many keys with the same digit appear before it.
*/
__global__ void localScanKernelMultibitRadix(const unsigned int* d_input, unsigned int* d_localOffsets,
                                             unsigned int* d_blockHist, int N, unsigned int iter, unsigned int r) {
    const unsigned int numBuckets = 1 << r;  // 2^r, e.g. 2^4 = 16
    extern __shared__ unsigned int shared[];

    // we store local histograms + each thread's digit
    unsigned int* s_hist = shared;
    unsigned int* s_digits = (unsigned int*)&s_hist[numBuckets];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    unsigned int digit = 0;
    if (gid < N) {
        unsigned int key = d_input[gid];
        digit = (key >> (iter * r) & (numBuckets - 1));
    }
    s_digits[tid] = digit;

    // init a shared histogram for block
    for (unsigned int i = tid; i < numBuckets; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();

    // use atomic operation to increase bucket value for a digit
    if (gid < N) {
        atomicAdd(&s_hist[digit], 1);
    }
    __syncthreads();

    // write the histograms into the global memory, to be used later
    for (unsigned int i = tid; i < numBuckets; i += blockDim.x) {
        d_blockHist[blockIdx.x * numBuckets + i] = s_hist[i];
    }
    __syncthreads();

    // calculate how many numbers with the same digit for this iteration are before
    unsigned int local_offset = 0;
    for (unsigned j = 0; j < tid; j++) {
        if (s_digits[j] == digit) {
            local_offset++;
        }
    }
    if (gid < N) {
        d_localOffsets[gid] = local_offset;
    }
}

/*
We use this kernel to calculate the global destination index for each element of d_input.
we calculate the destination by bucketPrefix[digit] (how many items in buckets before current bucket)
+ per-block offset for bucket (how many items in the current block in bucket before current element)
+ local_offset (how many same values for bit in the current block before current element)
*/
__global__ void scatterKernelMultibitRadix(const unsigned int* d_input, unsigned int* d_output,
                                           const unsigned int* d_localOffsets, const unsigned int* d_globalOffsets,
                                           int N, unsigned int iter, unsigned int r) {
    const unsigned int numBuckets = 1 << r;
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    if (gid < N) {
        unsigned int key = d_input[gid];
        unsigned int digit = (key >> (iter * r) & (numBuckets - 1));

        unsigned int block_offset = d_globalOffsets[blockIdx.x * numBuckets + digit];
        unsigned int local_offset = d_localOffsets[gid];
        unsigned int dest = block_offset + local_offset;
        // coaleased memory write
        d_output[dest] = key;
    }
}

void gpuRadixSortCoalescedMultibitRadix(unsigned int* d_input, int N, unsigned int r) {
    const unsigned int numBuckets = 1 << r;
    unsigned int numPasses = (32 + r - 1) / r;  // assuming we work with 32-bit keys

    unsigned int* d_output;
    CUDA_CHECK(cudaMalloc((void**)&d_output, N * sizeof(unsigned int)));

    unsigned int* d_localOffsets;
    CUDA_CHECK(cudaMalloc((void**)&d_localOffsets, N * sizeof(unsigned int)));

    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int* d_blockHist;
    CUDA_CHECK(cudaMalloc((void**)&d_blockHist, numBlocks * numBuckets * sizeof(unsigned int)));

    // d_globalOffsets: one per block and per bucket.
    unsigned int* d_globalOffsets;
    CUDA_CHECK(cudaMalloc((void**)&d_globalOffsets, numBlocks * numBuckets * sizeof(unsigned int)));

    // allocate host arrays to store per-block histograms and temporarily the global offsets
    unsigned int* h_blockHist = (unsigned int*)malloc(numBlocks * numBuckets * sizeof(unsigned int));
    unsigned int* h_globalOffsets = (unsigned int*)malloc(numBlocks * numBuckets * sizeof(unsigned int));
    if (!h_blockHist || !h_globalOffsets) {
        fprintf(stderr, "Failed to allocate host arrays for histogram offsets.\n");
        exit(EXIT_FAILURE);
    }

    // allocate temporary host arrays for total counts per bucket and bucket prefix
    unsigned int* total_bucket = (unsigned int*)malloc(numBuckets * sizeof(unsigned int));
    unsigned int* prefix_bucket = (unsigned int*)malloc(numBuckets * sizeof(unsigned int));
    if (!total_bucket || !prefix_bucket) {
        fprintf(stderr, "Failed to allocate temporary host arrays.\n");
        exit(EXIT_FAILURE);
    }

    // shared  memoory size: numBuckets ints for histogram + BLOCK_SIZE ints for storing each thread's digit
    size_t sharedMemSize = numBuckets * sizeof(unsigned int) + BLOCK_SIZE * sizeof(unsigned int);
    for (unsigned int pass = 0; pass < numPasses; pass++) {
        localScanKernelMultibitRadix<<<numBlocks, BLOCK_SIZE, sharedMemSize>>>(d_input, d_localOffsets, d_blockHist, N,
                                                                               pass, r);

        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(
            cudaMemcpy(h_blockHist, d_blockHist, numBlocks * numBuckets * sizeof(unsigned int), cudaMemcpyDeviceToHost))

        // for every of 2^r buckets, iterate over all blocks and calculate the total of items for each bucket
        for (unsigned int bucket = 0; bucket < numBuckets; bucket++) {
            unsigned int sum = 0;
            for (unsigned int block = 0; block < numBlocks; block++) {
                sum += h_blockHist[block * numBuckets + bucket];
            }
            total_bucket[bucket] = sum;
        }

        // we want to know where each bucket starts in the global output array
        // we calculate an exclusive scan on all buckets; we know that first bucket starts at 0
        prefix_bucket[0] = 0;
        for (unsigned int bucket = 1; bucket < numBuckets; bucket++) {
            prefix_bucket[bucket] = prefix_bucket[bucket - 1] + total_bucket[bucket - 1];
        }

        // now for every local bucket we want to know how much should we offset it based on the global buckets order
        for (unsigned int bucket = 0; bucket < numBuckets; bucket++) {
            unsigned int sum = 0;
            for (unsigned int block = 0; block < numBlocks; block++) {
                h_globalOffsets[block * numBuckets + bucket] = prefix_bucket[bucket] + sum;
                sum += h_blockHist[block * numBuckets + bucket];
            }
        }

        CUDA_CHECK(cudaMemcpy(d_globalOffsets, h_globalOffsets, numBlocks * numBuckets * sizeof(unsigned int),
                              cudaMemcpyHostToDevice));

        // launch scatter kernel to reposition keys in the coaleased manner
        scatterKernelMultibitRadix<<<numBlocks, BLOCK_SIZE>>>(d_input, d_output, d_localOffsets, d_globalOffsets, N,
                                                              pass, r);

        CUDA_CHECK(cudaDeviceSynchronize());

        // output becomes input in the next itertion
        unsigned int* temp = d_input;
        d_input = d_output;
        d_output = temp;
    }

    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_localOffsets));
    CUDA_CHECK(cudaFree(d_blockHist));
    CUDA_CHECK(cudaFree(d_globalOffsets));

    free(h_blockHist);
    free(h_globalOffsets);
    free(total_bucket);
    free(prefix_bucket);
}

/*
For each key in d_input we extract the digit. In shared memory we build a histogram (s_hist) for the block.
We also store the keys' digit in the latter part of the shared memory (s_digits), where now s_digits
has BLOCK_SIZE*COARSE_FACTOR entries. Then, each thread computes, for each of its keys, the local offset,
i.e. how many keys with the same digit appear before it in block order.
*/
__global__ void localScanKernelMultibitRadixCoarsening(const unsigned int* d_input, unsigned int* d_localOffsets,
                                                       unsigned int* d_blockHist, int N, unsigned int iter,
                                                       unsigned int r) {
    const unsigned int numBuckets = 1 << r;  // 2^r, e.g. 2^4 = 16

    // Shared memory layout
    // [0, numBuckets - 1]: histogram
    // [ numBuckets, numBuckets + (blockDim.x*COARSE_FACTOR)-1 ]: digits for the block (s_digits)
    extern __shared__ unsigned int shared[];

    // we store local histograms + each thread's digit
    unsigned int* s_hist = shared;
    unsigned int* s_digits = (unsigned int*)&s_hist[numBuckets];

    int tid = threadIdx.x;
    // Each block now processes blockThreadCount * COARSE_FACTOR keys.
    int baseIdx = blockIdx.x * blockDim.x * COARSE_FACTOR + tid * COARSE_FACTOR;

    for (unsigned int i = 0; i < COARSE_FACTOR; i++) {
        int idx = baseIdx + i;
        unsigned int digit = 0;
        if (idx < N) {
            unsigned int key = d_input[idx];
            digit = (key >> (iter * r) & (numBuckets - 1));
        }
        s_digits[tid * COARSE_FACTOR + i] = digit;
    }
    __syncthreads();

    // init a shared histogram for block
    for (unsigned int i = tid; i < numBuckets; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Each thread loops over its keys and atomically updates the block histogram.
    for (unsigned int i = 0; i < COARSE_FACTOR; i++) {
        int idx = baseIdx + i;
        if (idx < N) {
            unsigned int digit = s_digits[tid * COARSE_FACTOR + i];
            atomicAdd(&s_hist[digit], 1);
        }
    }
    __syncthreads();

    // write the histograms into the global memory, to be used later
    for (unsigned int i = tid; i < numBuckets; i += blockDim.x) {
        d_blockHist[blockIdx.x * numBuckets + i] = s_hist[i];
    }
    __syncthreads();

    // For each key processed by this thread, compute the local offset:
    // count, among all keys earlier in the block (in shared memory order), how many have the same digit.
    for (int i = 0; i < COARSE_FACTOR; i++) {
        int globalPos = tid * COARSE_FACTOR + i;  // position in the block’s order

        unsigned int myDigit = s_digits[globalPos];
        unsigned int local_offset = 0;
        for (int j = 0; j < globalPos; j++) {
            if (s_digits[j] == myDigit) {
                local_offset++;
            }
        }

        int outIdx = baseIdx + i;
        if (outIdx < N) {
            d_localOffsets[outIdx] = local_offset;
        }
    }
}

/*
Each thread processes COARSE_FACTOR keys. For each key, we use its digit to get
the corresponding global bucket offset (from d_globalOffsets)
and then add its computed local offset to determine the destination in d_output.
*/
__global__ void scatterKernelMultibitRadixThreadCoarsening(const unsigned int* d_input, unsigned int* d_output,
                                                           const unsigned int* d_localOffsets,
                                                           const unsigned int* d_globalOffsets, int N,
                                                           unsigned int iter, unsigned int r) {
    const unsigned int numBuckets = 1 << r;
    int tid = threadIdx.x;
    int blockThreadCount = blockDim.x;
    int baseIdx = blockIdx.x * blockThreadCount * COARSE_FACTOR + tid * COARSE_FACTOR;

    for (int i = 0; i < COARSE_FACTOR; i++) {
        int idx = baseIdx + i;
        if (idx < N) {
            unsigned int key = d_input[idx];
            unsigned int digit = (key >> (iter * r)) & (numBuckets - 1);

            unsigned int block_offset = d_globalOffsets[blockIdx.x * numBuckets + digit];
            unsigned int local_offset = d_localOffsets[idx];

            unsigned int dest = block_offset + local_offset;
            d_output[dest] = key;
        }
    }
}

void gpuRadixSortCoalescedMultibitRadixThreadCoarsening(unsigned int* d_input, int N, unsigned int r) {
    const unsigned int numBuckets = 1 << r;
    unsigned int numPasses = (32 + r - 1) / r;  // for 32-bit keys

    unsigned int* d_output;
    CUDA_CHECK(cudaMalloc((void**)&d_output, N * sizeof(unsigned int)));

    unsigned int* d_localOffsets;
    CUDA_CHECK(cudaMalloc((void**)&d_localOffsets, N * sizeof(unsigned int)));

    // Note: each block now processes BLOCK_SIZE*COARSE_FACTOR keys.
    int numBlocks = (N + (BLOCK_SIZE * COARSE_FACTOR) - 1) / (BLOCK_SIZE * COARSE_FACTOR);
    unsigned int* d_blockHist;
    CUDA_CHECK(cudaMalloc((void**)&d_blockHist, numBlocks * numBuckets * sizeof(unsigned int)));

    // d_globalOffsets: one per block and per bucket.
    unsigned int* d_globalOffsets;
    CUDA_CHECK(cudaMalloc((void**)&d_globalOffsets, numBlocks * numBuckets * sizeof(unsigned int)));

    // Allocate host arrays to store per-block histograms and temporary global offsets.
    unsigned int* h_blockHist = (unsigned int*)malloc(numBlocks * numBuckets * sizeof(unsigned int));
    unsigned int* h_globalOffsets = (unsigned int*)malloc(numBlocks * numBuckets * sizeof(unsigned int));
    if (!h_blockHist || !h_globalOffsets) {
        fprintf(stderr, "Failed to allocate host arrays for histogram offsets.\n");
        exit(EXIT_FAILURE);
    }

    // Temporary host arrays for total counts per bucket and bucket prefix.
    unsigned int* total_bucket = (unsigned int*)malloc(numBuckets * sizeof(unsigned int));
    unsigned int* prefix_bucket = (unsigned int*)malloc(numBuckets * sizeof(unsigned int));
    if (!total_bucket || !prefix_bucket) {
        fprintf(stderr, "Failed to allocate temporary host arrays.\n");
        exit(EXIT_FAILURE);
    }

    // Shared memory size: histogram + (BLOCK_SIZE*COARSE_FACTOR) digits.
    size_t sharedMemSize = numBuckets * sizeof(unsigned int) + (BLOCK_SIZE * COARSE_FACTOR) * sizeof(unsigned int);

    for (unsigned int pass = 0; pass < numPasses; pass++) {
        localScanKernelMultibitRadixCoarsening<<<numBlocks, BLOCK_SIZE, sharedMemSize>>>(d_input, d_localOffsets,
                                                                                         d_blockHist, N, pass, r);

        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_blockHist, d_blockHist, numBlocks * numBuckets * sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));

        // For every bucket, iterate over all blocks to compute total count.
        for (unsigned int bucket = 0; bucket < numBuckets; bucket++) {
            unsigned int sum = 0;
            for (unsigned int block = 0; block < numBlocks; block++) {
                sum += h_blockHist[block * numBuckets + bucket];
            }
            total_bucket[bucket] = sum;
        }

        // Compute exclusive scan over buckets to determine global starting offsets.
        prefix_bucket[0] = 0;
        for (unsigned int bucket = 1; bucket < numBuckets; bucket++) {
            prefix_bucket[bucket] = prefix_bucket[bucket - 1] + total_bucket[bucket - 1];
        }

        // For every block and bucket, compute the global offset.
        for (unsigned int bucket = 0; bucket < numBuckets; bucket++) {
            unsigned int sum = 0;
            for (unsigned int block = 0; block < numBlocks; block++) {
                h_globalOffsets[block * numBuckets + bucket] = prefix_bucket[bucket] + sum;
                sum += h_blockHist[block * numBuckets + bucket];
            }
        }

        CUDA_CHECK(cudaMemcpy(d_globalOffsets, h_globalOffsets, numBlocks * numBuckets * sizeof(unsigned int),
                              cudaMemcpyHostToDevice));

        // Launch scatter kernel.
        scatterKernelMultibitRadixThreadCoarsening<<<numBlocks, BLOCK_SIZE>>>(d_input, d_output, d_localOffsets,
                                                                              d_globalOffsets, N, pass, r);

        CUDA_CHECK(cudaDeviceSynchronize());

        // Prepare for next pass: swap input and output.
        unsigned int* temp = d_input;
        d_input = d_output;
        d_output = temp;
    }

    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_localOffsets));
    CUDA_CHECK(cudaFree(d_blockHist));
    CUDA_CHECK(cudaFree(d_globalOffsets));

    free(h_blockHist);
    free(h_globalOffsets);
    free(total_bucket);
    free(prefix_bucket);
}
