# Chapter 13

## Code

We implement all of the kernels mentioned in chapter 13. In particular:

- Vanilla parallel radix sort with three separate kernels
- Radix sort with coaleased writing to the memory
- Radix sort with coalesced writing to the memory and multibit radix
- Radix sort with coalesced writing to the memory and multibit radix and with thread coalescing
- Parallel merge sort

We compare all of these to the ``quicksort`` executed on the GPU and report the speedup. While it is not fully fair, we want to give the reader the sense of "How fast GPU sort is compared to the implementation CPU.

To execute the benchmark, simply run:

```bash
make
```

For `N = 10,000,000;` you should be seeing something like:
```
=== Performance Summary ===
+---------------------------------------------------------------------+-------------+----------+
|                            Sort Method                              | Time (ms)   | Speedup  |
+---------------------------------------------------------------------+-------------+----------+
| Naive Parallel Radix Sort                                           |      14.099 |  149.70x |
| Memory-coalesced GPU sort                                           |      18.055 |  116.90x |
| Memory-coalesced GPU sort (multiradix)                              |      20.671 |  102.11x |
| Memory-coalesced GPU sort (multiradix and thread coarsening)        |      39.275 |   53.74x |
| GPU merge sort                                                      |     135.828 |   15.54x |
+---------------------------------------------------------------------+-------------+----------+
```

There are probably some further optimizations to be implemented here to get every last squeeze of performance, but we consider this a form of an exercise and don't delve much deeper.

We also implement Vanilla parallel radix sort with a single kernel. The main challenge here is the synchronization across the entire grid. The hard part is that the first block needs to wait for the last one because we need the total number of zeros in the entire input. Unfortunately, after crossing somewhere above 100k elements, it gets deadlocked. Hence we leave it out of the benchmark.
For simple experimentation with different implementations of sort, I recommend you use [code/sort.cu](sort.cu) where you can easily try new parallel sort implementations. It will be compared to the `quicksort` standard library. This is where you can try the single kernel version of parallel radix sort mentioned above.

```bash
nvcc sort.cu -o sort
```

## Exercises

### Exercise 1
**Extend the kernel in Fig. 13.4 by using shared memory to improve memory coalescing.**

The kernel is quite extensive; hence, it is best if you just look directly at the implementation to be found in [gpu_radix_sort.cu](https://github.com/tugot17/pmpp/chapter-13/code/gpu_radix_sort.cu#L298)

### Exercise 2

**Extend the kernel in Fig. 13.4 to work for a multibit radix.**

As above, the kernel is quite extensive; hence, it is best if you just look directly at the implementation to be found in [gpu_radix_sort.cu](https://github.com/tugot17/pmpp/chapter-13/code/gpu_radix_sort.cu#L459)

### Exercise 3

**Extend the kernel in Fig. 13.4 by applying thread coarsening to improve memory coalescing.**

As above, the kernel is quite extensive; hence, it is best if you just look directly at the implementation to be found in [gpu_radix_sort.cu](https://github.com/tugot17/pmpp/chapter-13/code/gpu_radix_sort.cu#670).

### Exercise 4

**Implement parallel merge sort using the parallel merge implementation from Chapter 12, Merge.**

As above, the kernel is quite extensive; hence, it is best if you just look directly at the implementation to be found in [gpu_merge_sort.cu](https://github.com/tugot17/pmpp/chapter-13/code/gpu_merge_sort.cu#L117).

