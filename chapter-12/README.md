# Chapter 12

## Code

## Exercises

### Exercise 1

**Assume that we need to merge two lists A=(1, 7, 8, 9, 10) and B=(7, 10, 10, 12). What are the co-rank values for C[8]?**

### Exercise 2

**Complete the calculation of co-rank functions for thread 2 in Fig. 12.6.**

### Exercise 3
**For the for-loops that load A and B tiles in Fig. 12.12, add a call to the co- rank function so that we can load only the A and B elements that will be consumed in the current generation of the while-loop.**

### Exercise 4

**Consider a parallel merge of two arrays of size 1,030,400 and 608,000. Assume that each thread merges eight elements and that a thread block size of 1024 is used.**
**a. In the basic merge kernel in Fig. 12.9, how many threads perform a binary search on the data in the global memory?**
**b. In the tiled merge kernel in Figs. 12.11 - 12.13, how many threads perform a binary search on the data in the global memory?**
**c. In the tiled merge kernel in Figs. 12.11 - 12.13, how many threads perform a binary search on the data in the shared memory?**