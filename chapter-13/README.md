# Chapter 13

## Code

We implement all of the kernels mentioned in chapter 13. In particural:

- vanilla parallel radix sort with three seperate lanuches of the grid
- vanilla parallel radix sort with a single kernel - the channelge remains entrie grid sync that is required. Unfortunatelly after corssing somwwhere above 100k elements it get's deadlocked. The challenge is that the first block needs to wait for the last one cause we need the total number of zeros in the entire input. 
- radix sort with coaleased writing to the memory
- radix sort with coaleased writing to the memory and with thread coeaeasing
- parallel merge sort



For simple experimentation with different implementations of sort I recommend you use [code/sort.cu](sort.cu) where you can easily try new implementation and it will be compared to the `quicksort` from the standard liberary. 


## Exercises

### Exercise 1
**Extend the kernel in Fig. 13.4 by using shared memory to improve memory coalescing.**

### Exercise 2

**Extend the kernel in Fig. 13.4 to work for a multibit radix.**

### Exercise 3

**Extend the kernel in Fig. 13.4 by applying thread coarsening to improve memory coalescing.**

### Exercise 4

**Implement parallel merge sort using the parallel merge implementation from Chapter 12, Merge.**

