# Programming Massively Parallel Processors book, solutions


The following repo documents my journey to learn CUDA programming. You can find code for examples from the book and detailed answers to the exercises. LFG.

## Where to find solutions by chapter

- [Chapter 2](chapter-2/README.md)
- [Chapter 3](chapter-3/README.md)
- [Chapter 4](chapter-4/README.md)

## Running the code

The codebase assumes you have access to Nvidia hardware and have the NVCC installed. For every chapter, we provide details on how to run the code. Some times it is based on C (usually we give you a make file); other times, for simplicity, we use Python. To configure the Python environment, run the following:


```bash
conda create -n pmpp python=3.11
```

```bash
conda activate pmpp
```

```bash
pip install -r requirements
```
