# Programming Massively Parallel Processors - Complete Solutions 

<div align="center">

<img src="image.png" alt="Book Cover" width="300">

[![CUDA](https://img.shields.io/badge/CUDA-Enabled-green?style=for-the-badge&logo=nvidia)](https://developer.nvidia.com/cuda-zone)
[![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

**Complete solutions to Kirk & Hwu's Programming Massively Parallel Processors (4th Edition)**

*Theoretical explanations + Working implementations + Performance analysis*

</div>

---

## Overview

This repository contains comprehensive solutions to all exercises in Programming Massively Parallel Processors by David Kirk and Wen-mei Hwu (4th Edition). Each chapter includes:

- **Detailed exercise solutions** with step-by-step explanations
- **Working code implementations** in both CUDA C and Python
- **Performance benchmarks** comparing different approaches
- **Visual diagrams** for complex algorithms

## Chapter Organization

Each chapter follows this structure:
```chapter-XX/
├── code/
│   ├── *.cu          # CUDA implementations
│   ├── *.py          # Python alternatives  
│   ├── Makefile      # Build configuration
│   └── ...
└── README.md         # Theory + Exercises + Solutions
```

## Available Chapters

| Chapter | Topic | Focus Areas |
|---------|-------|-------------|
| [Chapter 2](chapter-02/README.md) | **Heterogeneous Data Parallel Computing** | Vector operations, thread mapping, CUDA basics |
| [Chapter 3](chapter-03/README.md) | **Multidimensional Grids and Data** | Grid organization, thread hierarchy |
| [Chapter 4](chapter-04/README.md) | **Compute Architecture and Scheduling** | GPU architecture, warps, occupancy |
| [Chapter 5](chapter-05/README.md) | **Memory Architecture and Data Locality** | Memory types, tiling, bandwidth optimization |
| [Chapter 6](chapter-06/README.md) | **Performance Considerations** | Memory coalescing, latency hiding |
| [Chapter 7](chapter-07/README.md) | **Convolution** | Constant memory, caching, halo cells |
| [Chapter 8](chapter-08/README.md) | **Stencil** | 2D/3D stencil computations, register tiling |
| [Chapter 9](chapter-09/README.md) | **Parallel Histogram** | Atomic operations, privatization, aggregation |
| [Chapter 10](chapter-10/README.md) | **Reduction** | Tree reduction, divergence minimization |
| [Chapter 11](chapter-11/README.md) | **Prefix Sum (Scan)** | Work-efficient algorithms, Kogge-Stone, Brent-Kung |
| [Chapter 12](chapter-12/README.md) | **Merge** | Co-rank function, circular buffers |
| [Chapter 13](chapter-13/README.md) | **Sorting** | Radix sort, merge sort optimization |
| [Chapter 14](chapter-14/README.md) | **Sparse Matrix Computation** | SpMV, CSR/ELL/COO formats |
| [Chapter 15](chapter-15/README.md) | **Graph Traversal** | BFS algorithms, frontier-based approaches |
| [Chapter 16](chapter-16/README.md) | **Deep Learning** | CNN implementation, GEMM formulation |
| [Chapter 17](chapter-17/README.md) | **Iterative MRI Reconstruction** | Medical imaging algorithms |
| [Chapter 18](chapter-18/README.md) | **Electrostatic Potential Map** | Scatter vs gather, cutoff binning |
| [Chapter 19](chapter-19/README.md) | **Parallel Programming and Computational Thinking** | Algorithm selection, problem decomposition |
| [Chapter 20](chapter-20/README.md) | **Heterogeneous Computing Cluster** | CUDA streams, MPI integration |
| [Chapter 21](chapter-21/README.md) | **CUDA Dynamic Parallelism** | Recursive algorithms, quadtrees |

## Quick Start

### Prerequisites
- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- Python 3.11+ (optional, for Python examples)

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd pmpp

# For Python examples (optional)
conda create -n pmpp python=3.11
conda activate pmpp
pip install -r requirements.txt
```

### Running Examples

**CUDA/C Examples:**
```bash
cd chapter-XX/code
make
./program_name
```

**Python Examples:**
```bash
cd chapter-XX/code
python script_name.py
```

## Contributing

Found an error? Please open an issue using this template:

**Describe the bug**

Describe where the problem is and what precisely is wrong.

**Proposed solution**

Here paste your proposed solution. Please include the reasoning behind why you believe your solution is correct.

### Contribution Guidelines
- Maintain the existing explanation style with clear reasoning
- Include working code for any new implementations
- Add performance data where relevant
- Follow the existing code formatting standards

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
