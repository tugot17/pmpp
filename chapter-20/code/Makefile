# Makefile for MPI + CUDA stencil code

# Compiler settings
NVCC = nvcc
MPICC = mpicc
TARGET = stencil_mpi
SOURCE = stencil_mpi.cu

# Get MPI compilation flags
MPI_COMPILE_FLAGS = $(shell mpicc --showme:compile)
MPI_LINK_FLAGS = $(shell mpicc --showme:link)

# CUDA flags
CUDA_FLAGS = -lcuda -lcudart

# Compilation rule using nvcc with MPI flags
$(TARGET): $(SOURCE)
	$(NVCC) -o $(TARGET) $(SOURCE) $(MPI_COMPILE_FLAGS) $(MPI_LINK_FLAGS)

# Alternative compilation using mpicc
alt: $(SOURCE)
	$(MPICC) -o $(TARGET) $(SOURCE) $(CUDA_FLAGS)

# Clean rule
clean:
	rm -f $(TARGET)

# Run with 4 processes
run: $(TARGET)
	mpirun -np 3 ./$(TARGET)

.PHONY: clean run alt