# Makefile for CUDA 3D Stencil Benchmark

# Compiler and flags
NVCC = nvcc
TARGET = stencil_benchmark

# CUDA architectures (adjust based on your GPU)
# Common options: 
# - sm_35 for older GPUs (Kepler)
# - sm_50, sm_52 for Maxwell
# - sm_60, sm_61 for Pascal  
# - sm_70, sm_75 for Volta/Turing
# - sm_80, sm_86 for Ampere
# - sm_89, sm_90 for Ada Lovelace/Hopper
CUDA_ARCH = -arch=sm_70

# Compiler flags
NVCCFLAGS = $(CUDA_ARCH) -O3 -std=c++11 -Xcompiler -Wall
LDFLAGS = -lcudart

# Source files
SOURCES = stencil.cu benchmark.cu
HEADERS = stencil.h
OBJECTS = $(SOURCES:.cu=.o)

# Default target
all: $(TARGET)

# Link the final executable
$(TARGET): $(OBJECTS)
	@echo "Linking $(TARGET)..."
	$(NVCC) $(NVCCFLAGS) $(OBJECTS) -o $(TARGET) $(LDFLAGS)

# Compile CUDA source files
%.o: %.cu $(HEADERS)
	@echo "Compiling $<..."
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Clean target
clean:
	@echo "Cleaning up..."
	rm -f $(OBJECTS) $(TARGET)

# Run target (optional)
run: $(TARGET)
	./$(TARGET)

# Debug target with debug symbols
debug: NVCCFLAGS += -g -G -lineinfo
debug: $(TARGET)

# Profile target with profiling flags
profile: NVCCFLAGS += -lineinfo
profile: $(TARGET)

# Help target
help:
	@echo "Available targets:"
	@echo "  all     - Build the stencil benchmark (default)"
	@echo "  clean   - Remove object files and executable"
	@echo "  run     - Build and run the benchmark"
	@echo "  debug   - Build with debug symbols"
	@echo "  profile - Build with profiling information"
	@echo "  help    - Show this help message"
	@echo ""
	@echo "To change CUDA architecture, modify CUDA_ARCH variable"
	@echo "Current setting: $(CUDA_ARCH)"

# Declare phony targets
.PHONY: all clean run debug profile help

# Dependency tracking
stencil.o: stencil.cu stencil.h
benchmark.o: benchmark.cu stencil.h