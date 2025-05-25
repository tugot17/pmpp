# Unified Makefile for CUDA 3D Stencil Benchmark and Heat Simulation

# Compiler and flags
NVCC = nvcc
BENCHMARK_TARGET = stencil_benchmark
SHARED_LIB = libheat_cuda.so

# CUDA architectures (adjust based on your GPU)
CUDA_ARCH = -arch=sm_70

# Compiler flags
NVCCFLAGS = $(CUDA_ARCH) -O3 -std=c++11 -Xcompiler -Wall
SHARED_FLAGS = -shared -Xcompiler -fPIC
LDFLAGS = -lcudart

# Source files
STENCIL_SOURCES = stencil.cu
BENCHMARK_SOURCES = benchmark.cu  
HEAT_SOURCES = heat_interface.cu
HEADERS = stencil.h

# Object files
STENCIL_OBJECTS = $(STENCIL_SOURCES:.cu=.o)
BENCHMARK_OBJECTS = $(BENCHMARK_SOURCES:.cu=.o)
HEAT_OBJECTS = $(HEAT_SOURCES:.cu=.o)
ALL_OBJECTS = $(STENCIL_OBJECTS) $(BENCHMARK_OBJECTS) $(HEAT_OBJECTS)

# Default target - build both benchmark and shared library
all: $(BENCHMARK_TARGET) $(SHARED_LIB)

# Benchmark executable (original functionality)
$(BENCHMARK_TARGET): $(STENCIL_OBJECTS) $(BENCHMARK_OBJECTS)
	@echo "Linking $(BENCHMARK_TARGET)..."
	$(NVCC) $(NVCCFLAGS) $(STENCIL_OBJECTS) $(BENCHMARK_OBJECTS) -o $(BENCHMARK_TARGET) $(LDFLAGS)

# Shared library for Python interface
$(SHARED_LIB): $(STENCIL_OBJECTS) $(HEAT_OBJECTS)
	@echo "Creating shared library $(SHARED_LIB)..."
	$(NVCC) $(NVCCFLAGS) $(SHARED_FLAGS) $(STENCIL_OBJECTS) $(HEAT_OBJECTS) -o $(SHARED_LIB) $(LDFLAGS)

# Compile CUDA source files (with PIC for shared library)
%.o: %.cu $(HEADERS)
	@echo "Compiling $<..."
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fPIC -c $< -o $@

# Clean target
clean:
	@echo "Cleaning up..."
	rm -f $(ALL_OBJECTS) $(BENCHMARK_TARGET) $(SHARED_LIB)

# Build only benchmark
benchmark: $(BENCHMARK_TARGET)

# Build only shared library
shared: $(SHARED_LIB)

# Run benchmark
run: $(BENCHMARK_TARGET)
	./$(BENCHMARK_TARGET)

# Test heat simulation
test_heat: $(SHARED_LIB)
	@echo "Testing heat simulation..."
	python3 test_heat.py

# Run heat simulation
heat: $(SHARED_LIB)
	@echo "Starting heat simulation..."
	python3 heat_simulation.py

# Test shared library
test_shared: $(SHARED_LIB)
	@echo "Testing shared library..."
	@if [ -f $(SHARED_LIB) ]; then \
		echo "✓ Shared library $(SHARED_LIB) created successfully"; \
		ldd $(SHARED_LIB) | grep cuda || echo "⚠ CUDA libraries not found in ldd output"; \
	else \
		echo "✗ Shared library creation failed"; \
	fi

# Debug target with debug symbols
debug: NVCCFLAGS += -g -G -lineinfo
debug: $(BENCHMARK_TARGET) $(SHARED_LIB)

# Profile target with profiling flags
profile: NVCCFLAGS += -lineinfo
profile: $(BENCHMARK_TARGET) $(SHARED_LIB)

# Install shared library to system (optional)
install: $(SHARED_LIB)
	@echo "Installing $(SHARED_LIB) to /usr/local/lib..."
	@sudo cp $(SHARED_LIB) /usr/local/lib/
	@sudo ldconfig

# Help target
help:
	@echo "Available targets:"
	@echo "  all         - Build both benchmark and shared library (default)"
	@echo "  benchmark   - Build only the benchmark executable"
	@echo "  shared      - Build only the shared library for Python"
	@echo "  run         - Build and run the benchmark"
	@echo "  heat        - Build shared library and run heat simulation"
	@echo "  test_heat   - Build shared library and run heat tests"
	@echo "  test_shared - Test if shared library was created properly"
	@echo "  clean       - Remove all object files and executables"
	@echo "  debug       - Build with debug symbols"
	@echo "  profile     - Build with profiling information"
	@echo "  install     - Install shared library to system (requires sudo)"
	@echo "  help        - Show this help message"
	@echo ""
	@echo "Quick start for heat simulation:"
	@echo "  make heat"
	@echo ""
	@echo "To change CUDA architecture, modify CUDA_ARCH variable"
	@echo "Current setting: $(CUDA_ARCH)"

# Declare phony targets
.PHONY: all clean shared benchmark run heat test_heat test_shared debug profile install help

# Dependency tracking
stencil.o: stencil.cu stencil.h
benchmark.o: benchmark.cu stencil.h
heat_interface.o: heat_interface.cu stencil.h