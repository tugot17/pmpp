# Use nvcc for compiling CUDA code
NVCC = nvcc
TARGET = radix_sort_app
SRCS = main.cu gpu_radix_sort.cu gpu_merge_sort.cu
OBJS = $(SRCS:.cu=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) -o $@ $^
	@rm -f $(OBJS)
	@echo "Build complete. Object files removed."

%.o: %.cu
	$(NVCC) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
