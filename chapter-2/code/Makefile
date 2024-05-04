export PATH := /usr/local/cuda/bin:$(PATH)
export LD_LIBRARY_PATH := /usr/local/cuda/lib64:$(LD_LIBRARY_PATH)

all: run clean

run: vecMul
	./vecMul

vecMul: vecMul.cu
	nvcc -o vecMul vecMul.cu

clean:
	rm -f vecMul