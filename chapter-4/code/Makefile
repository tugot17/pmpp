export PATH := /usr/local/cuda/bin:$(PATH)
export LD_LIBRARY_PATH := /usr/local/cuda/lib64:$(LD_LIBRARY_PATH)

all: run clean

run: deviceProperties
	./deviceProperties

deviceProperties: deviceProperties.cu
	nvcc -o deviceProperties deviceProperties.cu

clean:
	rm -f deviceProperties