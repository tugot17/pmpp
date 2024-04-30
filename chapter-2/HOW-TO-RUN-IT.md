## How to run it

Export paths:

```
export PATH=/usr/local/cuda/bin:$PATH
```

```
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

Compile:

```
nvcc -o vecMul vecMul.cu
```

Run

```
./vecMul
```
