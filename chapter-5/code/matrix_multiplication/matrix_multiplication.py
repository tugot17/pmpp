from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline


def compile_extension():
    cuda_source = (Path(__file__).parent / "matrix_multiplication.cu").read_text()
    cpp_source = "torch::Tensor tiledSquareMatrixMul(torch::Tensor M, torch::Tensor N);"

    return load_inline(
        name="matrixMul_extension",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["tiledSquareMatrixMul"],
        with_cuda=True,
    )

import sys
import io

def capture_cuda_output(func):
    def wrapper(*args, **kwargs):
        # Redirect stdout to capture CUDA output
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        try:
            result = func(*args, **kwargs)
            cuda_output = sys.stdout.getvalue()
            print("CUDA Kernel Output:")
            print(cuda_output)
        finally:
            # Restore stdout
            sys.stdout = old_stdout
        
        return result
    return wrapper


def main():
    ext = compile_extension()

    DEVICE, DTYPE = "cuda", torch.float32

    M = torch.randn(16, 15).to(DEVICE, DTYPE)
    N = torch.randn(128, 128).to(DEVICE, DTYPE)

    P = ext.tiledSquareMatrixMul(M, N)

    torch_P = torch.matmul(M, N)

    print(torch.allclose(P, torch_P, rtol=1e-3, atol=1e-3))
    print()
    print(P[:4, :4])
    print(torch_P[:4, :4])


if __name__ == "__main__":
    main()
