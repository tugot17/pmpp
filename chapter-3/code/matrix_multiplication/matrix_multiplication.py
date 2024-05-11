from pathlib import Path
from torch.utils.cpp_extension import load_inline
import torch


def compile_extension():
    cuda_source = (Path(__file__).parent / "matrix_multiplication.cu").read_text()
    cpp_source = "torch::Tensor matrixMul(torch::Tensor M, torch::Tensor N);"

    return load_inline(
        name="matrixMul_extension",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["matrixMul"],
        with_cuda=True,
    )


def main():
    current_dir = Path(__file__).parent

    ext = compile_extension()

    DEVICE, DTYPE = "cuda", torch.float32

    M = torch.randn(3, 3).to(DEVICE, DTYPE)
    N = torch.randn(3, 3).to(DEVICE, DTYPE)

    P = ext.matrixMul(M, N)

    print(torch.allclose(P, M@N))
    print()
    print(P)
    print(M@N)



if __name__ == "__main__":
    main()
