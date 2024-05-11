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
    ext = compile_extension()

    DEVICE, DTYPE = "cuda", torch.float32

    M = torch.randn(300, 128).to(DEVICE, DTYPE)
    N = torch.randn(128, 300).to(DEVICE, DTYPE)

    P = ext.matrixMul(M, N)

    print(torch.allclose(P, M@N, rtol=1e-3, atol=1e-3))
    diff = P - (M@N)
    # print(f"Sum: {torch.abs(diff).sum()}, Mean {torch.abs(diff).mean()}, Max {torch.abs(diff).max()}")
    print()
    print(P[:4, :4])
    print((M@N)[:4, :4])



if __name__ == "__main__":
    main()
