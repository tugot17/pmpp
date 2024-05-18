from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline


def compile_extension():
    cuda_source = (
        Path(__file__).parent / "matrix_vector_multiplication.cu"
    ).read_text()
    cpp_source = (
        "torch::Tensor matrix_vector_multiplication(torch::Tensor B, torch::Tensor c);"
    )

    return load_inline(
        name="matrixMul_extension",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["matrix_vector_multiplication"],
        with_cuda=True,
    )


def main():
    ext = compile_extension()

    DEVICE, DTYPE = "cuda", torch.float32

    B = torch.randn(1000, 256).to(DEVICE, DTYPE)
    c = torch.randn(256).to(DEVICE, DTYPE)

    res = ext.matrix_vector_multiplication(B, c)

    torch_res = torch.matmul(B, c)

    print(res.shape, torch_res.shape)

    print(torch.allclose(res, torch_res, rtol=1e-3, atol=1e-3))
    print()
    print(res[:4])
    print(torch_res[:4])


if __name__ == "__main__":
    main()
