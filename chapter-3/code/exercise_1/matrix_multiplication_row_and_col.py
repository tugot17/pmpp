from pathlib import Path
from torch.utils.cpp_extension import load_inline
import torch


def compile_extension():
    cuda_source = (Path(__file__).parent / "matrix_multiplication_row_and_col.cu").read_text()
    
    cpp_sources = ["torch::Tensor matrixRowMul(torch::Tensor M, torch::Tensor N);", "torch::Tensor matrixColMul(torch::Tensor M, torch::Tensor N);"]
    
    return load_inline(
        name="matrixMul_extension",
        cpp_sources=cpp_sources,
        cuda_sources=cuda_source,
        functions=["matrixRowMul", "matrixColMul"],
        with_cuda=True,
    )


def main():
    ext = compile_extension()

    DEVICE, DTYPE = "cuda", torch.float32

    M = torch.randn(4, 4).to(DEVICE, DTYPE)
    N = torch.randn(4, 4).to(DEVICE, DTYPE)

    P_row = ext.matrixRowMul(M, N)
    P_col = ext.matrixColMul(M, N)

    torch_P = torch.matmul(M, N)

    all_close = torch.allclose(P_row, torch_P, rtol=1e-3, atol=1e-3) & torch.allclose(P_col, torch_P, rtol=1e-3, atol=1e-3)

    print(f"All close: {all_close}")
    print()
    print(P_row[:4, :4])
    print()
    print(P_col[:4, :4])
    print()
    print(torch_P[:4, :4])



if __name__ == "__main__":
    main()
