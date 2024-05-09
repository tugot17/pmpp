from pathlib import Path
from time import time

import torch
from torch.utils.cpp_extension import load_inline


def vector_multipication_loop(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.device == B.device and A.device.type == "cuda"
    assert A.dtype == B.dtype == torch.float32
    assert A.size() == B.size()

    C = torch.empty_like(A)
    n = A.size(0)
    for i in range(n):
        C[i] = A[i] * B[i]
    return C


def compile_extension():
    cuda_source = (Path(__file__).parent / "vecMulTorchTensor.cu").read_text()
    cpp_source = (
        "torch::Tensor vector_multiplication(torch::Tensor A, torch::Tensor B_h);"
    )

    return load_inline(
        name="extension",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["vector_multiplication"],
        with_cuda=True,
        # extra_cuda_cflags=["-O2"]
    )


def main():
    ext = compile_extension()

    DEVICE = "cuda"
    DTYPE = torch.float32
    NUM_ELEMENTS = 500_000

    A = torch.tensor([i for i in range(NUM_ELEMENTS)]).to(DEVICE, DTYPE)
    B = torch.tensor([i for i in range(NUM_ELEMENTS)]).to(DEVICE, DTYPE)

    start = time()
    y_custom_kernel = ext.vector_multiplication(A, B)
    stop = time()
    print(f"Cuda custom kernel multiply: {stop - start:.2f}s")

    start = time()
    vector_multipication_loop(A, B)
    stop = time()
    print(f"Python loop: {stop - start:.2f}s")

    start = time()
    A + B
    stop = time()
    print(f"Adding via PyTorch addition: {stop - start:.2f}s")

    print("Size:", y_custom_kernel.size())
    print("Y:", y_custom_kernel[:10])


if __name__ == "__main__":
    main()
