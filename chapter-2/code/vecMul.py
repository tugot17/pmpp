from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline

def compile_extension():
    cuda_source = (Path(__file__).parent / "vecMulTorchTensor.cu").read_text()
    cpp_source = "torch::Tensor vecMulDevice"

    return load_inline(
        name="vector_multiplication",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["vecMulDevice"],
        with_cuda=True,
        extra_cuda_cflags=["-O2"]
    )

def main():
    ext = compile_extension()

    a = torch.tensor([i for i in range(1000)]).cuda()
    b = torch.tensor([i for i in range(1000)]).cuda()

    y = ext(a, b)

    print("Size:", y.size())
    print("Y:", y)

if __name__ == "__main__":
    main()
