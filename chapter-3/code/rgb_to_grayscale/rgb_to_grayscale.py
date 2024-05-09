from pathlib import Path

from torch.utils.cpp_extension import load_inline
from torchvision.io import read_image, write_png


def compile_extension():
    cuda_source = (Path(__file__).parent / "rgb_to_grayscale.cu").read_text()
    cpp_source = "torch::Tensor rgb_to_gray(torch::Tensor img);"

    return load_inline(
        name="rgb_to_grey_extension",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["rgb_to_gray"],
        with_cuda=True,
    )


def main():
    current_dir = Path(__file__).parent
    x = read_image(current_dir.parent / "Grace_Hopper.jpg").permute(1, 2, 0).cuda()

    print("Original:")
    print("mean:", x.float().mean())
    print("Input image:", x.shape, x.dtype)
    print()

    ext = compile_extension()
    y = ext.rgb_to_gray(x)

    print("Converted:")
    print("mean:", y.float().mean())
    print("Input image:", y.shape, y.dtype)

    save_path = current_dir / "output.png"
    print(f"Converted to grayscale, saved at {save_path}")
    write_png(y.permute(2, 0, 1).cpu(), save_path)


if __name__ == "__main__":
    main()
