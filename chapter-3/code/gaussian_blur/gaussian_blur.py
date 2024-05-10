from pathlib import Path

from torch.utils.cpp_extension import load_inline
from torchvision.io import read_image, write_png


def compile_extension():
    cuda_source = (Path(__file__).parent / "gaussian_blur.cu").read_text()
    cpp_source = "torch::Tensor gaussian_blur(torch::Tensor img, int blurSize);"

    return load_inline(
        name="gaussian_blur_extension",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["gaussian_blur"],
        with_cuda=True,
    )


def main():
    current_dir = Path(__file__).parent
    x = read_image(current_dir.parent / "Grace_Hopper.jpg").contiguous().cuda()

    print("Original:")
    print("mean:", x.float().mean())
    print("Input image:", x.shape, x.dtype)
    print()

    ext = compile_extension()
    blur_size = 3

    y = ext.gaussian_blur(x, blur_size)

    print("Converted:")
    print("mean:", y.float().mean())
    print("Input image:", y.shape, y.dtype)

    save_path = current_dir / "output.png"
    print(f"Blured the image. It is saved at {save_path}")
    write_png(y.cpu(), save_path)


if __name__ == "__main__":
    main()
