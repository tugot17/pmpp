from pathlib import Path
import torch
from torchvision.io import read_image, write_png
from torch.utils.cpp_extension import load_inline
from PIL import Image
from torchvision.transforms import ToTensor

def compile_extension(): 
    cuda_source = (Path(__file__).parent / "rgb_to_grayscale.cu").read_text()
    cpp_source = "torch::Tensor rgb_to_grey(torch::Tensor img);"

    return load_inline(
        name="rgb_to_grey_extension",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["rgb_to_grey"],
        with_cuda=True
    )

def main():
    
    DEVICE = "cuda"
    DTYPE = torch.uint8
    SIZE = (128, 128)
    current_dir = Path(__file__).parent

    # img_path = Path(__file__).parent.parent / "wroclaw_colorful.png"
    # img = Image.open(img_path).convert('RGB')
    # print(img)
    # x = ToTensor()(img).permute(1, 2, 0).to(DEVICE, DTYPE)
    # print(x)
    x = read_image(current_dir / "Grace_Hopper.jpg").permute(1, 2, 0).cuda()

    print("mean:", x.float().mean())
    print("Input image:", x.shape, x.dtype)

    ext = compile_extension()
    y = ext.rgb_to_grey(x)

    print()
    print("mean:", y.float().mean())
    print("Input image:", y.shape, y.dtype)
    write_png(y.permute(2, 0, 1).cpu(), current_dir / "output.png")


    



if __name__ == "__main__":
    main()