from pathlib import Path
import torch
from torchvision.io import read_image, write_png
from torch.utils.cpp_extension import load_inline

def compile_extension(): 
    cuda_source = Path("rgb_to_grayscale.c")

    