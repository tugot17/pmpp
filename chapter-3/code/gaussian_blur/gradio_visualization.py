import gradio as gr
from gaussian_blur import compile_extension
from pathlib import Path
import numpy as np
import torch

ext = compile_extension()

def blur(image: np.ndarray, blur_size: int):
    image = torch.tensor(image).to("cuda").permute(2, 0, 1).contiguous()
    y = ext.gaussian_blur(image, blur_size)
    return y.cpu().permute(1, 2, 0).numpy()

image_path = Path(__file__).parent.parent / "Grace_Hopper.jpg"

demo = gr.Interface(
    fn=blur,
    inputs=["image", gr.Slider(minimum=0, maximum=30, step=1, value=3)],
    outputs=["image"],
    examples=[[str(image_path), 3]]
)

if __name__ == "__main__":
    demo.launch()   
