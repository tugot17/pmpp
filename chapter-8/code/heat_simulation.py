#!/usr/bin/env python3
"""
3D Heat Equation Simulation using CUDA Stencil Implementation

This script demonstrates how to use the CUDA stencil implementation
for solving the 3D heat equation and creating animations.
"""

import ctypes
import os
from typing import Optional

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


class HeatEquationCUDA:
    """3D Heat Equation solver using CUDA implementation"""

    def __init__(self, library_path: str = "./libheat_cuda.so"):
        """Initialize the CUDA heat equation solver

        Args:
            library_path: Path to the compiled shared library
        """
        self.lib = None
        self.load_library(library_path)

    def load_library(self, library_path: str):
        """Load the CUDA shared library"""
        if not os.path.exists(library_path):
            raise FileNotFoundError(f"Shared library not found: {library_path}")

        try:
            self.lib = ctypes.CDLL(library_path)

            # Define function signatures
            # void heat_step_cuda_float(float* data, unsigned int N, float alpha, float dt, float dx)
            self.lib.heat_step_cuda_float.argtypes = [
                ctypes.POINTER(ctypes.c_float),  # data
                ctypes.c_uint,  # N
                ctypes.c_float,  # alpha
                ctypes.c_float,  # dt
                ctypes.c_float,  # dx
            ]
            self.lib.heat_step_cuda_float.restype = None

            # void init_cuda()
            self.lib.init_cuda.argtypes = []
            self.lib.init_cuda.restype = None

            # void cleanup_cuda()
            self.lib.cleanup_cuda.argtypes = []
            self.lib.cleanup_cuda.restype = None

            # Initialize CUDA
            self.lib.init_cuda()
            print("✓ CUDA library loaded successfully")

        except Exception as e:
            print(f"✗ Failed to load CUDA library: {e}")
            raise

    def solve_step(self, data: np.ndarray, alpha: float, dt: float, dx: float):
        """Perform one time step of heat equation solution

        Args:
            data: 3D numpy array containing temperature data
            alpha: Thermal diffusivity
            dt: Time step
            dx: Spatial step size
        """
        if data.dtype != np.float32:
            raise ValueError("Data must be float32 type")

        N = data.shape[0]
        if data.shape != (N, N, N):
            raise ValueError("Data must be cubic (N×N×N)")

        # Ensure contiguous memory layout
        data_flat = np.ascontiguousarray(data.flatten().astype(np.float32))

        # Create ctypes pointer
        data_ptr = data_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Call CUDA function
        self.lib.heat_step_cuda_float(data_ptr, N, alpha, dt, dx)

        # Reshape back to 3D
        return data_flat.reshape((N, N, N))

    def __del__(self):
        """Cleanup CUDA resources"""
        if hasattr(self, "lib") and self.lib:
            self.lib.cleanup_cuda()


class HeatSimulation:
    """3D Heat Equation Simulation Manager"""

    def __init__(self, N: int = 64, alpha: float = 0.01, dx: float = 1.0):
        """Initialize simulation parameters

        Args:
            N: Grid size (N×N×N)
            alpha: Thermal diffusivity
            dx: Spatial step size
        """
        self.N = N
        self.alpha = alpha
        self.dx = dx

        # Calculate stable time step (CFL condition for 3D)
        # For stability: dt ≤ dx²/(6*alpha)
        self.dt_max = dx * dx / (6.0 * alpha)
        self.dt = 0.8 * self.dt_max  # Use 80% of maximum for safety

        print("Simulation parameters:")
        print(f"  Grid size: {N}×{N}×{N}")
        print(f"  Thermal diffusivity α: {alpha}")
        print(f"  Spatial step dx: {dx}")
        print(f"  Time step dt: {self.dt:.6f} (max: {self.dt_max:.6f})")
        print(f"  Diffusion number r: {alpha * self.dt / (dx * dx):.6f}")

        # Initialize CUDA solver
        self.solver = HeatEquationCUDA()

        # Initialize data
        self.data = self.create_initial_conditions()
        self.time = 0.0

    def create_initial_conditions(self) -> np.ndarray:
        """Create initial temperature distribution"""
        data = np.zeros((self.N, self.N, self.N), dtype=np.float32)

        # Create a hot sphere in the center
        center = self.N // 2
        radius = self.N // 8

        for i in range(self.N):
            for j in range(self.N):
                for k in range(self.N):
                    dist_sq = (i - center) ** 2 + (j - center) ** 2 + (k - center) ** 2
                    if dist_sq <= radius**2:
                        # Hot spot with temperature 100
                        data[i, j, k] = 100.0
                    else:
                        # Ambient temperature
                        data[i, j, k] = 20.0

        return data

    def step(self):
        """Advance simulation by one time step"""
        self.data = self.solver.solve_step(self.data, self.alpha, self.dt, self.dx)
        self.time += self.dt

    def get_slice(self, axis: int = 2, index: Optional[int] = None) -> np.ndarray:
        """Get a 2D slice of the 3D data for visualization

        Args:
            axis: Which axis to slice (0, 1, or 2)
            index: Index along the axis (None for center)
        """
        if index is None:
            index = self.N // 2

        if axis == 0:
            return self.data[index, :, :]
        elif axis == 1:
            return self.data[:, index, :]
        elif axis == 2:
            return self.data[:, :, index]
        else:
            raise ValueError("Axis must be 0, 1, or 2")


class HeatAnimator:
    """Animator for heat equation simulation"""

    def __init__(self, simulation: HeatSimulation):
        self.sim = simulation
        self.fig = None
        self.ax = None
        self.im = None
        self.cbar = None
        self.text = None

    def setup_plot(self):
        """Setup matplotlib figure and axes"""
        self.fig, self.ax = plt.subplots(figsize=(10, 8))

        # Initial plot
        initial_slice = self.sim.get_slice()
        self.im = self.ax.imshow(
            initial_slice,
            cmap="hot",
            interpolation="bilinear",
            vmin=0,
            vmax=100,
            origin="lower",
        )

        self.cbar = plt.colorbar(self.im, ax=self.ax)
        self.cbar.set_label("Temperature")

        self.ax.set_title("3D Heat Equation - Central Z-slice")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")

        # Time text
        self.text = self.ax.text(
            0.02,
            0.98,
            "",
            transform=self.ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        plt.tight_layout()

    def animate_frame(self, frame):
        """Animation function for matplotlib"""
        # Advance simulation
        for _ in range(5):  # Multiple steps per frame for faster animation
            self.sim.step()

        # Update plot
        slice_data = self.sim.get_slice()
        self.im.set_array(slice_data)

        # Update time text
        self.text.set_text(
            f"Time: {self.sim.time:.4f}\nMax T: {np.max(self.sim.data):.1f}°\nMin T: {np.min(self.sim.data):.1f}°"
        )

        return [self.im, self.text]

    def create_animation(self, frames: int = 200, interval: int = 50):
        """Create and return animation object"""
        self.setup_plot()
        anim = animation.FuncAnimation(
            self.fig,
            self.animate_frame,
            frames=frames,
            interval=interval,
            blit=True,
            repeat=True,
        )
        return anim


def main():
    """Main function to run heat equation simulation"""

    # Check if shared library exists
    lib_path = "./libheat_cuda.so"
    if not os.path.exists(lib_path):
        print("✗ Shared library not found!")
        print("Please compile it first using:")
        print("  make -f Makefile_shared shared")
        return

    try:
        # Create simulation
        print("Initializing 3D Heat Equation Simulation...")
        sim = HeatSimulation(N=64, alpha=0.01, dx=1.0)

        # Create animator
        animator = HeatAnimator(sim)

        # Create and show animation
        print("Creating animation...")
        anim = animator.create_animation(frames=300, interval=50)

        print("Starting animation (close window to exit)...")
        plt.show()

        # Optionally save animation
        save_animation = input("Save animation? (y/n): ").lower() == "y"
        if save_animation:
            # Check available writers
            available_writers = animation.writers.list()
            print(f"Available animation writers: {available_writers}")

            if "ffmpeg" in available_writers:
                try:
                    print("Saving animation as MP4... (this may take a while)")
                    anim.save(
                        "heat_equation_3d.mp4", writer="ffmpeg", fps=20, bitrate=1800
                    )
                    print("✓ Animation saved as heat_equation_3d.mp4")
                except Exception as e:
                    print(f"✗ Failed to save as MP4: {e}")
                    print("Trying GIF format instead...")
                    try:
                        anim.save("heat_equation_3d.gif", writer="pillow", fps=10)
                        print("✓ Animation saved as heat_equation_3d.gif")
                    except Exception as e2:
                        print(f"✗ Failed to save animation: {e2}")
            else:
                # Fallback to GIF
                try:
                    print(
                        "ffmpeg not available. Saving as GIF... (this may take a while)"
                    )
                    anim.save("heat_equation_3d.gif", writer="pillow", fps=10)
                    print("✓ Animation saved as heat_equation_3d.gif")
                except Exception as e:
                    print(f"✗ Failed to save animation: {e}")

    except Exception as e:
        print(f"✗ Error running simulation: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
