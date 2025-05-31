#!/usr/bin/env python3
"""
Bezier Curve Tessellation and Visualization using CUDA
Compiles CUDA code and visualizes tessellated Bezier curves
"""

import ctypes
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os
import sys
from dataclasses import dataclass
from typing import List, Tuple

# Structure definition matching the C code
class BezierLineC(ctypes.Structure):
    _fields_ = [
        ("CP", ctypes.c_float * 6),        # 3 control points * 2 coordinates
        ("vertexPos", ctypes.c_float * 64), # MAX_TESS_POINTS * 2 coordinates  
        ("nVertices", ctypes.c_int)
    ]

@dataclass
class BezierCurve:
    """Python-friendly Bezier curve representation"""
    p0: Tuple[float, float]  # Start point
    p1: Tuple[float, float]  # Control point
    p2: Tuple[float, float]  # End point
    
    def to_c_struct(self) -> BezierLineC:
        """Convert to C structure"""
        line = BezierLineC()
        line.CP[0] = self.p0[0]  # P0.x
        line.CP[1] = self.p0[1]  # P0.y
        line.CP[2] = self.p1[0]  # P1.x
        line.CP[3] = self.p1[1]  # P1.y
        line.CP[4] = self.p2[0]  # P2.x
        line.CP[5] = self.p2[1]  # P2.y
        line.nVertices = 0
        return line

class BezierTessellator:
    """CUDA-accelerated Bezier curve tessellator"""
    
    def __init__(self, cuda_lib_path: str = None):
        self.lib = None
        self.cuda_lib_path = cuda_lib_path or "libbezier.so"
        self._compile_and_load()
    
    def _compile_and_load(self):
        """Compile CUDA code and load shared library"""
        print("Compiling CUDA code...")
        
        # Check if helper_math.h exists, if not create a minimal version
        if not os.path.exists("helper_math.h"):
            self._create_helper_math()
        
        # Compile command
        compile_cmd = [
            "nvcc", 
            "-shared", 
            "-Xcompiler", "-fPIC",
            "bezier_curves.cu", 
            "-o", self.cuda_lib_path
        ]
        
        try:
            result = subprocess.run(compile_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print("Compilation failed!")
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                sys.exit(1)
            print("Compilation successful!")
        except FileNotFoundError:
            print("Error: nvcc not found. Make sure CUDA toolkit is installed and in PATH.")
            sys.exit(1)
        
        # Load the shared library
        try:
            self.lib = ctypes.CDLL(f"./{self.cuda_lib_path}")
            self._setup_function_signatures()
            print("Library loaded successfully!")
        except OSError as e:
            print(f"Error loading library: {e}")
            sys.exit(1)
    
    def _create_helper_math(self):
        """Create a minimal helper_math.h if it doesn't exist"""
        helper_math_content = """
#ifndef HELPER_MATH_H
#define HELPER_MATH_H

#include <cuda_runtime.h>
#include <math.h>

// Basic float2 operations
__device__ __host__ inline float2 make_float2(float x, float y) {
    float2 t; t.x = x; t.y = y; return t;
}

__device__ __host__ inline float2 operator+(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

__device__ __host__ inline float2 operator-(float2 a, float2 b) {
    return make_float2(a.x - b.x, a.y - b.y);
}

__device__ __host__ inline float2 operator*(float s, float2 a) {
    return make_float2(s * a.x, s * a.y);
}

__device__ __host__ inline float2 operator/(float2 a, float s) {
    return make_float2(a.x / s, a.y / s);
}

__device__ __host__ inline float length(float2 v) {
    return sqrtf(v.x * v.x + v.y * v.y);
}

__device__ __host__ inline float dot(float2 a, float2 b) {
    return a.x * b.x + a.y * b.y;
}

#endif
"""
        with open("helper_math.h", "w") as f:
            f.write(helper_math_content)
        print("Created helper_math.h")
    
    def _setup_function_signatures(self):
        """Setup function signatures for ctypes"""
        # tessellate_bezier_curves function
        self.lib.tessellate_bezier_curves.argtypes = [
            ctypes.POINTER(BezierLineC), 
            ctypes.c_int
        ]
        self.lib.tessellate_bezier_curves.restype = ctypes.c_int
        
        # Helper functions
        self.lib.get_cuda_device_count.restype = ctypes.c_int
        self.lib.print_cuda_error.argtypes = []
        self.lib.print_cuda_error.restype = None
    
    def tessellate(self, curves: List[BezierCurve]) -> List[np.ndarray]:
        """
        Tessellate Bezier curves using CUDA
        
        Args:
            curves: List of BezierCurve objects
            
        Returns:
            List of numpy arrays, each containing tessellated points (Nx2)
        """
        if not curves:
            return []
        
        # Convert to C structures
        n_curves = len(curves)
        c_lines = (BezierLineC * n_curves)()
        for i, curve in enumerate(curves):
            c_lines[i] = curve.to_c_struct()
        
        # Call CUDA function
        result = self.lib.tessellate_bezier_curves(c_lines, n_curves)
        
        if result != 0:
            print(f"CUDA tessellation failed with error code: {result}")
            self.lib.print_cuda_error()
            return []
        
        # Extract results
        tessellated_curves = []
        for i in range(n_curves):
            n_vertices = c_lines[i].nVertices
            if n_vertices > 0:
                # Extract vertices
                vertices = np.zeros((n_vertices, 2))
                for j in range(n_vertices):
                    vertices[j, 0] = c_lines[i].vertexPos[j * 2]     # x
                    vertices[j, 1] = c_lines[i].vertexPos[j * 2 + 1] # y
                tessellated_curves.append(vertices)
            else:
                tessellated_curves.append(np.array([]))
        
        return tessellated_curves
    
    def get_device_count(self) -> int:
        """Get number of CUDA devices"""
        return self.lib.get_cuda_device_count()

def create_sample_curves() -> List[BezierCurve]:
    """Create some interesting sample Bezier curves"""
    curves = [
        # Simple arc
        BezierCurve(p0=(0.0, 0.0), p1=(0.5, 1.0), p2=(1.0, 0.0)),
        
        # S-curve
        BezierCurve(p0=(0.0, 1.0), p1=(0.8, 1.5), p2=(1.0, 2.0)),
        
        # Sharp turn
        BezierCurve(p0=(1.0, 2.0), p1=(1.8, 1.2), p2=(2.0, 2.0)),
        
        # Loop-like curve
        BezierCurve(p0=(2.0, 0.0), p1=(3.5, 1.5), p2=(2.5, 0.5)),
        
        # Nearly straight line (low curvature)
        BezierCurve(p0=(0.0, 3.0), p1=(1.0, 3.1), p2=(2.0, 3.0)),
        
        # High curvature curve
        BezierCurve(p0=(2.5, 1.0), p1=(4.0, 0.0), p2=(2.5, 2.0)),
    ]
    return curves

def plot_curves(curves: List[BezierCurve], tessellated: List[np.ndarray]):
    """Visualize the original control points and tessellated curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Colors for different curves
    colors = plt.cm.tab10(np.linspace(0, 1, len(curves)))
    
    # Plot 1: Control points and theoretical curves
    ax1.set_title("Bezier Curves - Control Points", fontsize=14, fontweight='bold')
    
    for i, (curve, color) in enumerate(zip(curves, colors)):
        # Plot control points
        control_x = [curve.p0[0], curve.p1[0], curve.p2[0]]
        control_y = [curve.p0[1], curve.p1[1], curve.p2[1]]
        
        ax1.plot(control_x, control_y, 'o--', color=color, alpha=0.7, 
                linewidth=1, markersize=6, label=f'Curve {i+1} Control')
        
        # Plot theoretical Bezier curve (for reference)
        t = np.linspace(0, 1, 100)
        x_theo = (1-t)**2 * curve.p0[0] + 2*(1-t)*t * curve.p1[0] + t**2 * curve.p2[0]
        y_theo = (1-t)**2 * curve.p0[1] + 2*(1-t)*t * curve.p1[1] + t**2 * curve.p2[1]
        ax1.plot(x_theo, y_theo, '-', color=color, alpha=0.8, linewidth=2)
    
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.set_aspect('equal')
    
    # Plot 2: CUDA tessellated curves
    ax2.set_title("CUDA Tessellated Curves", fontsize=14, fontweight='bold')
    
    for i, (curve, tess_points, color) in enumerate(zip(curves, tessellated, colors)):
        if len(tess_points) > 0:
            # Plot tessellated points
            ax2.plot(tess_points[:, 0], tess_points[:, 1], 'o-', color=color, 
                    markersize=4, linewidth=2, alpha=0.8, 
                    label=f'Curve {i+1} ({len(tess_points)} pts)')
            
            # Highlight start and end points
            ax2.plot(tess_points[0, 0], tess_points[0, 1], 's', color=color, 
                    markersize=8, markeredgecolor='black', markeredgewidth=1)
            ax2.plot(tess_points[-1, 0], tess_points[-1, 1], '^', color=color, 
                    markersize=8, markeredgecolor='black', markeredgewidth=1)
    
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()
    plt.savefig("bezier_curve_visualization.png")

def analyze_tessellation(curves: List[BezierCurve], tessellated: List[np.ndarray]):
    """Analyze the tessellation results"""
    print("\n" + "="*60)
    print("TESSELLATION ANALYSIS")
    print("="*60)
    
    for i, (curve, tess_points) in enumerate(zip(curves, tessellated)):
        if len(tess_points) > 0:
            # Calculate curvature estimate
            p0, p1, p2 = np.array(curve.p0), np.array(curve.p1), np.array(curve.p2)
            chord = p2 - p0
            to_control = p1 - p0
            
            chord_length = np.linalg.norm(chord)
            if chord_length > 0.001:
                chord_norm = chord / chord_length
                projection = np.dot(to_control, chord_norm)
                perpendicular = to_control - projection * chord_norm
                curvature = np.linalg.norm(perpendicular)
            else:
                curvature = 0.0
            
            print(f"Curve {i+1}:")
            print(f"  Control Points: {curve.p0} -> {curve.p1} -> {curve.p2}")
            print(f"  Estimated Curvature: {curvature:.4f}")
            print(f"  Tessellation Points: {len(tess_points)}")
            print(f"  Point Density: {len(tess_points)/chord_length:.2f} pts/unit" if chord_length > 0 else "  Point Density: N/A")
            print()

def main():
    """Main function to demonstrate Bezier curve tessellation"""
    print("CUDA Bezier Curve Tessellation Demo")
    print("="*50)
    
    # Initialize tessellator
    tessellator = BezierTessellator()
    
    # Check CUDA devices
    device_count = tessellator.get_device_count()
    print(f"CUDA devices available: {device_count}")
    
    if device_count == 0:
        print("No CUDA devices found! Exiting.")
        return
    
    # Create sample curves
    curves = create_sample_curves()
    print(f"Created {len(curves)} sample Bezier curves")
    
    # Tessellate curves
    print("Tessellating curves using CUDA...")
    tessellated = tessellator.tessellate(curves)
    
    if not tessellated:
        print("Tessellation failed!")
        return
    
    print("Tessellation successful!")
    
    # Analyze results
    analyze_tessellation(curves, tessellated)
    
    # Visualize results
    print("Displaying visualization...")
    plot_curves(curves, tessellated)

if __name__ == "__main__":
    main()