#!/usr/bin/env python3
import ctypes
import numpy as np
import matplotlib.pyplot as plt

class BezierLineC(ctypes.Structure):
    _fields_ = [
        ("CP", ctypes.c_float * 6),        
        ("vertexPos", ctypes.c_float * 64), 
        ("nVertices", ctypes.c_int)
    ]

def simple_tessellate(control_points_list):
    """
    Simple tessellation using your existing library
    
    Args:
        control_points_list: List of [(x0,y0), (x1,y1), (x2,y2)] tuples
    """
    lib = ctypes.CDLL("./libbezier.so")
    lib.tessellate_bezier_curves.argtypes = [ctypes.POINTER(BezierLineC), ctypes.c_int]
    lib.tessellate_bezier_curves.restype = ctypes.c_int
    
    # Setup curves
    n = len(control_points_list)
    curves = (BezierLineC * n)()
    
    for i, points in enumerate(control_points_list):
        for j, (x, y) in enumerate(points):
            curves[i].CP[j*2] = x
            curves[i].CP[j*2+1] = y
    
    # Tessellate
    result = lib.tessellate_bezier_curves(curves, n)
    if result != 0:
        print(f"Tessellation failed: {result}")
        return []
    
    # Extract results
    tessellated = []
    for i in range(n):
        nv = curves[i].nVertices
        if nv > 0:
            vertices = []
            for j in range(nv):
                x = curves[i].vertexPos[j*2]
                y = curves[i].vertexPos[j*2+1]
                vertices.append([x, y])
            tessellated.append(np.array(vertices))
        else:
            tessellated.append(np.array([]))
    
    return tessellated

def plot_curves(control_points_list, tessellated):
    """Visualize the original control points and tessellated curves (original style)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Colors for different curves
    colors = plt.cm.tab10(np.linspace(0, 1, len(control_points_list)))
    
    # Plot 1: Control points and theoretical curves
    ax1.set_title("Bezier Curves - Control Points", fontsize=14, fontweight='bold')
    
    for i, (cp, color) in enumerate(zip(control_points_list, colors)):
        # Plot control points
        cp_array = np.array(cp)
        control_x = cp_array[:, 0]
        control_y = cp_array[:, 1]
        
        ax1.plot(control_x, control_y, 'o--', color=color, alpha=0.7, 
                linewidth=1, markersize=6, label=f'Curve {i+1} Control')
        
        # Plot theoretical Bezier curve (for reference)
        t = np.linspace(0, 1, 100)
        x_theo = (1-t)**2 * cp[0][0] + 2*(1-t)*t * cp[1][0] + t**2 * cp[2][0]
        y_theo = (1-t)**2 * cp[0][1] + 2*(1-t)*t * cp[1][1] + t**2 * cp[2][1]
        ax1.plot(x_theo, y_theo, '-', color=color, alpha=0.8, linewidth=2)
    
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.set_aspect('equal')
    
    # Plot 2: CUDA tessellated curves
    ax2.set_title("CUDA Tessellated Curves", fontsize=14, fontweight='bold')
    
    for i, (cp, tess_points, color) in enumerate(zip(control_points_list, tessellated, colors)):
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

def analyze_tessellation(control_points_list, tessellated):
    """Analyze the tessellation results (original style)"""
    print("\n" + "="*60)
    print("TESSELLATION ANALYSIS")
    print("="*60)
    
    for i, (cp, tess_points) in enumerate(zip(control_points_list, tessellated)):
        if len(tess_points) > 0:
            # Calculate curvature estimate
            p0, p1, p2 = np.array(cp[0]), np.array(cp[1]), np.array(cp[2])
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
            print(f"  Control Points: {cp[0]} -> {cp[1]} -> {cp[2]}")
            print(f"  Estimated Curvature: {curvature:.4f}")
            print(f"  Tessellation Points: {len(tess_points)}")
            print(f"  Point Density: {len(tess_points)/chord_length:.2f} pts/unit" if chord_length > 0 else "  Point Density: N/A")
            print()

# Test it
if __name__ == "__main__":
    print("CUDA Bezier Curve Tessellation Demo")
    print("="*50)
    
    # Create sample curves (like original)
    curves = [
        # Simple arc
        [(0.0, 0.0), (0.5, 1.0), (1.0, 0.0)],
        
        # S-curve
        [(0.0, 1.0), (0.8, 1.5), (1.0, 2.0)],
        
        # Sharp turn
        [(1.0, 2.0), (1.8, 1.2), (2.0, 2.0)],
        
        # Loop-like curve
        [(2.0, 0.0), (3.5, 1.5), (2.5, 0.5)],
        
        # Nearly straight line (low curvature)
        [(0.0, 3.0), (1.0, 3.1), (2.0, 3.0)],
        
        # High curvature curve
        [(2.5, 1.0), (4.0, 0.0), (2.5, 2.0)],
    ]
    
    print(f"Created {len(curves)} sample Bezier curves")
    print("Tessellating curves using CUDA...")
    
    tessellated = simple_tessellate(curves)
    
    if not tessellated:
        print("Tessellation failed!")
    else:
        print("Tessellation successful!")
        
        # Analyze results
        analyze_tessellation(curves, tessellated)
        
        # Visualize results
        print("Displaying visualization...")
        plot_curves(curves, tessellated)