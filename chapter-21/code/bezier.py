#!/usr/bin/env python3
import ctypes

import matplotlib.pyplot as plt
import numpy as np
import triton.testing


class BezierLineC(ctypes.Structure):
    _fields_ = [
        ("CP", ctypes.c_float * 6),
        ("vertexPos", ctypes.c_float * 64),
        ("nVertices", ctypes.c_int),
    ]


def simple_tessellate_dynamic(control_points_list):
    """
    Dynamic parallelism tessellation using CUDA dynamic parallelism

    Args:
        control_points_list: List of [(x0,y0), (x1,y1), (x2,y2)] tuples
    """
    lib = ctypes.CDLL("./libbezier.so")
    lib.tessellate_bezier_curves_dynamic.argtypes = [
        ctypes.POINTER(BezierLineC),
        ctypes.c_int,
    ]
    lib.tessellate_bezier_curves_dynamic.restype = ctypes.c_int

    # Setup curves
    n = len(control_points_list)
    curves = (BezierLineC * n)()

    for i, points in enumerate(control_points_list):
        for j, (x, y) in enumerate(points):
            curves[i].CP[j * 2] = x
            curves[i].CP[j * 2 + 1] = y

    # Tessellate using dynamic parallelism
    result = lib.tessellate_bezier_curves_dynamic(curves, n)
    if result != 0:
        print(f"Dynamic tessellation failed: {result}")
        return []

    # Extract results
    tessellated = []
    for i in range(n):
        nv = curves[i].nVertices
        if nv > 0:
            vertices = []
            for j in range(nv):
                x = curves[i].vertexPos[j * 2]
                y = curves[i].vertexPos[j * 2 + 1]
                vertices.append([x, y])
            tessellated.append(np.array(vertices))
        else:
            tessellated.append(np.array([]))

    return tessellated


def simple_tessellate_static(control_points_list):
    """
    Static parallelism tessellation (original version)

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
            curves[i].CP[j * 2] = x
            curves[i].CP[j * 2 + 1] = y

    # Tessellate using static parallelism
    result = lib.tessellate_bezier_curves(curves, n)
    if result != 0:
        print(f"Static tessellation failed: {result}")
        return []

    # Extract results
    tessellated = []
    for i in range(n):
        nv = curves[i].nVertices
        if nv > 0:
            vertices = []
            for j in range(nv):
                x = curves[i].vertexPos[j * 2]
                y = curves[i].vertexPos[j * 2 + 1]
                vertices.append([x, y])
            tessellated.append(np.array(vertices))
        else:
            tessellated.append(np.array([]))

    return tessellated


def plot_curves_comparison(
    control_points_list, tessellated_static, tessellated_dynamic
):
    """Compare static vs dynamic parallelism results"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(control_points_list)))

    # Plot 1: Control points and theoretical curves
    ax1.set_title("Bezier Curves - Control Points", fontsize=14, fontweight="bold")

    for i, (cp, color) in enumerate(zip(control_points_list, colors)):
        cp_array = np.array(cp)
        control_x = cp_array[:, 0]
        control_y = cp_array[:, 1]

        ax1.plot(
            control_x,
            control_y,
            "o--",
            color=color,
            alpha=0.7,
            linewidth=1,
            markersize=6,
            label=f"Curve {i+1} Control",
        )

        # Plot theoretical Bezier curve
        t = np.linspace(0, 1, 100)
        x_theo = (1 - t) ** 2 * cp[0][0] + 2 * (1 - t) * t * cp[1][0] + t**2 * cp[2][0]
        y_theo = (1 - t) ** 2 * cp[0][1] + 2 * (1 - t) * t * cp[1][1] + t**2 * cp[2][1]
        ax1.plot(x_theo, y_theo, "-", color=color, alpha=0.8, linewidth=2)

    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax1.set_aspect("equal")

    # Plot 2: Static parallelism results
    ax2.set_title("Static Parallelism Results", fontsize=14, fontweight="bold")

    for i, (cp, tess_points, color) in enumerate(
        zip(control_points_list, tessellated_static, colors)
    ):
        if len(tess_points) > 0:
            ax2.plot(
                tess_points[:, 0],
                tess_points[:, 1],
                "o-",
                color=color,
                markersize=4,
                linewidth=2,
                alpha=0.8,
                label=f"Curve {i+1} ({len(tess_points)} pts)",
            )

            ax2.plot(
                tess_points[0, 0],
                tess_points[0, 1],
                "s",
                color=color,
                markersize=8,
                markeredgecolor="black",
                markeredgewidth=1,
            )
            ax2.plot(
                tess_points[-1, 0],
                tess_points[-1, 1],
                "^",
                color=color,
                markersize=8,
                markeredgecolor="black",
                markeredgewidth=1,
            )

    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax2.set_aspect("equal")

    # Plot 3: Dynamic parallelism results
    ax3.set_title("Dynamic Parallelism Results", fontsize=14, fontweight="bold")

    for i, (cp, tess_points, color) in enumerate(
        zip(control_points_list, tessellated_dynamic, colors)
    ):
        if len(tess_points) > 0:
            ax3.plot(
                tess_points[:, 0],
                tess_points[:, 1],
                "o-",
                color=color,
                markersize=4,
                linewidth=2,
                alpha=0.8,
                label=f"Curve {i+1} ({len(tess_points)} pts)",
            )

            ax3.plot(
                tess_points[0, 0],
                tess_points[0, 1],
                "s",
                color=color,
                markersize=8,
                markeredgecolor="black",
                markeredgewidth=1,
            )
            ax3.plot(
                tess_points[-1, 0],
                tess_points[-1, 1],
                "^",
                color=color,
                markersize=8,
                markeredgecolor="black",
                markeredgewidth=1,
            )

    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax3.set_aspect("equal")

    plt.tight_layout()
    plt.show()
    plt.savefig("bezier_comparison_static_vs_dynamic.png")


def analyze_tessellation_comparison(
    control_points_list, tessellated_static, tessellated_dynamic
):
    """Analyze and compare static vs dynamic results"""
    print("\n" + "=" * 80)
    print("TESSELLATION COMPARISON: STATIC vs DYNAMIC PARALLELISM")
    print("=" * 80)

    for i, (cp, tess_static, tess_dynamic) in enumerate(
        zip(control_points_list, tessellated_static, tessellated_dynamic)
    ):
        if len(tess_static) > 0 and len(tess_dynamic) > 0:
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
            print(f"  Static Parallelism:  {len(tess_static)} points")
            print(f"  Dynamic Parallelism: {len(tess_dynamic)} points")

            # Check if results are identical
            if len(tess_static) == len(tess_dynamic):
                max_diff = (
                    np.max(np.abs(tess_static - tess_dynamic))
                    if len(tess_static) > 0
                    else 0
                )
                print(
                    f"  Max Difference: {max_diff:.8f}"
                    + (" (Identical)" if max_diff < 1e-6 else " (Different)")
                )
            else:
                print("  Result: Different number of points!")
            print()


def benchmark_performance(curves):
    """Simple performance benchmark using triton"""
    print("\n" + "=" * 60)
    print("PERFORMANCE BENCHMARK")
    print("=" * 60)

    # Test with different curve counts
    test_sizes = [100, 500, 1000]

    for size in test_sizes:
        print(f"\nBenchmarking with {size} curves:")

        # Create test curves
        test_curves = curves * (size // len(curves)) + curves[: size % len(curves)]

        # Benchmark static
        static_time = triton.testing.do_bench(
            lambda: simple_tessellate_static(test_curves), warmup=5, rep=25
        )

        # Benchmark dynamic
        dynamic_time = triton.testing.do_bench(
            lambda: simple_tessellate_dynamic(test_curves), warmup=5, rep=25
        )

        speedup = static_time / dynamic_time

        print(f"  Static:  {static_time:.3f} ms")
        print(f"  Dynamic: {dynamic_time:.3f} ms")
        print(
            f"  Speedup: {speedup:.2f}x {'(Dynamic faster)' if speedup > 1 else '(Static faster)'}"
        )


# Test both versions
if __name__ == "__main__":
    print("CUDA Bezier Curve Tessellation Demo - Static vs Dynamic Parallelism")
    print("=" * 75)

    # Create sample curves
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

    # Test static parallelism
    print("Tessellating curves using CUDA static parallelism...")
    tessellated_static = simple_tessellate_static(curves)

    # Test dynamic parallelism
    print("Tessellating curves using CUDA dynamic parallelism...")
    tessellated_dynamic = simple_tessellate_dynamic(curves)

    if not tessellated_static or not tessellated_dynamic:
        print("One or both tessellations failed!")
    else:
        print("Both tessellations successful!")

        # Analyze and compare results
        analyze_tessellation_comparison(curves, tessellated_static, tessellated_dynamic)

        # Simple performance benchmark
        benchmark_performance(curves)

        # Visualize comparison
        print("Displaying comparison visualization...")
        plot_curves_comparison(curves, tessellated_static, tessellated_dynamic)
