import ctypes
from ctypes import POINTER, c_float, c_int

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


class QuadtreeVisualizer:
    def __init__(self, cuda_lib_path="./quadtree.so"):
        """Initialize the quadtree visualizer with CUDA library"""
        try:
            # Load the compiled CUDA library
            self.lib = ctypes.CDLL(cuda_lib_path)

            # Define the function signature
            self.lib.build_quadtree.argtypes = [
                POINTER(c_float),  # h_x
                POINTER(c_float),  # h_y
                c_int,  # num_points
                c_int,  # max_depth
                c_int,  # min_points_per_node
                POINTER(POINTER(c_float)),  # result_x
                POINTER(POINTER(c_float)),  # result_y
                POINTER(c_float),  # bounds
                POINTER(c_int),  # num_result_points
            ]
            self.lib.build_quadtree.restype = c_int
            print("CUDA library loaded successfully!")

        except OSError:
            print(f"Could not load CUDA library at {cuda_lib_path}")
            print("Please compile the CUDA code first using:")
            print("nvcc -shared -fPIC -o quadtree.so quadtree.cu")
            self.lib = None

    def generate_sample_points(self, num_points=1000, pattern="random"):
        """Generate sample 2D points for testing"""
        np.random.seed(42)  # For reproducible results

        if pattern == "random":
            x = np.random.uniform(0, 10, num_points).astype(np.float32)
            y = np.random.uniform(0, 10, num_points).astype(np.float32)

        elif pattern == "clustered":
            # Create several clusters
            clusters = [
                (2, 2, 1.0),  # (center_x, center_y, std)
                (8, 8, 0.8),
                (3, 7, 0.6),
                (7, 3, 0.9),
            ]

            points_per_cluster = num_points // len(clusters)
            x_list, y_list = [], []

            for cx, cy, std in clusters:
                x_cluster = np.random.normal(cx, std, points_per_cluster)
                y_cluster = np.random.normal(cy, std, points_per_cluster)
                x_list.extend(x_cluster)
                y_list.extend(y_cluster)

            # Add remaining points randomly
            remaining = num_points - len(x_list)
            x_list.extend(np.random.uniform(0, 10, remaining))
            y_list.extend(np.random.uniform(0, 10, remaining))

            x = np.array(x_list, dtype=np.float32)
            y = np.array(y_list, dtype=np.float32)

        elif pattern == "grid":
            # Create a regular grid with some noise
            side = int(np.sqrt(num_points))
            x_grid = np.linspace(0.5, 9.5, side)
            y_grid = np.linspace(0.5, 9.5, side)
            xx, yy = np.meshgrid(x_grid, y_grid)

            # Add some random noise
            x = (xx.flatten() + np.random.normal(0, 0.2, xx.size))[:num_points].astype(
                np.float32
            )
            y = (yy.flatten() + np.random.normal(0, 0.2, yy.size))[:num_points].astype(
                np.float32
            )

        # Ensure points are within bounds
        x = np.clip(x, 0, 10)
        y = np.clip(y, 0, 10)

        return x, y

    def build_quadtree(self, x, y, max_depth=4, min_points_per_node=10):
        """Build quadtree using CUDA implementation"""
        if self.lib is None:
            print("CUDA library not available. Using mock data.")
            return x, y  # Return original points if CUDA not available

        num_points = len(x)

        # Convert to ctypes arrays
        x_array = (c_float * num_points)(*x)
        y_array = (c_float * num_points)(*y)

        # Set bounds [min_x, min_y, max_x, max_y]
        bounds = (c_float * 4)(0.0, 0.0, 10.0, 10.0)

        # Prepare output pointers
        result_x = POINTER(c_float)()
        result_y = POINTER(c_float)()
        num_result_points = c_int()

        # Call CUDA function
        result = self.lib.build_quadtree(
            x_array,
            y_array,
            num_points,
            max_depth,
            min_points_per_node,
            ctypes.byref(result_x),
            ctypes.byref(result_y),
            bounds,
            ctypes.byref(num_result_points),
        )

        if result != 0:
            print(f"Error in quadtree construction: {result}")
            return x, y

        # Convert results back to numpy arrays
        result_x_np = np.array([result_x[i] for i in range(num_result_points.value)])
        result_y_np = np.array([result_y[i] for i in range(num_result_points.value)])

        return result_x_np, result_y_np

    def visualize_quadtree_division(
        self,
        x_orig,
        y_orig,
        x_reordered,
        y_reordered,
        max_depth=4,
        min_points_per_node=10,
    ):
        """Visualize the original points and quadtree division"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        # Plot 1: Original points
        ax1.scatter(x_orig, y_orig, alpha=0.6, s=20, c="blue")
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 10)
        ax1.set_title(f"Original Points ({len(x_orig)} points)")
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")

        # Plot 2: Reordered points with color gradient
        ax2.scatter(
            x_reordered,
            y_reordered,
            alpha=0.7,
            s=20,
            c=range(len(x_reordered)),
            cmap="viridis",
        )
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 10)
        ax2.set_title("Reordered Points (Quadtree Order)")
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")

        # Plot 3: Simulated quadtree boundaries
        ax3.scatter(x_orig, y_orig, alpha=0.6, s=20, c="blue")
        self._draw_quadtree_boundaries(
            ax3, x_orig, y_orig, 0, 0, 10, 10, max_depth, min_points_per_node, 0
        )
        ax3.set_xlim(0, 10)
        ax3.set_ylim(0, 10)
        ax3.set_title("Estimated Quadtree Boundaries")
        ax3.grid(True, alpha=0.3)
        ax3.set_xlabel("X")
        ax3.set_ylabel("Y")

        plt.tight_layout()
        plt.show()
        plt.savefig("quadtree.png")

    def _draw_quadtree_boundaries(
        self,
        ax,
        x,
        y,
        min_x,
        min_y,
        max_x,
        max_y,
        max_depth,
        min_points_per_node,
        current_depth,
    ):
        """Recursively draw quadtree boundaries (approximation)"""
        # Count points in current region
        mask = (x >= min_x) & (x < max_x) & (y >= min_y) & (y < max_y)
        points_in_region = np.sum(mask)

        # Stop if we've reached max depth or have too few points
        if current_depth >= max_depth or points_in_region <= min_points_per_node:
            # Draw boundary
            rect = patches.Rectangle(
                (min_x, min_y),
                max_x - min_x,
                max_y - min_y,
                linewidth=1,
                edgecolor="red",
                facecolor="none",
                alpha=0.7,
            )
            ax.add_patch(rect)
            return

        # Calculate center
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

        # Draw center lines
        ax.axvline(
            x=center_x,
            ymin=(min_y / 10),
            ymax=(max_y / 10),
            color="red",
            alpha=0.5,
            linewidth=0.8,
        )
        ax.axhline(
            y=center_y,
            xmin=(min_x / 10),
            xmax=(max_x / 10),
            color="red",
            alpha=0.5,
            linewidth=0.8,
        )

        # Recursively divide into 4 quadrants
        # Top-left
        self._draw_quadtree_boundaries(
            ax,
            x,
            y,
            min_x,
            center_y,
            center_x,
            max_y,
            max_depth,
            min_points_per_node,
            current_depth + 1,
        )
        # Top-right
        self._draw_quadtree_boundaries(
            ax,
            x,
            y,
            center_x,
            center_y,
            max_x,
            max_y,
            max_depth,
            min_points_per_node,
            current_depth + 1,
        )
        # Bottom-left
        self._draw_quadtree_boundaries(
            ax,
            x,
            y,
            min_x,
            min_y,
            center_x,
            center_y,
            max_depth,
            min_points_per_node,
            current_depth + 1,
        )
        # Bottom-right
        self._draw_quadtree_boundaries(
            ax,
            x,
            y,
            center_x,
            min_y,
            max_x,
            center_y,
            max_depth,
            min_points_per_node,
            current_depth + 1,
        )


def main():
    """Main demonstration function"""
    print("=== CUDA Quadtree Demonstration ===\n")

    # Initialize visualizer
    visualizer = QuadtreeVisualizer()

    # Parameters
    num_points = 500
    max_depth = 4
    min_points_per_node = 8

    print("Parameters:")
    print(f"  Number of points: {num_points}")
    print(f"  Max depth: {max_depth}")
    print(f"  Min points per node: {min_points_per_node}\n")

    # Test different point patterns
    patterns = ["random", "clustered", "grid"]

    for pattern in patterns:
        print(f"Testing with {pattern} point pattern...")

        # Generate points
        x_orig, y_orig = visualizer.generate_sample_points(num_points, pattern)

        # Build quadtree
        print("Building quadtree...")
        x_reordered, y_reordered = visualizer.build_quadtree(
            x_orig, y_orig, max_depth, min_points_per_node
        )

        # Visualize
        print("Creating visualization...")
        visualizer.visualize_quadtree_division(
            x_orig, y_orig, x_reordered, y_reordered, max_depth, min_points_per_node
        )

        # Show statistics
        print(
            f"  Original points range: X[{x_orig.min():.2f}, {x_orig.max():.2f}], "
            f"Y[{y_orig.min():.2f}, {y_orig.max():.2f}]"
        )
        print(f"  Points successfully reordered: {len(x_reordered)}/{len(x_orig)}\n")


if __name__ == "__main__":
    # Print compilation instructions
    print("To compile the CUDA code, run:")
    print("nvcc -shared -fPIC -o quadtree.so quadtree.cu")
    print("Make sure you have CUDA toolkit installed.\n")

    main()
