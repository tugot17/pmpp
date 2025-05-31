#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Forward declarations of classes
class Points;
class Bounding_box;
class Quadtree_node;
struct Parameters;

// A structure of 2D points
class Points {
    float* m_x;
    float* m_y;

public:
    // Constructor
    __host__ __device__ Points() : m_x(NULL), m_y(NULL) {}

    // Constructor
    __host__ __device__ Points(float* x, float* y) : m_x(x), m_y(y) {}

    // Get a point
    __host__ __device__ __forceinline__ float2 get_point(int idx) const {
        return make_float2(m_x[idx], m_y[idx]);
    }

    // Set a point
    __host__ __device__ __forceinline__ void set_point(int idx, const float2& p) {
        m_x[idx] = p.x;
        m_y[idx] = p.y;
    }

    // Set the pointers
    __host__ __device__ __forceinline__ void set(float* x, float* y) {
        m_x = x;
        m_y = y;
    }
};

// A 2D bounding box
class Bounding_box {
    // The two points of the bounding box
    float2 m_p_min;
    float2 m_p_max;

public:
    // Constructor. Create a unit box
    __host__ __device__ Bounding_box() {
        m_p_min = make_float2(0.0f, 0.0f);
        m_p_max = make_float2(1.0f, 1.0f);
    }

    // Compute the center of the bounding box
    __host__ __device__ void compute_center(float2* center) const {
        center->x = 0.5f * (m_p_min.x + m_p_max.x);
        center->y = 0.5f * (m_p_min.y + m_p_max.y);
    }

    // The points of the box
    __host__ __device__ __forceinline__ const float2& get_max() const {
        return m_p_max;
    }

    __host__ __device__ __forceinline__ const float2& get_min() const {
        return m_p_min;
    }

    // Does a box contain a point
    __host__ __device__ bool contains(const float2& p) const {
        return p.x >= m_p_min.x && p.x < m_p_max.x && p.y >= m_p_min.y && p.y < m_p_max.y;
    }

    // Define the bounding box
    __host__ __device__ void set(float min_x, float min_y, float max_x, float max_y) {
        m_p_min.x = min_x;
        m_p_min.y = min_y;
        m_p_max.x = max_x;
        m_p_max.y = max_y;
    }
};

// A node of a quadtree
class Quadtree_node {
    // The identifier of the node
    int m_id;
    // The bounding box of the tree
    Bounding_box m_bounding_box;
    // The range of points
    int m_begin, m_end;

public:
    // Constructor
    __host__ __device__ Quadtree_node() : m_id(0), m_begin(0), m_end(0) {}

    // The ID of the node (at its level)
    __host__ __device__ int id() const {
        return m_id;
    }

    // The ID of a node at its level
    __host__ __device__ void set_id(int new_id) {
        m_id = new_id;
    }

    // The bounding box
    __host__ __device__ __forceinline__ const Bounding_box& bounding_box() const {
        return m_bounding_box;
    }

    // Set the bounding box
    __host__ __device__ __forceinline__ void set_bounding_box(float min_x, float min_y, float max_x, float max_y) {
        m_bounding_box.set(min_x, min_y, max_x, max_y);
    }

    // The number of points in the tree
    __host__ __device__ __forceinline__ int num_points() const {
        return m_end - m_begin;
    }

    // The range of points in the tree
    __host__ __device__ __forceinline__ int points_begin() const {
        return m_begin;
    }

    __host__ __device__ __forceinline__ int points_end() const {
        return m_end;
    }

    // Define the range for that node
    __host__ __device__ __forceinline__ void set_range(int begin, int end) {
        m_begin = begin;
        m_end = end;
    }
};

// Algorithm parameters
struct Parameters {
    // Choose the right set of points to use as in/out
    int point_selector;
    // The number of nodes at a given level (2^k for level k)
    int num_nodes_at_this_level;
    // The recursion depth
    int depth;
    // The max value for depth
    const int max_depth;
    // The minimum number of points in a node to stop recursion
    const int min_points_per_node;

    // Constructor set to default values
    __host__ __device__ Parameters(int max_depth, int min_points_per_node)
        : point_selector(0),
          num_nodes_at_this_level(1),
          depth(0),
          max_depth(max_depth),
          min_points_per_node(min_points_per_node) {}

    // Copy constructor. Changes the values for next iteration
    __host__ __device__ Parameters(const Parameters& params, bool)
        : point_selector((params.point_selector + 1) % 2),
          num_nodes_at_this_level(4 * params.num_nodes_at_this_level),
          depth(params.depth + 1),
          max_depth(params.max_depth),
          min_points_per_node(params.min_points_per_node) {}
};

// Check the number of points and its depth
__device__ bool check_num_points_and_depth(Quadtree_node* node, Points* points, int num_points, Parameters params) {
    if (params.depth >= params.max_depth || num_points <= params.min_points_per_node) {
        // Stop the recursion here. Make sure points[0] contains all the points
        if (params.point_selector == 1) {
            int it = node->points_begin(), end = node->points_end();
            for (it += threadIdx.x; it < end; it += blockDim.x) {
                if (it < end) {
                    points[0].set_point(it, points[1].get_point(it));
                }
            }
        }
        return true;
    }
    return false;
}

// Count the number of points in each quadrant
__device__ void count_points_in_children(const Points& in_points, int* smem, int range_begin, int range_end,
                                         float2 center) {
    // Initialize shared memory
    if (threadIdx.x < 4) {
        smem[threadIdx.x] = 0;
    }
    __syncthreads();
    // Compute the number of points
    for (int iter = range_begin + threadIdx.x; iter < range_end; iter += blockDim.x) {
        float2 p = in_points.get_point(iter);  // Load the coordinates of the point
        if (p.x < center.x && p.y >= center.y) {
            atomicAdd(&smem[0], 1);  // Top-left point?
        }
        if (p.x >= center.x && p.y >= center.y) {
            atomicAdd(&smem[1], 1);  // Top-right point?
        }
        if (p.x < center.x && p.y < center.y) {
            atomicAdd(&smem[2], 1);  // Bottom-left point?
        }
        if (p.x >= center.x && p.y < center.y) {
            atomicAdd(&smem[3], 1);  // Bottom-right point?
        }
    }
    __syncthreads();
}

// Scan quadrants' results to obtain reordering offset
__device__ void scan_for_offsets(int node_points_begin, int* smem) {
    int* smem2 = &smem[4];
    if (threadIdx.x == 0) {
        // smem2 will contain starting positions for writing each quadrant
        smem2[0] = node_points_begin;   // Top-left starts at begin
        smem2[1] = smem2[0] + smem[0];  // Top-right starts after top-left
        smem2[2] = smem2[1] + smem[1];  // Bottom-left starts after top-right
        smem2[3] = smem2[2] + smem[2];  // Bottom-right starts after bottom-left
    }
    __syncthreads();
}

// Reorder points in order to group the points in each quadrant
__device__ void reorder_points(Points* out_points, const Points& in_points, int* smem, int range_begin, int range_end,
                               float2 center) {
    int* smem2 = &smem[4];
    // Reorder points
    for (int iter = range_begin + threadIdx.x; iter < range_end; iter += blockDim.x) {
        float2 p = in_points.get_point(iter);  // Load the coordinates of the point
        int dest = -1;

        // Determine which quadrant the point belongs to
        if (p.x < center.x && p.y >= center.y) {
            dest = atomicAdd(&smem2[0], 1);  // Top-left point
        } else if (p.x >= center.x && p.y >= center.y) {
            dest = atomicAdd(&smem2[1], 1);  // Top-right point
        } else if (p.x < center.x && p.y < center.y) {
            dest = atomicAdd(&smem2[2], 1);  // Bottom-left point
        } else if (p.x >= center.x && p.y < center.y) {
            dest = atomicAdd(&smem2[3], 1);  // Bottom-right point
        }

        // Move point to its destination
        if (dest >= 0) {
            out_points->set_point(dest, p);
        }
    }
    __syncthreads();
}

// Prepare children launch
__device__ void prepare_children(Quadtree_node* children, Quadtree_node* node, const Bounding_box& bbox, int* smem) {
    if (threadIdx.x == 0) {
        // Points to the bounding-box
        const float2& p_min = bbox.get_min();
        const float2& p_max = bbox.get_max();

        // Compute center for children bounding boxes
        float2 center;
        bbox.compute_center(&center);

        int* smem2 = &smem[4];  // Starting positions for each quadrant

        // Set up the 4 children only if they have points
        for (int i = 0; i < 4; i++) {
            if (smem[i] > 0) {  // Only set up children that have points
                children[i].set_id(i);
                children[i].set_range(smem2[i], smem2[i] + smem[i]);

                // Set bounding boxes based on quadrant
                if (i == 0) {  // Top-left
                    children[i].set_bounding_box(p_min.x, center.y, center.x, p_max.y);
                } else if (i == 1) {  // Top-right
                    children[i].set_bounding_box(center.x, center.y, p_max.x, p_max.y);
                } else if (i == 2) {  // Bottom-left
                    children[i].set_bounding_box(p_min.x, p_min.y, center.x, center.y);
                } else {  // Bottom-right
                    children[i].set_bounding_box(center.x, p_min.y, p_max.x, center.y);
                }
            }
        }
    }
    __syncthreads();
}

__global__ void build_quadtree_kernel(Quadtree_node* nodes, Points* points, Parameters params) {
    __shared__ int smem[8];  // To store the number of points in each quadrant

    // The current node in the quadtree
    Quadtree_node* node = &nodes[blockIdx.x];

    int num_points = node->num_points();  // The number of points in the node

    // Check the number of points and its depth
    bool exit = check_num_points_and_depth(node, points, num_points, params);
    if (exit) {
        return;
    }

    // Compute the center of the bounding box of the points
    const Bounding_box& bbox = node->bounding_box();
    float2 center;
    bbox.compute_center(&center);

    // Range of points
    int range_begin = node->points_begin();
    int range_end = node->points_end();
    const Points& in_points = points[params.point_selector];        // Input points
    Points* out_points = &points[(params.point_selector + 1) % 2];  // Output points

    // Count the number of points in each child
    count_points_in_children(in_points, smem, range_begin, range_end, center);

    // Scan the quadrants' results to know the reordering offset
    scan_for_offsets(node->points_begin(), smem);

    // Move points
    reorder_points(out_points, in_points, smem, range_begin, range_end, center);

    // Launch new blocks for children
    if (threadIdx.x == 0) {
        // Check if any child has enough points to subdivide
        bool should_recurse = false;
        for (int i = 0; i < 4; i++) {
            if (smem[i] > params.min_points_per_node && params.depth + 1 < params.max_depth) {
                should_recurse = true;
                break;
            }
        }

        if (should_recurse) {
            // Allocate space for 4 children
            Quadtree_node* children = &nodes[params.num_nodes_at_this_level + blockIdx.x * 4];

            // Prepare children
            prepare_children(children, node, bbox, smem);

            // Launch kernel for each child that has points
            Parameters next_params(params, true);
            for (int i = 0; i < 4; i++) {
                if (smem[i] > 0) {
                    build_quadtree_kernel<<<1, blockDim.x, 8 * sizeof(int)>>>(&children[i], points, next_params);
                }
            }
        }
    }
}

// Host wrapper function
extern "C" {
int build_quadtree(float* h_x, float* h_y, int num_points, int max_depth, int min_points_per_node, float** result_x,
                   float** result_y, float* bounds, int* num_result_points) {
    // Allocate device memory for points
    float *d_x[2], *d_y[2];
    cudaMalloc(&d_x[0], num_points * sizeof(float));
    cudaMalloc(&d_y[0], num_points * sizeof(float));
    cudaMalloc(&d_x[1], num_points * sizeof(float));
    cudaMalloc(&d_y[1], num_points * sizeof(float));

    // Copy input points to device
    cudaMemcpy(d_x[0], h_x, num_points * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y[0], h_y, num_points * sizeof(float), cudaMemcpyHostToDevice);

    // Create Points objects
    Points h_points[2];
    h_points[0].set(d_x[0], d_y[0]);
    h_points[1].set(d_x[1], d_y[1]);

    Points* d_points;
    cudaMalloc(&d_points, 2 * sizeof(Points));
    cudaMemcpy(d_points, h_points, 2 * sizeof(Points), cudaMemcpyHostToDevice);

    // Calculate maximum number of nodes needed (conservative estimate)
    int max_nodes = 1;
    for (int i = 1; i <= max_depth; i++) {
        max_nodes += (int)pow(4, i);
    }
    max_nodes *= 2;  // Extra safety margin

    // Allocate device memory for nodes
    Quadtree_node* d_nodes;
    cudaMalloc(&d_nodes, max_nodes * sizeof(Quadtree_node));
    cudaMemset(d_nodes, 0, max_nodes * sizeof(Quadtree_node));

    // Initialize root node
    Quadtree_node root;
    root.set_id(0);
    root.set_range(0, num_points);
    root.set_bounding_box(bounds[0], bounds[1], bounds[2], bounds[3]);
    cudaMemcpy(d_nodes, &root, sizeof(Quadtree_node), cudaMemcpyHostToDevice);

    // Create parameters
    Parameters params(max_depth, min_points_per_node);

    // Launch kernel with single block for root
    build_quadtree_kernel<<<1, 32>>>(d_nodes, d_points, params);

    // Wait for completion and check for errors
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Copy result back (final points are in buffer 0)
    *result_x = (float*)malloc(num_points * sizeof(float));
    *result_y = (float*)malloc(num_points * sizeof(float));

    cudaMemcpy(*result_x, d_x[0], num_points * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(*result_y, d_y[0], num_points * sizeof(float), cudaMemcpyDeviceToHost);

    *num_result_points = num_points;

    // Cleanup
    cudaFree(d_x[0]);
    cudaFree(d_y[0]);
    cudaFree(d_x[1]);
    cudaFree(d_y[1]);
    cudaFree(d_points);
    cudaFree(d_nodes);

    return 0;
}
}
