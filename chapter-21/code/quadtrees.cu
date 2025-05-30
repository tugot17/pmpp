__global__ void build_quadtree_kernel
       (Quadtree_node *nodes, Points *points, Parameters params) {
   __shared__ int smem[8]; // To store the number of points in each quadrant
   
   // The current node in the quadtree
   Quadtree_node &node = nodes[blockIdx.x];
   node.set_id(node.id() + blockIdx.x);
   int num_points = node.num_points(); // The number of points in the node
   
   // Check the number of points and its depth
   bool exit = check_num_points_and_depth(node, points, num_points, params);
   if(exit) return;
   
   // Compute the center of the bounding box of the points
   const Bounding_box &bbox = node.bounding_box();
   float2 center;
   bbox.compute_center(center);
   
   // Range of points
   int range_begin = node.points_begin();
   int range_end   = node.points_end();
   const Points &in_points = points[params.point_selector];       // Input points
   Points &out_points = points[(params.point_selector+1) % 2];    // Output points
   
   // Count the number of points in each child
   count_points_in_children(in_points, smem, range_begin, range_end, center);
   
   // Scan the quadrants' results to know the reordering offset
   scan_for_offsets(node.points_begin(), smem);
   
   // Move points
   reorder_points(out_points, in_points, smem, range_begin, range_end, center);
   
   // Launch new blocks
   if (threadIdx.x == blockDim.x-1) {
       // The children
       Quadtree_node *children = &nodes[params.num_nodes_at_this_level];
       
       // Prepare children launch
       prepare_children(children, node, bbox, smem);
       
       // Launch 4 children.
       build_quadtree_kernel<<<4, blockDim.x, 8 *sizeof(int)>>>
               (children, points, Parameters(params, true));
   }
}