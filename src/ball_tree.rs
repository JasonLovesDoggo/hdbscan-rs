//! Ball tree for dual-tree Boruvka MST in medium-to-high dimensions.
//!
//! Unlike kd-trees which use axis-aligned bounding boxes, ball trees bound
//! each node with a hypersphere (centroid + radius). This provides much
//! tighter distance bounds in dimensions > ~8, where kd-tree bounding box
//! pruning degrades due to the curse of dimensionality.

use ndarray::ArrayView2;

/// Default leaf size for ball tree. Larger leaves reduce tree overhead
/// but give coarser spatial partitioning.
pub const LEAF_SIZE: usize = 40;

pub const NO_CHILD: usize = usize::MAX;

/// A ball tree node. Each node represents a hypersphere containing all
/// its descendant points.
#[derive(Clone)]
pub struct BallNode {
    /// Centroid of the bounding ball
    pub centroid: Vec<f64>,
    /// Squared radius of the bounding ball (max squared dist from centroid to any point)
    pub radius_sq: f64,
    /// Radius of the bounding ball
    pub radius: f64,
    /// Left child index (or NO_CHILD if leaf)
    pub left: usize,
    /// Right child index (or NO_CHILD if leaf)
    pub right: usize,
    /// Number of points in this subtree
    pub count: usize,
    /// Start index in the sorted_indices array
    pub idx_start: usize,
    /// End index (exclusive) in the sorted_indices array
    pub idx_end: usize,
    /// Whether this node is a leaf
    pub is_leaf: bool,
}

/// Ball tree for dual-tree algorithms.
/// Points are stored only in leaves; internal nodes contain bounding ball metadata.
pub struct BallTree {
    pub nodes: Vec<BallNode>,
    /// The raw data, stored as a flat [n * dim] array for cache efficiency.
    pub data: Vec<f64>,
    pub dim: usize,
    pub n: usize,
    /// Sorted point indices (each leaf's points are contiguous in this array)
    pub sorted_indices: Vec<usize>,
}

impl BallTree {
    /// Build a ball tree from an n x d data matrix.
    pub fn build(data: &ArrayView2<f64>) -> Self {
        let n = data.nrows();
        let dim = data.ncols();

        let data_contiguous = data.as_standard_layout();
        let flat_data: Vec<f64> = data_contiguous.as_slice().unwrap().to_vec();

        let mut sorted_indices: Vec<usize> = (0..n).collect();
        let mut nodes = Vec::with_capacity(2 * n / LEAF_SIZE + 1);

        if n > 0 {
            Self::build_recursive(&flat_data, &mut sorted_indices, 0, n, dim, &mut nodes);
        }

        BallTree {
            nodes,
            data: flat_data,
            dim,
            n,
            sorted_indices,
        }
    }

    fn build_recursive(
        data: &[f64],
        indices: &mut [usize],
        start: usize,
        end: usize,
        dim: usize,
        nodes: &mut Vec<BallNode>,
    ) -> usize {
        if start >= end {
            return NO_CHILD;
        }

        let count = end - start;

        // Compute centroid
        let mut centroid = vec![0.0f64; dim];
        for &idx in &indices[start..end] {
            let base = idx * dim;
            for d in 0..dim {
                centroid[d] += data[base + d];
            }
        }
        let inv_count = 1.0 / count as f64;
        for d in 0..dim {
            centroid[d] *= inv_count;
        }

        // Compute radius (max distance from centroid to any point)
        let mut radius_sq = 0.0f64;
        for &idx in &indices[start..end] {
            let base = idx * dim;
            let mut dist_sq = 0.0f64;
            for d in 0..dim {
                let diff = data[base + d] - centroid[d];
                dist_sq += diff * diff;
            }
            if dist_sq > radius_sq {
                radius_sq = dist_sq;
            }
        }
        let radius = radius_sq.sqrt();

        // Leaf node
        if count <= LEAF_SIZE {
            let node_idx = nodes.len();
            nodes.push(BallNode {
                centroid,
                radius_sq,
                radius,
                left: NO_CHILD,
                right: NO_CHILD,
                count,
                idx_start: start,
                idx_end: end,
                is_leaf: true,
            });
            return node_idx;
        }

        // Split: find the two points farthest apart to define the split direction.
        // First find the point farthest from the centroid.
        let mut farthest_a = indices[start];
        let mut max_dist_sq = 0.0f64;
        for &idx in &indices[start..end] {
            let base = idx * dim;
            let mut dist_sq = 0.0f64;
            for d in 0..dim {
                let diff = data[base + d] - centroid[d];
                dist_sq += diff * diff;
            }
            if dist_sq > max_dist_sq {
                max_dist_sq = dist_sq;
                farthest_a = idx;
            }
        }

        // Then find the point farthest from farthest_a.
        let mut farthest_b = farthest_a;
        max_dist_sq = 0.0;
        let base_a = farthest_a * dim;
        for &idx in &indices[start..end] {
            let base = idx * dim;
            let mut dist_sq = 0.0f64;
            for d in 0..dim {
                let diff = data[base + d] - data[base_a + d];
                dist_sq += diff * diff;
            }
            if dist_sq > max_dist_sq {
                max_dist_sq = dist_sq;
                farthest_b = idx;
            }
        }

        // Project all points onto the line between farthest_a and farthest_b,
        // partition around the median projection.
        let base_b = farthest_b * dim;
        let mut projections: Vec<f64> = Vec::with_capacity(count);
        for &idx in &indices[start..end] {
            let base = idx * dim;
            let mut proj = 0.0f64;
            for d in 0..dim {
                let dir = data[base_b + d] - data[base_a + d];
                let offset = data[base + d] - data[base_a + d];
                proj += dir * offset;
            }
            projections.push(proj);
        }

        // Partition around median using select_nth_unstable
        let mid = start + count / 2;
        let proj_slice = &mut indices[start..end];
        proj_slice.select_nth_unstable_by(mid - start, |&a_idx, &b_idx| {
            // Find positions of a_idx and b_idx in the original indices to get projections
            // This is tricky because select_nth_unstable moves elements around.
            // Instead, recompute projections inline.
            let base_ai = a_idx * dim;
            let base_bi = b_idx * dim;
            let mut proj_a = 0.0f64;
            let mut proj_b = 0.0f64;
            for d in 0..dim {
                let dir = data[base_b + d] - data[base_a + d];
                proj_a += dir * (data[base_ai + d] - data[base_a + d]);
                proj_b += dir * (data[base_bi + d] - data[base_a + d]);
            }
            proj_a
                .partial_cmp(&proj_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let node_idx = nodes.len();
        nodes.push(BallNode {
            centroid,
            radius_sq,
            radius,
            left: NO_CHILD,
            right: NO_CHILD,
            count,
            idx_start: start,
            idx_end: end,
            is_leaf: false,
        });

        let left = Self::build_recursive(data, indices, start, mid, dim, nodes);
        let right = Self::build_recursive(data, indices, mid, end, dim, nodes);

        nodes[node_idx].left = left;
        nodes[node_idx].right = right;

        node_idx
    }

    /// Minimum possible squared distance between two ball tree nodes.
    #[inline]
    pub fn min_dist_sq_node_to_node(&self, node_a: usize, node_b: usize) -> f64 {
        let a = &self.nodes[node_a];
        let b = &self.nodes[node_b];

        let centroid_dist_sq =
            crate::simd_distance::squared_euclidean_simd(&a.centroid, &b.centroid);
        let centroid_dist = centroid_dist_sq.sqrt();
        let gap = centroid_dist - a.radius - b.radius;

        if gap <= 0.0 {
            0.0
        } else {
            gap * gap
        }
    }

    /// Compute squared Euclidean distance between two points.
    #[inline]
    pub fn dist_sq(&self, i: usize, j: usize) -> f64 {
        crate::simd_distance::squared_euclidean_flat(&self.data, i, j, self.dim)
    }

    /// Get the point indices for a node's subtree.
    #[inline]
    pub fn node_points(&self, node_idx: usize) -> &[usize] {
        let node = &self.nodes[node_idx];
        &self.sorted_indices[node.idx_start..node.idx_end]
    }

    /// Find k nearest neighbors of query point. Returns (distance, index) sorted by distance.
    pub fn query_knn(&self, query: &[f64], k: usize) -> Vec<(f64, usize)> {
        if self.nodes.is_empty() || k == 0 {
            return vec![];
        }
        let mut heap = crate::knn_heap::KnnHeap::new(k);
        // Cache sqrt(max_dist) to avoid sqrt in the pruning hot path.
        let mut sqrt_max_dist = f64::INFINITY;
        let root_centroid_dist_sq =
            crate::simd_distance::squared_euclidean_simd(query, &self.nodes[0].centroid);
        self.knn_recursive(0, query, &mut heap, &mut sqrt_max_dist, root_centroid_dist_sq);
        heap.into_sorted_distances()
    }

    fn knn_recursive(
        &self,
        node_idx: usize,
        query: &[f64],
        heap: &mut crate::knn_heap::KnnHeap,
        sqrt_max_dist: &mut f64,
        centroid_dist_sq: f64,
    ) {
        let node = &self.nodes[node_idx];

        // Pruning: check if minimum possible distance from query to this ball
        // can beat the current k-th best. Uses cached sqrt(max_dist) to avoid sqrt.
        if heap.is_full() {
            let bound = *sqrt_max_dist + node.radius;
            if centroid_dist_sq >= bound * bound {
                return;
            }
        }

        if node.is_leaf {
            let old_max = heap.max_dist_sq();
            for &idx in &self.sorted_indices[node.idx_start..node.idx_end] {
                let dist_sq = crate::simd_distance::squared_euclidean_simd(
                    query,
                    &self.data[idx * self.dim..(idx + 1) * self.dim],
                );
                heap.push(dist_sq, idx);
            }
            let new_max = heap.max_dist_sq();
            if new_max != old_max {
                *sqrt_max_dist = new_max.sqrt();
            }
        } else {
            let left = node.left;
            let right = node.right;

            // Compute centroid distances to children — passed down to avoid recomputation
            let left_centroid_dist_sq = if left != NO_CHILD {
                crate::simd_distance::squared_euclidean_simd(query, &self.nodes[left].centroid)
            } else {
                f64::INFINITY
            };

            let right_centroid_dist_sq = if right != NO_CHILD {
                crate::simd_distance::squared_euclidean_simd(query, &self.nodes[right].centroid)
            } else {
                f64::INFINITY
            };

            let (first, first_dist, second, second_dist) =
                if left_centroid_dist_sq <= right_centroid_dist_sq {
                    (left, left_centroid_dist_sq, right, right_centroid_dist_sq)
                } else {
                    (right, right_centroid_dist_sq, left, left_centroid_dist_sq)
                };

            if first != NO_CHILD {
                self.knn_recursive(first, query, heap, sqrt_max_dist, first_dist);
            }
            if second != NO_CHILD {
                self.knn_recursive(second, query, heap, sqrt_max_dist, second_dist);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_ball_tree_build() {
        let data = array![
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [10.0, 10.0],
            [11.0, 10.0],
        ];
        let tree = BallTree::build(&data.view());
        assert_eq!(tree.n, 6);
        assert_eq!(tree.dim, 2);
        assert!(!tree.nodes.is_empty());
        assert_eq!(tree.nodes[0].count, 6);
    }

    #[test]
    fn test_all_points_in_leaves() {
        let n = 100;
        let mut data = ndarray::Array2::zeros((n, 3));
        for i in 0..n {
            data[[i, 0]] = (i as f64) * 0.1;
            data[[i, 1]] = ((i * 7) % 13) as f64;
            data[[i, 2]] = ((i * 3) % 11) as f64;
        }
        let tree = BallTree::build(&data.view());

        let mut seen = vec![false; n];
        for node in &tree.nodes {
            if node.is_leaf {
                for &idx in &tree.sorted_indices[node.idx_start..node.idx_end] {
                    assert!(!seen[idx], "Point {} appears in multiple leaves", idx);
                    seen[idx] = true;
                }
            }
        }
        for (i, &s) in seen.iter().enumerate() {
            assert!(s, "Point {} not in any leaf", i);
        }
    }

    #[test]
    fn test_dist_sq() {
        let data = array![[0.0, 0.0], [3.0, 4.0]];
        let tree = BallTree::build(&data.view());
        let d = tree.dist_sq(0, 1);
        assert!((d - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_min_dist_separated_balls() {
        // Two well-separated clusters, enough points to force internal nodes
        let n = 100;
        let mut rows = Vec::new();
        for i in 0..n / 2 {
            rows.push(vec![
                (i as f64) * 0.01,
                (i as f64) * 0.01,
            ]);
        }
        for i in 0..n / 2 {
            rows.push(vec![
                100.0 + (i as f64) * 0.01,
                100.0 + (i as f64) * 0.01,
            ]);
        }
        let mut data = ndarray::Array2::zeros((n, 2));
        for (i, row) in rows.iter().enumerate() {
            data[[i, 0]] = row[0];
            data[[i, 1]] = row[1];
        }
        let tree = BallTree::build(&data.view());

        // Root should not be a leaf
        let root = &tree.nodes[0];
        assert!(!root.is_leaf);

        // The min distance between the two child balls should be large
        if root.left != NO_CHILD && root.right != NO_CHILD {
            let min_d_sq = tree.min_dist_sq_node_to_node(root.left, root.right);
            assert!(min_d_sq > 50.0 * 50.0, "min_d_sq = {}", min_d_sq);
        }
    }

    #[test]
    fn test_min_dist_overlapping_balls() {
        // Overlapping points -- min distance should be 0
        let data = array![
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.0],
            [0.3, 0.0],
            [0.7, 0.0],
            [0.2, 0.0],
        ];
        let tree = BallTree::build(&data.view());
        // Root with itself should be 0
        let d = tree.min_dist_sq_node_to_node(0, 0);
        assert!(d == 0.0);
    }

    #[test]
    fn test_high_dim_build() {
        // Test that ball tree builds correctly in high dimensions
        let dim = 256;
        let n = 200;
        let mut data = ndarray::Array2::zeros((n, dim));
        for i in 0..n {
            for d in 0..dim {
                data[[i, d]] = ((i * 17 + d * 31) % 100) as f64 * 0.01;
            }
        }
        let tree = BallTree::build(&data.view());
        assert_eq!(tree.n, n);
        assert_eq!(tree.dim, dim);

        // Verify all points accounted for
        let mut seen = vec![false; n];
        for node in &tree.nodes {
            if node.is_leaf {
                for &idx in &tree.sorted_indices[node.idx_start..node.idx_end] {
                    seen[idx] = true;
                }
            }
        }
        assert!(seen.iter().all(|&s| s));
    }
}
