//! KD-tree with bounding boxes for dual-tree Boruvka MST.
//!
//! Points are stored ONLY in leaf nodes. Internal nodes are pure split nodes
//! with bounding box metadata for pruning. This is the standard structure
//! for dual-tree algorithms.

use ndarray::ArrayView2;

/// A KD-tree node with bounding box information for subtree pruning.
#[derive(Clone)]
pub struct BKdNode {
    /// Split dimension (only meaningful for internal nodes)
    pub split_dim: usize,
    /// Split value (only meaningful for internal nodes)
    pub split_val: f64,
    /// Left child index (or NO_CHILD if leaf)
    pub left: usize,
    /// Right child index (or NO_CHILD if leaf)
    pub right: usize,
    /// Bounding box minimum for each dimension
    pub bbox_min: Vec<f64>,
    /// Bounding box maximum for each dimension
    pub bbox_max: Vec<f64>,
    /// Number of points in this subtree
    pub count: usize,
    /// Start index in the sorted_indices array
    pub idx_start: usize,
    /// End index (exclusive) in the sorted_indices array
    pub idx_end: usize,
    /// Whether this node is a leaf
    pub is_leaf: bool,
}

pub const NO_CHILD: usize = usize::MAX;

/// Compute optimal leaf size based on dimensionality.
/// Smaller leaves give tighter bounding boxes in low dims;
/// larger leaves reduce tree overhead in higher dims.
#[inline]
pub fn leaf_size(dim: usize) -> usize {
    if dim <= 4 {
        10
    } else {
        20
    }
}

/// KD-tree with bounding boxes for dual-tree algorithms.
/// Points are stored only in leaves; internal nodes contain split metadata.
pub struct BoundedKdTree {
    pub nodes: Vec<BKdNode>,
    /// The raw data, stored as a flat [n * dim] array for cache efficiency.
    pub data: Vec<f64>,
    pub dim: usize,
    pub n: usize,
    /// Sorted point indices (each leaf's points are contiguous in this array)
    pub sorted_indices: Vec<usize>,
}

impl BoundedKdTree {
    /// Build a bounded KD-tree from an n×d data matrix.
    pub fn build(data: &ArrayView2<f64>) -> Self {
        let n = data.nrows();
        let dim = data.ncols();

        let data_contiguous = data.as_standard_layout();
        let flat_data: Vec<f64> = data_contiguous.as_slice().unwrap().to_vec();

        let mut sorted_indices: Vec<usize> = (0..n).collect();
        let ls = leaf_size(dim);
        let mut nodes = Vec::with_capacity(2 * n / ls + 1);

        if n > 0 {
            Self::build_recursive(&flat_data, &mut sorted_indices, 0, n, dim, &mut nodes);
        }

        BoundedKdTree {
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
        nodes: &mut Vec<BKdNode>,
    ) -> usize {
        if start >= end {
            return NO_CHILD;
        }

        let count = end - start;

        // Compute bounding box
        let mut bbox_min = vec![f64::INFINITY; dim];
        let mut bbox_max = vec![f64::NEG_INFINITY; dim];
        for &idx in &indices[start..end] {
            let base = idx * dim;
            for d in 0..dim {
                let v = data[base + d];
                if v < bbox_min[d] {
                    bbox_min[d] = v;
                }
                if v > bbox_max[d] {
                    bbox_max[d] = v;
                }
            }
        }

        // Leaf node: store all points
        if count <= leaf_size(dim) {
            let node_idx = nodes.len();
            nodes.push(BKdNode {
                split_dim: 0,
                split_val: 0.0,
                left: NO_CHILD,
                right: NO_CHILD,
                bbox_min,
                bbox_max,
                count,
                idx_start: start,
                idx_end: end,
                is_leaf: true,
            });
            return node_idx;
        }

        // Find dimension with max spread
        let mut best_dim = 0;
        let mut best_spread = f64::NEG_INFINITY;
        for d in 0..dim {
            let spread = bbox_max[d] - bbox_min[d];
            if spread > best_spread {
                best_spread = spread;
                best_dim = d;
            }
        }

        // Partition around median
        let mid = start + count / 2;
        indices[start..end].select_nth_unstable_by(mid - start, |&a, &b| {
            let va = data[a * dim + best_dim];
            let vb = data[b * dim + best_dim];
            va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
        });

        let split_val = data[indices[mid] * dim + best_dim];

        let node_idx = nodes.len();
        nodes.push(BKdNode {
            split_dim: best_dim,
            split_val,
            left: NO_CHILD,
            right: NO_CHILD,
            bbox_min,
            bbox_max,
            count,
            idx_start: start,
            idx_end: end,
            is_leaf: false,
        });

        // All points go into children — no point stored at internal node
        let left = Self::build_recursive(data, indices, start, mid, dim, nodes);
        let right = Self::build_recursive(data, indices, mid, end, dim, nodes);

        nodes[node_idx].left = left;
        nodes[node_idx].right = right;

        node_idx
    }

    /// Compute the minimum possible squared distance between two nodes' bounding boxes.
    #[inline]
    pub fn min_dist_sq_node_to_node(&self, node_a: usize, node_b: usize) -> f64 {
        let a = &self.nodes[node_a];
        let b = &self.nodes[node_b];
        let a_min = a.bbox_min.as_slice();
        let a_max = a.bbox_max.as_slice();
        let b_min = b.bbox_min.as_slice();
        let b_max = b.bbox_max.as_slice();
        let dim = self.dim;
        let mut dist_sq = 0.0f64;
        for d in 0..dim {
            unsafe {
                let a_lo = *a_min.get_unchecked(d);
                let b_hi = *b_max.get_unchecked(d);
                if a_lo > b_hi {
                    let diff = a_lo - b_hi;
                    dist_sq += diff * diff;
                } else {
                    let b_lo = *b_min.get_unchecked(d);
                    let a_hi = *a_max.get_unchecked(d);
                    if b_lo > a_hi {
                        let diff = b_lo - a_hi;
                        dist_sq += diff * diff;
                    }
                }
            }
        }
        dist_sq
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

    /// Find the k-th nearest distance and the nearest non-self neighbor.
    #[inline]
    pub fn query_core_dist(&self, query: &[f64], k: usize, self_idx: usize) -> (f64, usize) {
        if self.nodes.is_empty() || k == 0 {
            return (0.0, 0);
        }
        let mut heap = crate::knn_heap::KnnHeap::new(k);
        self.knn_recursive(0, query, &mut heap);
        let core_dist = heap.max_dist_sq().sqrt();
        let nn = heap.nearest_non_self(self_idx);
        (core_dist, nn)
    }

    /// Find k nearest neighbors of query point.
    /// Returns (distance, index) sorted by distance.
    pub fn query_knn(&self, query: &[f64], k: usize) -> Vec<(f64, usize)> {
        if self.nodes.is_empty() || k == 0 {
            return vec![];
        }
        let mut heap = crate::knn_heap::KnnHeap::new(k);
        self.knn_recursive(0, query, &mut heap);
        heap.into_sorted_distances()
    }

    fn knn_recursive(&self, node_idx: usize, query: &[f64], heap: &mut crate::knn_heap::KnnHeap) {
        let node = &self.nodes[node_idx];
        let dim = self.dim;

        // Pruning: min distance from query to bounding box
        let mut min_dist_sq = 0.0f64;
        let bbox_min = node.bbox_min.as_slice();
        let bbox_max = node.bbox_max.as_slice();
        for d in 0..dim {
            unsafe {
                let q = *query.get_unchecked(d);
                let lo = *bbox_min.get_unchecked(d);
                if q < lo {
                    let diff = lo - q;
                    min_dist_sq += diff * diff;
                } else {
                    let hi = *bbox_max.get_unchecked(d);
                    if q > hi {
                        let diff = q - hi;
                        min_dist_sq += diff * diff;
                    }
                }
            }
        }
        if heap.is_full() && min_dist_sq >= heap.max_dist_sq() {
            return;
        }

        if node.is_leaf {
            let data = &self.data;
            for &idx in &self.sorted_indices[node.idx_start..node.idx_end] {
                let off = idx * dim;
                let point = unsafe { data.get_unchecked(off..off + dim) };
                let dist_sq = crate::simd_distance::squared_euclidean_simd(query, point);
                heap.push(dist_sq, idx);
            }
        } else {
            // Visit closer child first
            let split_diff = query[node.split_dim] - node.split_val;
            let (first, second) = if split_diff <= 0.0 {
                (node.left, node.right)
            } else {
                (node.right, node.left)
            };

            if first != NO_CHILD {
                self.knn_recursive(first, query, heap);
            }
            if second != NO_CHILD {
                // Only visit far side if splitting plane distance could beat current k-th best
                let plane_dist_sq = split_diff * split_diff;
                if !heap.is_full() || plane_dist_sq < heap.max_dist_sq() {
                    self.knn_recursive(second, query, heap);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_bounded_kdtree_build() {
        let data = array![
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [10.0, 10.0],
            [11.0, 10.0],
        ];
        let tree = BoundedKdTree::build(&data.view());
        assert_eq!(tree.n, 6);
        assert_eq!(tree.dim, 2);
        assert!(!tree.nodes.is_empty());

        // Root should contain all 6 points
        let root = &tree.nodes[0];
        assert_eq!(root.count, 6);
    }

    #[test]
    fn test_all_points_in_leaves() {
        let data = array![
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
            [5.0, 0.0],
            [6.0, 0.0],
            [7.0, 0.0],
            [8.0, 0.0],
            [9.0, 0.0],
            [10.0, 0.0],
            [11.0, 0.0],
            [12.0, 0.0],
            [13.0, 0.0],
            [14.0, 0.0],
            [15.0, 0.0],
            [16.0, 0.0],
            [17.0, 0.0],
            [18.0, 0.0],
            [19.0, 0.0],
            [20.0, 0.0],
            [21.0, 0.0],
            [22.0, 0.0],
            [23.0, 0.0],
            [24.0, 0.0],
            [25.0, 0.0],
            [26.0, 0.0],
            [27.0, 0.0],
            [28.0, 0.0],
            [29.0, 0.0],
            [30.0, 0.0],
            [31.0, 0.0],
            [32.0, 0.0],
            [33.0, 0.0],
            [34.0, 0.0],
            [35.0, 0.0],
            [36.0, 0.0],
            [37.0, 0.0],
            [38.0, 0.0],
            [39.0, 0.0],
            [40.0, 0.0],
            [41.0, 0.0],
            [42.0, 0.0],
            [43.0, 0.0],
            [44.0, 0.0],
            [45.0, 0.0],
            [46.0, 0.0],
            [47.0, 0.0],
            [48.0, 0.0],
            [49.0, 0.0],
        ];
        let tree = BoundedKdTree::build(&data.view());

        // Verify all points appear exactly once across all leaves
        let mut seen = vec![false; 50];
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
        let tree = BoundedKdTree::build(&data.view());
        let d = tree.dist_sq(0, 1);
        assert!((d - 25.0).abs() < 1e-10);
    }
}
