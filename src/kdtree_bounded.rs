//! KD-tree with bounding boxes for dual-tree Boruvka MST.
//!
//! Points are stored ONLY in leaf nodes. Internal nodes are pure split nodes
//! with bounding box metadata for pruning. This is the standard structure
//! for dual-tree algorithms.
//!
//! Bounding boxes are stored in Structure-of-Arrays layout: two contiguous
//! f64 arrays (all mins, all maxes) indexed by `node_idx * dim + d`. This
//! eliminates per-node Vec allocations and improves cache locality in the
//! hot `min_dist_sq_node_to_node` inner loop.

use ndarray::ArrayView2;

/// A KD-tree node (without bounding box data — stored in SoA arrays on the tree).
#[derive(Clone, Copy)]
pub struct BKdNode {
    /// Split dimension (only meaningful for internal nodes)
    pub split_dim: usize,
    /// Split value (only meaningful for internal nodes)
    pub split_val: f64,
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
///
/// Bounding boxes use SoA layout for cache-friendly access:
/// `bbox_min[node_idx * dim + d]` gives the minimum value for node `node_idx` in dimension `d`.
pub struct BoundedKdTree {
    pub nodes: Vec<BKdNode>,
    /// Contiguous bounding box minimums: `bbox_min[node * dim + d]`
    pub bbox_min: Vec<f64>,
    /// Contiguous bounding box maximums: `bbox_max[node * dim + d]`
    pub bbox_max: Vec<f64>,
    /// The raw data, stored as a flat [n * dim] array in ORIGINAL index order.
    pub data: Vec<f64>,
    /// Data reordered to match sorted_indices (tree traversal order).
    /// `tree_data[pos * dim .. (pos+1) * dim]` = data for sorted_indices[pos].
    /// Enables sequential memory access when iterating leaf points.
    pub tree_data: Vec<f64>,
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
        let max_nodes = 2 * n / ls + 1;
        let mut nodes = Vec::with_capacity(max_nodes);
        let mut bbox_min_buf = Vec::with_capacity(max_nodes * dim);
        let mut bbox_max_buf = Vec::with_capacity(max_nodes * dim);

        if n > 0 {
            Self::build_recursive(
                &flat_data,
                &mut sorted_indices,
                0,
                n,
                dim,
                &mut nodes,
                &mut bbox_min_buf,
                &mut bbox_max_buf,
            );
        }

        // Build tree-ordered data for cache-friendly leaf access
        let mut tree_data = vec![0.0f64; n * dim];
        for (pos, &orig_idx) in sorted_indices.iter().enumerate() {
            let src = orig_idx * dim;
            let dst = pos * dim;
            tree_data[dst..dst + dim].copy_from_slice(&flat_data[src..src + dim]);
        }

        BoundedKdTree {
            nodes,
            bbox_min: bbox_min_buf,
            bbox_max: bbox_max_buf,
            data: flat_data,
            tree_data,
            dim,
            n,
            sorted_indices,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn build_recursive(
        data: &[f64],
        indices: &mut [usize],
        start: usize,
        end: usize,
        dim: usize,
        nodes: &mut Vec<BKdNode>,
        bbox_min_buf: &mut Vec<f64>,
        bbox_max_buf: &mut Vec<f64>,
    ) -> usize {
        if start >= end {
            return NO_CHILD;
        }

        let count = end - start;

        // Compute bounding box into temporary storage
        let bb_start = bbox_min_buf.len();
        bbox_min_buf.extend(std::iter::repeat_n(f64::INFINITY, dim));
        bbox_max_buf.extend(std::iter::repeat_n(f64::NEG_INFINITY, dim));

        let bb_min = &mut bbox_min_buf[bb_start..bb_start + dim];
        let bb_max = &mut bbox_max_buf[bb_start..bb_start + dim];

        for &idx in &indices[start..end] {
            let base = idx * dim;
            for d in 0..dim {
                let v = data[base + d];
                if v < bb_min[d] {
                    bb_min[d] = v;
                }
                if v > bb_max[d] {
                    bb_max[d] = v;
                }
            }
        }

        // Leaf node: store all points
        if count <= leaf_size(dim) {
            let node_idx = nodes.len();
            debug_assert_eq!(node_idx * dim, bb_start);
            nodes.push(BKdNode {
                split_dim: 0,
                split_val: 0.0,
                left: NO_CHILD,
                right: NO_CHILD,
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
            let spread = bb_max[d] - bb_min[d];
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
        debug_assert_eq!(node_idx * dim, bb_start);
        nodes.push(BKdNode {
            split_dim: best_dim,
            split_val,
            left: NO_CHILD,
            right: NO_CHILD,
            count,
            idx_start: start,
            idx_end: end,
            is_leaf: false,
        });

        // All points go into children — no point stored at internal node
        let left = Self::build_recursive(data, indices, start, mid, dim, nodes, bbox_min_buf, bbox_max_buf);
        let right = Self::build_recursive(data, indices, mid, end, dim, nodes, bbox_min_buf, bbox_max_buf);

        nodes[node_idx].left = left;
        nodes[node_idx].right = right;

        node_idx
    }

    /// Compute the minimum possible squared distance between two nodes' bounding boxes.
    #[inline]
    pub fn min_dist_sq_node_to_node(&self, node_a: usize, node_b: usize) -> f64 {
        let dim = self.dim;
        let off_a = node_a * dim;
        let off_b = node_b * dim;
        let a_min = &self.bbox_min;
        let a_max = &self.bbox_max;
        let b_min = &self.bbox_min;
        let b_max = &self.bbox_max;
        let mut dist_sq = 0.0f64;
        // Branchless: for each dim, gap = max(a_lo - b_hi, 0) + max(b_lo - a_hi, 0).
        // At most one term is non-zero. Compiles to maxpd + vfmadd (SIMD-friendly).
        for d in 0..dim {
            unsafe {
                let a_lo = *a_min.get_unchecked(off_a + d);
                let a_hi = *a_max.get_unchecked(off_a + d);
                let b_lo = *b_min.get_unchecked(off_b + d);
                let b_hi = *b_max.get_unchecked(off_b + d);
                let gap = f64::max(a_lo - b_hi, 0.0) + f64::max(b_lo - a_hi, 0.0);
                dist_sq += gap * gap;
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

    /// Public wrapper for knn_recursive, used by CoreDistQuery::query_core_dist_reuse.
    #[inline]
    pub fn knn_recursive_pub(&self, node_idx: usize, query: &[f64], heap: &mut crate::knn_heap::KnnHeap) {
        self.knn_recursive(node_idx, query, heap);
    }

    fn knn_recursive(&self, node_idx: usize, query: &[f64], heap: &mut crate::knn_heap::KnnHeap) {
        let node = &self.nodes[node_idx];
        let dim = self.dim;

        // Pruning: min distance from query to bounding box (SoA access)
        let mut min_dist_sq = 0.0f64;
        let bb_off = node_idx * dim;
        for d in 0..dim {
            unsafe {
                let q = *query.get_unchecked(d);
                let lo = *self.bbox_min.get_unchecked(bb_off + d);
                let hi = *self.bbox_max.get_unchecked(bb_off + d);
                let gap = f64::max(lo - q, 0.0) + f64::max(q - hi, 0.0);
                min_dist_sq += gap * gap;
            }
        }
        if heap.is_full() && min_dist_sq >= heap.max_dist_sq() {
            return;
        }

        if node.is_leaf {
            let tree_data = &self.tree_data;
            for pos in node.idx_start..node.idx_end {
                let off = pos * dim;
                let point = unsafe { tree_data.get_unchecked(off..off + dim) };
                let dist_sq = crate::simd_distance::squared_euclidean_simd(query, point);
                let idx = self.sorted_indices[pos];
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

        // Verify SoA bbox data is correct for root (node 0)
        assert_eq!(tree.bbox_min[0], 0.0); // dim 0 min
        assert_eq!(tree.bbox_min[1], 0.0); // dim 1 min
        assert_eq!(tree.bbox_max[0], 11.0); // dim 0 max
        assert_eq!(tree.bbox_max[1], 10.0); // dim 1 max
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
