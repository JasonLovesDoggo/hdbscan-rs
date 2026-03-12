//! Trait abstracting over spatial tree implementations (kd-tree, ball tree)
//! for use in dual-tree Boruvka MST construction.

/// A spatial tree node with the fields needed by dual-tree Boruvka.
pub trait SpatialNode {
    fn is_leaf(&self) -> bool;
    fn left(&self) -> usize;
    fn right(&self) -> usize;
    fn count(&self) -> usize;
    fn idx_start(&self) -> usize;
    fn idx_end(&self) -> usize;
}

/// A spatial tree that supports dual-tree traversal and distance queries.
pub trait SpatialTree {
    type Node: SpatialNode;

    fn nodes(&self) -> &[Self::Node];
    fn data(&self) -> &[f64];
    fn dim(&self) -> usize;
    fn n(&self) -> usize;
    fn sorted_indices(&self) -> &[usize];

    /// Minimum possible squared distance between two nodes.
    fn min_dist_sq_node_to_node(&self, node_a: usize, node_b: usize) -> f64;

    /// Squared Euclidean distance between two points (by original index).
    fn dist_sq(&self, i: usize, j: usize) -> f64;

    /// Get point indices for a node's subtree.
    fn node_points(&self, node_idx: usize) -> &[usize];

    /// Tree-ordered data for cache-friendly leaf access.
    /// `tree_data[pos * dim .. (pos+1) * dim]` = data for sorted_indices[pos].
    /// Returns None if not available (falls back to original data + sorted_indices).
    fn tree_data(&self) -> Option<&[f64]> {
        None
    }

    /// f32 copy of tree_data for half-bandwidth leaf distance computation.
    fn tree_data_f32(&self) -> Option<&[f32]> {
        None
    }
}

// --- BoundedKdTree impl ---

impl SpatialNode for crate::kdtree_bounded::BKdNode {
    #[inline]
    fn is_leaf(&self) -> bool {
        self.is_leaf
    }
    #[inline]
    fn left(&self) -> usize {
        self.left
    }
    #[inline]
    fn right(&self) -> usize {
        self.right
    }
    #[inline]
    fn count(&self) -> usize {
        self.count
    }
    #[inline]
    fn idx_start(&self) -> usize {
        self.idx_start
    }
    #[inline]
    fn idx_end(&self) -> usize {
        self.idx_end
    }
}

impl SpatialTree for crate::kdtree_bounded::BoundedKdTree {
    type Node = crate::kdtree_bounded::BKdNode;

    #[inline]
    fn nodes(&self) -> &[Self::Node] {
        &self.nodes
    }
    #[inline]
    fn data(&self) -> &[f64] {
        &self.data
    }
    #[inline]
    fn dim(&self) -> usize {
        self.dim
    }
    #[inline]
    fn n(&self) -> usize {
        self.n
    }
    #[inline]
    fn sorted_indices(&self) -> &[usize] {
        &self.sorted_indices
    }
    #[inline]
    fn min_dist_sq_node_to_node(&self, node_a: usize, node_b: usize) -> f64 {
        self.min_dist_sq_node_to_node(node_a, node_b)
    }
    #[inline]
    fn dist_sq(&self, i: usize, j: usize) -> f64 {
        self.dist_sq(i, j)
    }
    #[inline]
    fn node_points(&self, node_idx: usize) -> &[usize] {
        self.node_points(node_idx)
    }
    #[inline]
    fn tree_data(&self) -> Option<&[f64]> {
        Some(&self.tree_data)
    }
    #[inline]
    fn tree_data_f32(&self) -> Option<&[f32]> {
        Some(&self.tree_data_f32)
    }
}

// --- BallTree impl ---

impl SpatialNode for crate::ball_tree::BallNode {
    #[inline]
    fn is_leaf(&self) -> bool {
        self.is_leaf
    }
    #[inline]
    fn left(&self) -> usize {
        self.left
    }
    #[inline]
    fn right(&self) -> usize {
        self.right
    }
    #[inline]
    fn count(&self) -> usize {
        self.count
    }
    #[inline]
    fn idx_start(&self) -> usize {
        self.idx_start
    }
    #[inline]
    fn idx_end(&self) -> usize {
        self.idx_end
    }
}

impl SpatialTree for crate::ball_tree::BallTree {
    type Node = crate::ball_tree::BallNode;

    #[inline]
    fn nodes(&self) -> &[Self::Node] {
        &self.nodes
    }
    #[inline]
    fn data(&self) -> &[f64] {
        &self.data
    }
    #[inline]
    fn dim(&self) -> usize {
        self.dim
    }
    #[inline]
    fn n(&self) -> usize {
        self.n
    }
    #[inline]
    fn sorted_indices(&self) -> &[usize] {
        &self.sorted_indices
    }
    #[inline]
    fn min_dist_sq_node_to_node(&self, node_a: usize, node_b: usize) -> f64 {
        self.min_dist_sq_node_to_node(node_a, node_b)
    }
    #[inline]
    fn dist_sq(&self, i: usize, j: usize) -> f64 {
        self.dist_sq(i, j)
    }
    #[inline]
    fn node_points(&self, node_idx: usize) -> &[usize] {
        self.node_points(node_idx)
    }
    #[inline]
    fn tree_data(&self) -> Option<&[f64]> {
        Some(&self.tree_data)
    }
    #[inline]
    fn tree_data_f32(&self) -> Option<&[f32]> {
        Some(&self.tree_data_f32)
    }
}
