pub mod boruvka;
pub mod dual_tree_boruvka;
pub mod prim;

pub use boruvka::boruvka_mst;
pub use dual_tree_boruvka::dual_tree_boruvka_mst;
pub use prim::prim_mst;

use crate::ball_tree::BallTree;
use crate::kdtree_bounded::BoundedKdTree;
use crate::params::Metric;
use crate::types::MstEdge;
use ndarray::{ArrayView1, ArrayView2};

/// Maximum dimensionality for kd-tree Boruvka. Above this, ball tree is used.
const KDTREE_MAX_DIM: usize = 16;

/// Select the best MST algorithm based on n, dimensionality, and metric.
///
/// The crossover point between O(n^2) Prim's and O(n log n) tree-based Boruvka
/// depends on dimensionality: tree pruning degrades in higher dims, so we need
/// more points before the tree approach wins. Empirically tuned on real
/// sklearn-like blob data (Gaussian clusters with overlapping tails).
pub fn dual_tree_threshold(dim: usize) -> usize {
    if dim <= 4 {
        1500
    } else if dim <= 16 {
        // kd-tree region: Boruvka + shared tree wins above this threshold
        4000
    } else if dim <= 64 {
        // Medium dims: Prim's O(n^2) still competitive; ball tree pruning
        // degrades at high dims so we need large n for the tree to win.
        8000
    } else {
        // High dims (LLM embeddings): Prim's O(n^2 * d) is expensive.
        // Ball tree Boruvka at large n.
        2000
    }
}

/// Build MST on the mutual reachability graph, automatically selecting
/// the best algorithm based on dataset size, dimensionality, and metric.
///
/// - Euclidean, low dim, large n: Dual-tree Boruvka with kd-tree
/// - Euclidean, high dim, large n: Dual-tree Boruvka with ball tree
/// - Small n or non-Euclidean: Prim's O(n^2)
pub fn auto_mst(
    data: &ArrayView2<f64>,
    core_distances: &ArrayView1<f64>,
    metric: &Metric,
    alpha: f64,
    nn_indices: Option<&[usize]>,
) -> Vec<MstEdge> {
    let n = data.nrows();
    let dim = data.ncols();
    let threshold = dual_tree_threshold(dim);

    match metric {
        Metric::Euclidean if n > threshold && dim <= KDTREE_MAX_DIM => {
            let tree = BoundedKdTree::build(data);
            dual_tree_boruvka_mst(&tree, core_distances, alpha, nn_indices)
        }
        Metric::Euclidean if n > threshold => {
            let tree = BallTree::build(data);
            dual_tree_boruvka_mst(&tree, core_distances, alpha, nn_indices)
        }
        _ => prim::prim_mst_seeded(data, core_distances, metric, alpha, nn_indices),
    }
}
