pub mod boruvka;
pub mod dual_tree_boruvka;
pub mod prim;

pub use boruvka::boruvka_mst;
pub use dual_tree_boruvka::dual_tree_boruvka_mst;
pub use prim::prim_mst;

use crate::kdtree_bounded::BoundedKdTree;
use crate::params::Metric;
use crate::types::MstEdge;
use ndarray::{ArrayView1, ArrayView2};

/// Threshold above which dual-tree Boruvka is used for Euclidean metric.
/// Below this, optimized Prim's with on-the-fly distances is faster due
/// to lower constant factors.
const DUAL_TREE_THRESHOLD: usize = 1500;

/// Build MST on the mutual reachability graph, automatically selecting
/// the best algorithm based on dataset size and metric.
///
/// - Euclidean with n > DUAL_TREE_THRESHOLD: Dual-tree Boruvka (O(n log n))
/// - Otherwise: Prim's with on-the-fly distances (O(n²) with pruning)
pub fn auto_mst(
    data: &ArrayView2<f64>,
    core_distances: &ArrayView1<f64>,
    metric: &Metric,
    alpha: f64,
) -> Vec<MstEdge> {
    let n = data.nrows();

    match metric {
        Metric::Euclidean if n > DUAL_TREE_THRESHOLD => {
            let tree = BoundedKdTree::build(data);
            dual_tree_boruvka_mst(&tree, core_distances, alpha)
        }
        _ => prim_mst(data, core_distances, metric, alpha),
    }
}
