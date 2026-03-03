pub mod prim;

pub use prim::prim_mst;

use crate::params::Metric;
use crate::types::MstEdge;
use ndarray::{ArrayView1, ArrayView2};

/// Build MST on the mutual reachability graph.
pub fn auto_mst(
    data: &ArrayView2<f64>,
    core_distances: &ArrayView1<f64>,
    metric: &Metric,
    alpha: f64,
) -> Vec<MstEdge> {
    prim_mst(data, core_distances, metric, alpha)
}
