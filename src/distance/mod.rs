pub mod chebyshev;
pub mod cosine;
pub mod euclidean;
pub mod manhattan;
pub mod minkowski;
pub mod precomputed;

use crate::params::Metric;
use ndarray::{ArrayView1, ArrayView2};

/// Compute the distance between two points using the given metric.
#[inline]
pub fn compute_distance(a: &ArrayView1<f64>, b: &ArrayView1<f64>, metric: &Metric) -> f64 {
    match metric {
        Metric::Euclidean => euclidean::euclidean_distance(a, b),
        Metric::Manhattan => manhattan::manhattan_distance(a, b),
        Metric::Cosine => cosine::cosine_distance(a, b),
        Metric::Minkowski(p) => minkowski::minkowski_distance(a, b, *p),
        Metric::Precomputed => {
            panic!("compute_distance should not be called with Precomputed metric")
        }
    }
}

/// Build a full pairwise distance matrix for non-precomputed metrics.
pub fn pairwise_distances(data: &ArrayView2<f64>, metric: &Metric) -> ndarray::Array2<f64> {
    let n = data.nrows();
    let mut dist = ndarray::Array2::zeros((n, n));
    for i in 0..n {
        let row_i = data.row(i);
        for j in (i + 1)..n {
            let row_j = data.row(j);
            let d = compute_distance(&row_i, &row_j, metric);
            dist[[i, j]] = d;
            dist[[j, i]] = d;
        }
    }
    dist
}
