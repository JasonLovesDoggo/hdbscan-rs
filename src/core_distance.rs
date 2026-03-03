use crate::distance;
use crate::kdtree::KdTree;
use crate::params::Metric;
use ndarray::{Array1, ArrayView2};
use ordered_float::OrderedFloat;

/// Compute the core distance for each point.
///
/// The core distance of a point is the distance to its k-th nearest neighbor
/// (where k = min_samples). Following sklearn convention, the point itself is
/// included in the count (so k=5 means the 5th nearest including itself, i.e.,
/// the 4th nearest neighbor).
pub fn compute_core_distances(
    data: &ArrayView2<f64>,
    metric: &Metric,
    min_samples: usize,
) -> Array1<f64> {
    let n = data.nrows();
    let k = min_samples.min(n); // k-th nearest including self

    let core_distances = match metric {
        Metric::Euclidean => compute_core_distances_kdtree(data, k),
        Metric::Precomputed => compute_core_distances_precomputed(data, k),
        _ => compute_core_distances_brute(data, metric, k),
    };

    let zero_count = core_distances.iter().filter(|&&d| d == 0.0).count();
    if zero_count > 0 {
        log::warn!(
            "{} points have zero core distance (likely duplicates)",
            zero_count
        );
    }

    core_distances
}

/// KD-tree accelerated core distance computation for Euclidean metric.
/// O(n log n) instead of O(n²).
fn compute_core_distances_kdtree(data: &ArrayView2<f64>, k: usize) -> Array1<f64> {
    let n = data.nrows();
    let tree = KdTree::build(data);
    let mut core_distances = Array1::zeros(n);

    for i in 0..n {
        // query_knn returns k nearest including the point itself (dist=0)
        let query = data.row(i);
        let neighbors = tree.query_knn(query.as_slice().unwrap(), k);
        // The k-th nearest (last in sorted result) is the core distance
        if let Some(&(dist, _)) = neighbors.last() {
            core_distances[i] = dist;
        }
    }

    core_distances
}

fn compute_core_distances_precomputed(data: &ArrayView2<f64>, k: usize) -> Array1<f64> {
    let n = data.nrows();
    let mut core_distances = Array1::zeros(n);

    for i in 0..n {
        let mut dists: Vec<OrderedFloat<f64>> =
            (0..n).map(|j| OrderedFloat(data[[i, j]])).collect();
        dists.sort_unstable();
        let idx = k.min(n) - 1;
        core_distances[i] = dists[idx].into_inner();
    }

    core_distances
}

fn compute_core_distances_brute(
    data: &ArrayView2<f64>,
    metric: &Metric,
    k: usize,
) -> Array1<f64> {
    let n = data.nrows();
    let mut core_distances = Array1::zeros(n);

    for i in 0..n {
        let row_i = data.row(i);
        let mut dists: Vec<OrderedFloat<f64>> = (0..n)
            .map(|j| {
                if i == j {
                    OrderedFloat(0.0)
                } else {
                    OrderedFloat(distance::compute_distance(&row_i, &data.row(j), metric))
                }
            })
            .collect();
        dists.sort_unstable();
        let idx = k.min(n) - 1;
        core_distances[i] = dists[idx].into_inner();
    }

    core_distances
}

/// Build a KD-tree from data (for reuse by MST).
pub fn build_kdtree(data: &ArrayView2<f64>) -> KdTree {
    KdTree::build(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_core_distances_simple() {
        let data = array![[0.0], [1.0], [2.0], [3.0]];
        let cd = compute_core_distances(&data.view(), &Metric::Euclidean, 2);
        assert!((cd[0] - 1.0).abs() < 1e-12);
        assert!((cd[1] - 1.0).abs() < 1e-12);
        assert!((cd[2] - 1.0).abs() < 1e-12);
        assert!((cd[3] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_core_distances_duplicates() {
        let data = array![[0.0], [0.0], [0.0]];
        let cd = compute_core_distances(&data.view(), &Metric::Euclidean, 2);
        assert_eq!(cd[0], 0.0);
        assert_eq!(cd[1], 0.0);
        assert_eq!(cd[2], 0.0);
    }

    #[test]
    fn test_kdtree_matches_brute() {
        let data = array![
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [5.0, 5.0],
            [5.1, 5.0],
            [5.0, 5.1],
        ];
        let cd_kd = compute_core_distances_kdtree(&data.view(), 3);
        let cd_brute = compute_core_distances_brute(&data.view(), &Metric::Euclidean, 3);
        for i in 0..6 {
            assert!(
                (cd_kd[i] - cd_brute[i]).abs() < 1e-10,
                "Mismatch at {}: kdtree={} brute={}",
                i,
                cd_kd[i],
                cd_brute[i]
            );
        }
    }
}
