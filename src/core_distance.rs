use crate::ball_tree::BallTree;
use crate::distance;
use crate::kdtree_bounded::BoundedKdTree;
use crate::params::Metric;
use ndarray::{Array1, ArrayView2};
use ordered_float::OrderedFloat;

/// Max dimensionality for bounded kd-tree kNN. Above this, ball tree kNN is used.
/// Bounded kd-tree with AABB pruning outperforms ball tree up to ~10D.
const KDTREE_KNN_MAX_DIM: usize = 10;

/// Max dimensionality for ball tree kNN. Above this, brute force is faster
/// because tree pruning degrades due to curse of dimensionality.
const BALLTREE_KNN_MAX_DIM: usize = 512;

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
    let (core_distances, _) = compute_core_distances_with_nn(data, metric, min_samples);
    core_distances
}

/// Compute core distances and nearest non-self neighbor index for each point.
/// The nearest neighbor indices can be used to seed MST algorithms with good initial bounds.
pub fn compute_core_distances_with_nn(
    data: &ArrayView2<f64>,
    metric: &Metric,
    min_samples: usize,
) -> (Array1<f64>, Vec<usize>) {
    let n = data.nrows();
    let k = min_samples.min(n); // k-th nearest including self

    let dim = data.ncols();
    let (core_distances, nn_indices) = match metric {
        Metric::Euclidean if dim <= KDTREE_KNN_MAX_DIM => {
            let tree = BoundedKdTree::build(data);
            compute_core_distances_tree(&tree, data, k)
        }
        Metric::Euclidean if dim <= BALLTREE_KNN_MAX_DIM => {
            let tree = BallTree::build(data);
            compute_core_distances_tree(&tree, data, k)
        }
        Metric::Euclidean => compute_core_distances_brute_euclidean_with_nn(data, k),
        Metric::Precomputed => compute_core_distances_precomputed_with_nn(data, k),
        _ => compute_core_distances_brute_with_nn(data, metric, k),
    };

    let zero_count = core_distances.iter().filter(|&&d| d == 0.0).count();
    if zero_count > 0 {
        log::warn!(
            "{} points have zero core distance (likely duplicates)",
            zero_count
        );
    }

    (core_distances, nn_indices)
}

/// Compute core distances and nearest neighbors using any tree implementing CoreDistQuery.
fn compute_core_distances_tree<T: CoreDistQuery>(
    tree: &T,
    data: &ArrayView2<f64>,
    k: usize,
) -> (Array1<f64>, Vec<usize>) {
    let n = data.nrows();
    let mut core_distances = Array1::zeros(n);
    let mut nn_indices = vec![0usize; n];

    for i in 0..n {
        let query = data.row(i);
        let (core_dist, nn) = tree.query_core_dist(query.as_slice().unwrap(), k, i);
        core_distances[i] = core_dist;
        nn_indices[i] = nn;
    }

    (core_distances, nn_indices)
}

/// Trait for tree structures that can answer core distance queries.
pub trait CoreDistQuery {
    fn query_core_dist(&self, query: &[f64], k: usize, self_idx: usize) -> (f64, usize);
}

impl CoreDistQuery for BoundedKdTree {
    fn query_core_dist(&self, query: &[f64], k: usize, self_idx: usize) -> (f64, usize) {
        self.query_core_dist(query, k, self_idx)
    }
}

impl CoreDistQuery for BallTree {
    fn query_core_dist(&self, query: &[f64], k: usize, self_idx: usize) -> (f64, usize) {
        self.query_core_dist(query, k, self_idx)
    }
}

/// Brute-force Euclidean kNN for high dimensions where tree pruning is inefficient.
/// Uses SIMD-accelerated distance computation and KnnHeap for O(n^2 * d + n^2 * log k).
fn compute_core_distances_brute_euclidean_with_nn(
    data: &ArrayView2<f64>,
    k: usize,
) -> (Array1<f64>, Vec<usize>) {
    use crate::knn_heap::KnnHeap;

    let n = data.nrows();
    let dim = data.ncols();
    let data_contiguous = data.as_standard_layout();
    let data_slice = data_contiguous.as_slice().unwrap();

    let mut core_distances = Array1::zeros(n);
    let mut nn_indices = vec![0usize; n];

    for i in 0..n {
        let mut heap = KnnHeap::new(k);
        for j in 0..n {
            let d_sq = crate::simd_distance::squared_euclidean_flat(data_slice, i, j, dim);
            heap.push(d_sq, j);
        }
        core_distances[i] = heap.max_dist_sq().sqrt();
        nn_indices[i] = heap.nearest_non_self(i);
    }

    (core_distances, nn_indices)
}

fn compute_core_distances_precomputed_with_nn(
    data: &ArrayView2<f64>,
    k: usize,
) -> (Array1<f64>, Vec<usize>) {
    let n = data.nrows();
    let mut core_distances = Array1::zeros(n);
    let mut nn_indices = vec![0usize; n];

    for i in 0..n {
        let mut dists: Vec<(OrderedFloat<f64>, usize)> = (0..n)
            .map(|j| (OrderedFloat(data[[i, j]]), j))
            .collect();
        dists.sort_unstable();
        let idx = k.min(n) - 1;
        core_distances[i] = dists[idx].0.into_inner();
        // Nearest non-self
        if let Some(&(_, j)) = dists.iter().find(|&&(d, j)| j != i && d.0 > 0.0) {
            nn_indices[i] = j;
        } else if dists.len() > 1 {
            nn_indices[i] = dists[1].1;
        }
    }

    (core_distances, nn_indices)
}

fn compute_core_distances_brute_with_nn(
    data: &ArrayView2<f64>,
    metric: &Metric,
    k: usize,
) -> (Array1<f64>, Vec<usize>) {
    let n = data.nrows();
    let mut core_distances = Array1::zeros(n);
    let mut nn_indices = vec![0usize; n];

    for i in 0..n {
        let row_i = data.row(i);
        let mut dists: Vec<(OrderedFloat<f64>, usize)> = (0..n)
            .map(|j| {
                let d = if i == j {
                    0.0
                } else {
                    distance::compute_distance(&row_i, &data.row(j), metric)
                };
                (OrderedFloat(d), j)
            })
            .collect();
        dists.sort_unstable();
        let idx = k.min(n) - 1;
        core_distances[i] = dists[idx].0.into_inner();
        if let Some(&(_, j)) = dists.iter().find(|&&(d, j)| j != i && d.0 > 0.0) {
            nn_indices[i] = j;
        } else if dists.len() > 1 {
            nn_indices[i] = dists[1].1;
        }
    }

    (core_distances, nn_indices)
}

/// Compute core distances using an existing BallTree.
/// Avoids rebuilding the tree when it's already been constructed for MST.
pub fn compute_core_distances_with_balltree(
    tree: &BallTree,
    data: &ArrayView2<f64>,
    min_samples: usize,
) -> (Array1<f64>, Vec<usize>) {
    let k = min_samples.min(data.nrows());
    compute_core_distances_tree(tree, data, k)
}

/// Compute core distances using an existing BoundedKdTree.
pub fn compute_core_distances_with_bounded_kdtree(
    tree: &BoundedKdTree,
    data: &ArrayView2<f64>,
    min_samples: usize,
) -> (Array1<f64>, Vec<usize>) {
    let k = min_samples.min(data.nrows());
    compute_core_distances_tree(tree, data, k)
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
        let tree = BoundedKdTree::build(&data.view());
        let (cd_kd, _) = compute_core_distances_with_bounded_kdtree(&tree, &data.view(), 3);
        let (cd_brute, _) = compute_core_distances_brute_with_nn(&data.view(), &Metric::Euclidean, 3);
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
