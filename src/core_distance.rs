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
            compute_core_distances_bounded_kdtree_with_nn(data, k)
        }
        Metric::Euclidean if dim <= BALLTREE_KNN_MAX_DIM => {
            compute_core_distances_balltree_with_nn(data, k)
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

/// Bounded KD-tree accelerated core distance computation for Euclidean metric.
/// Uses AABB pruning for better kNN performance than basic KD-tree.
fn compute_core_distances_bounded_kdtree_with_nn(
    data: &ArrayView2<f64>,
    k: usize,
) -> (Array1<f64>, Vec<usize>) {
    let n = data.nrows();
    let tree = BoundedKdTree::build(data);
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

/// Ball tree accelerated core distance computation for medium-to-high dimensions.
fn compute_core_distances_balltree_with_nn(
    data: &ArrayView2<f64>,
    k: usize,
) -> (Array1<f64>, Vec<usize>) {
    let n = data.nrows();
    let tree = BallTree::build(data);
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

/// Brute-force Euclidean kNN for high dimensions where tree pruning is inefficient.
/// Uses SIMD-accelerated distance computation and a binary max-heap for O(n^2 * d + n^2 * log k).
fn compute_core_distances_brute_euclidean_with_nn(
    data: &ArrayView2<f64>,
    k: usize,
) -> (Array1<f64>, Vec<usize>) {
    let n = data.nrows();
    let dim = data.ncols();
    let data_contiguous = data.as_standard_layout();
    let data_slice = data_contiguous.as_slice().unwrap();

    let mut core_distances = Array1::zeros(n);
    let mut nn_indices = vec![0usize; n];

    for i in 0..n {
        // Use a simple bounded max-heap of size k
        let mut best = Vec::with_capacity(k);
        let mut max_dist_sq = f64::INFINITY;

        for j in 0..n {
            let d_sq = crate::simd_distance::squared_euclidean_flat(data_slice, i, j, dim);
            if best.len() < k {
                best.push((d_sq, j));
                if best.len() == k {
                    // Build max-heap
                    for idx in (0..k / 2).rev() {
                        sift_down_f64(&mut best, idx);
                    }
                    max_dist_sq = best[0].0;
                }
            } else if d_sq < max_dist_sq {
                best[0] = (d_sq, j);
                sift_down_f64(&mut best, 0);
                max_dist_sq = best[0].0;
            }
        }

        // k-th nearest distance (max in heap)
        core_distances[i] = max_dist_sq.sqrt();

        // Nearest non-self neighbor
        best.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        if let Some(&(_, idx)) = best.iter().find(|&&(_, idx)| idx != i) {
            nn_indices[i] = idx;
        } else if best.len() > 1 {
            nn_indices[i] = best[1].1;
        }
    }

    (core_distances, nn_indices)
}

/// Sift down for a max-heap of (f64, usize) pairs.
fn sift_down_f64(heap: &mut [(f64, usize)], mut idx: usize) {
    let len = heap.len();
    loop {
        let left = 2 * idx + 1;
        let right = 2 * idx + 2;
        let mut largest = idx;
        if left < len && heap[left].0 > heap[largest].0 {
            largest = left;
        }
        if right < len && heap[right].0 > heap[largest].0 {
            largest = right;
        }
        if largest != idx {
            heap.swap(idx, largest);
            idx = largest;
        } else {
            break;
        }
    }
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

/// Compute core distances using an existing BallTree, also returning nearest neighbors.
/// Avoids rebuilding the tree when it's already been constructed for MST.
pub fn compute_core_distances_with_balltree(
    tree: &BallTree,
    data: &ArrayView2<f64>,
    min_samples: usize,
) -> (Array1<f64>, Vec<usize>) {
    let n = data.nrows();
    let k = min_samples.min(n);
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

/// Compute core distances using an existing BoundedKdTree.
pub fn compute_core_distances_with_bounded_kdtree(
    tree: &BoundedKdTree,
    data: &ArrayView2<f64>,
    min_samples: usize,
) -> (Array1<f64>, Vec<usize>) {
    let n = data.nrows();
    let k = min_samples.min(n);
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
        let (cd_kd, _) = compute_core_distances_bounded_kdtree_with_nn(&data.view(), 3);
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
