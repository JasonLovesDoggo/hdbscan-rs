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

/// Compute core distances and ALL k-1 nearest neighbor indices for each point.
/// Returns (core_distances, all_knn_flat) where all_knn_flat[i*k_per_point..][..k_per_point]
/// are point i's k-1 neighbor indices. k_per_point = min_samples.min(n) - 1.
pub fn compute_core_distances_with_all_nn(
    data: &ArrayView2<f64>,
    metric: &Metric,
    min_samples: usize,
) -> (Array1<f64>, Vec<usize>, usize) {
    let n = data.nrows();
    let k = min_samples.min(n);
    let k_neighbors = k.saturating_sub(1); // exclude self

    let dim = data.ncols();
    match metric {
        Metric::Euclidean if dim <= KDTREE_KNN_MAX_DIM => {
            let tree = BoundedKdTree::build(data);
            let (cd, all_nn) = compute_core_distances_tree_all_nn(&tree, data, k);
            (cd, all_nn, k_neighbors)
        }
        Metric::Euclidean if dim <= BALLTREE_KNN_MAX_DIM => {
            let tree = BallTree::build(data);
            let (cd, all_nn) = compute_core_distances_tree_all_nn(&tree, data, k);
            (cd, all_nn, k_neighbors)
        }
        _ => {
            // Fallback: just use nearest neighbor
            let (cd, nn) = compute_core_distances_with_nn(data, metric, min_samples);
            let mut all = vec![0usize; n * k_neighbors];
            for i in 0..n {
                if k_neighbors > 0 {
                    all[i * k_neighbors] = nn[i];
                }
            }
            (cd, all, k_neighbors)
        }
    }
}

/// Core kNN query engine: runs parallel tree queries and extracts results.
///
/// When `return_all_neighbors` is true, returns all k-1 neighbor indices per point
/// in a flat vec (for KNN pre-merging). Otherwise returns only the nearest neighbor.
fn knn_query_tree<T: CoreDistQuery + Sync>(
    tree: &T,
    data: &ArrayView2<f64>,
    k: usize,
    return_all_neighbors: bool,
) -> (Array1<f64>, Vec<usize>) {
    let n = data.nrows();
    let dim = data.ncols();
    let k_neighbors = k.saturating_sub(1);
    let out_per_point = if return_all_neighbors { k_neighbors } else { 1 };
    let mut core_distances = Array1::zeros(n);
    let mut nn_out = vec![0usize; n * out_per_point];

    let data_contiguous = data.as_standard_layout();
    let data_slice = data_contiguous.as_slice().unwrap();

    #[cfg(not(target_arch = "wasm32"))]
    let n_threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1)
        .min(n);
    #[cfg(target_arch = "wasm32")]
    let n_threads = 1usize;

    if n_threads <= 1 || n < 256 {
        let mut heap = crate::knn_heap::KnnHeap::new(k);
        let mut nbuf = vec![0usize; k_neighbors];
        for i in 0..n {
            heap.clear();
            let query = &data_slice[i * dim..(i + 1) * dim];
            tree.query_core_dist_reuse(query, k, i, &mut heap);
            core_distances[i] = heap.max_dist_sq().sqrt();
            if return_all_neighbors {
                let written = heap.all_neighbors(i, &mut nbuf);
                nn_out[i * out_per_point..i * out_per_point + written]
                    .copy_from_slice(&nbuf[..written]);
            } else {
                nn_out[i] = heap.nearest_non_self(i);
            }
        }
    } else {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let cd_ptr = core_distances.as_slice_mut().unwrap().as_mut_ptr();
            let nn_ptr = nn_out.as_mut_ptr();
            let chunk_size = n.div_ceil(n_threads);

            // SAFETY: each thread writes to disjoint index ranges [start..end)
            // and only reads shared tree + data_slice (immutable).
            std::thread::scope(|s| {
                for t in 0..n_threads {
                    let start = t * chunk_size;
                    let end = (start + chunk_size).min(n);
                    if start >= end {
                        continue;
                    }
                    let cd_s = SendPtr(cd_ptr);
                    let nn_s = SendPtr(nn_ptr);

                    s.spawn(move || {
                        let cd = cd_s;
                        let nn = nn_s;
                        let mut heap = crate::knn_heap::KnnHeap::new(k);
                        let mut nbuf = vec![0usize; k_neighbors];
                        for i in start..end {
                            heap.clear();
                            let query = &data_slice[i * dim..(i + 1) * dim];
                            tree.query_core_dist_reuse(query, k, i, &mut heap);
                            unsafe {
                                *cd.0.add(i) = heap.max_dist_sq().sqrt();
                            }
                            if return_all_neighbors {
                                let written = heap.all_neighbors(i, &mut nbuf);
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        nbuf.as_ptr(),
                                        nn.0.add(i * out_per_point),
                                        written,
                                    );
                                }
                            } else {
                                unsafe {
                                    *nn.0.add(i) = heap.nearest_non_self(i);
                                }
                            }
                        }
                    });
                }
            });
        }
    }

    (core_distances, nn_out)
}

/// Compute core distances and nearest non-self neighbor using a tree.
fn compute_core_distances_tree<T: CoreDistQuery + Sync>(
    tree: &T,
    data: &ArrayView2<f64>,
    k: usize,
) -> (Array1<f64>, Vec<usize>) {
    knn_query_tree(tree, data, k, false)
}

/// Compute core distances and ALL k-1 neighbors using a tree.
fn compute_core_distances_tree_all_nn<T: CoreDistQuery + Sync>(
    tree: &T,
    data: &ArrayView2<f64>,
    k: usize,
) -> (Array1<f64>, Vec<usize>) {
    knn_query_tree(tree, data, k, true)
}

/// Wrapper to send raw pointers across thread boundaries.
/// SAFETY: caller must ensure disjoint access patterns.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy)]
pub struct SendPtr<T>(pub *mut T);
#[cfg(not(target_arch = "wasm32"))]
unsafe impl<T> Send for SendPtr<T> {}
#[cfg(not(target_arch = "wasm32"))]
unsafe impl<T> Sync for SendPtr<T> {}

/// Trait for tree structures that can answer core distance queries.
pub trait CoreDistQuery {
    fn query_core_dist(&self, query: &[f64], k: usize, self_idx: usize) -> (f64, usize);
    /// kNN search using a pre-allocated, pre-cleared heap (avoids per-query allocation).
    fn query_core_dist_reuse(
        &self,
        query: &[f64],
        k: usize,
        self_idx: usize,
        heap: &mut crate::knn_heap::KnnHeap,
    );
}

impl CoreDistQuery for BoundedKdTree {
    fn query_core_dist(&self, query: &[f64], k: usize, self_idx: usize) -> (f64, usize) {
        self.query_core_dist(query, k, self_idx)
    }
    fn query_core_dist_reuse(
        &self,
        query: &[f64],
        _k: usize,
        _self_idx: usize,
        heap: &mut crate::knn_heap::KnnHeap,
    ) {
        if !self.nodes.is_empty() {
            self.knn_recursive_pub(0, query, heap);
        }
    }
}

impl CoreDistQuery for BallTree {
    fn query_core_dist(&self, query: &[f64], k: usize, self_idx: usize) -> (f64, usize) {
        self.query_core_dist(query, k, self_idx)
    }
    fn query_core_dist_reuse(
        &self,
        query: &[f64],
        _k: usize,
        _self_idx: usize,
        heap: &mut crate::knn_heap::KnnHeap,
    ) {
        if !self.nodes.is_empty() {
            let mut sqrt_max_dist = f64::INFINITY;
            let root_centroid_dist_sq =
                crate::simd_distance::squared_euclidean_simd(query, self.centroid(0));
            self.knn_recursive_pub(0, query, heap, &mut sqrt_max_dist, root_centroid_dist_sq);
        }
    }
}

/// Upper-triangle brute-force Euclidean kNN using n simultaneous heaps.
/// Computes each pairwise distance once (instead of twice) by processing
/// both (i,j) and (j,i) in a single pass. Uses ~2x less total FLOPs.
pub fn compute_core_distances_brute_upper_triangle(
    data: &ArrayView2<f64>,
    min_samples: usize,
) -> (Array1<f64>, Vec<usize>) {
    use crate::knn_heap::KnnHeap;

    let n = data.nrows();
    let dim = data.ncols();
    let k = min_samples.min(n);
    let heap_k = k.saturating_sub(1);

    let data_contiguous = data.as_standard_layout();
    let data_slice = data_contiguous.as_slice().unwrap();

    let mut core_distances = Array1::zeros(n);
    let mut nn_indices = vec![0usize; n];

    if heap_k == 0 {
        return (core_distances, nn_indices);
    }

    let mut heaps: Vec<KnnHeap> = (0..n).map(|_| KnnHeap::new(heap_k)).collect();

    for i in 0..n {
        for j in (i + 1)..n {
            let d_sq = crate::simd_distance::squared_euclidean_flat(data_slice, i, j, dim);
            heaps[i].push(d_sq, j);
            heaps[j].push(d_sq, i);
        }
    }

    for i in 0..n {
        core_distances[i] = heaps[i].max_dist_sq().sqrt();
        nn_indices[i] = heaps[i].nearest_non_self(i);
    }

    (core_distances, nn_indices)
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

    let mut heap = KnnHeap::new(k);
    for i in 0..n {
        heap.clear();
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
        let mut dists: Vec<(OrderedFloat<f64>, usize)> =
            (0..n).map(|j| (OrderedFloat(data[[i, j]]), j)).collect();
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

/// Compute core distances and ALL k-1 neighbors using an existing BallTree.
pub fn compute_core_distances_all_nn_with_balltree(
    tree: &BallTree,
    data: &ArrayView2<f64>,
    min_samples: usize,
) -> (Array1<f64>, Vec<usize>, usize) {
    let k = min_samples.min(data.nrows());
    let k_neighbors = k.saturating_sub(1);
    let (cd, all_nn) = compute_core_distances_tree_all_nn(tree, data, k);
    (cd, all_nn, k_neighbors)
}

/// Compute core distances and ALL k-1 neighbors using an existing BoundedKdTree.
pub fn compute_core_distances_all_nn_with_bounded_kdtree(
    tree: &BoundedKdTree,
    data: &ArrayView2<f64>,
    min_samples: usize,
) -> (Array1<f64>, Vec<usize>, usize) {
    let k = min_samples.min(data.nrows());
    let k_neighbors = k.saturating_sub(1);
    let (cd, all_nn) = compute_core_distances_tree_all_nn(tree, data, k);
    (cd, all_nn, k_neighbors)
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
        let (cd_brute, _) =
            compute_core_distances_brute_with_nn(&data.view(), &Metric::Euclidean, 3);
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
