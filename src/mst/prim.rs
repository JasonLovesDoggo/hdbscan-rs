use crate::distance;
use crate::params::Metric;
use crate::types::MstEdge;
use ndarray::{ArrayView1, ArrayView2};

/// Build MST using Prim's algorithm on the mutual reachability graph.
///
/// Mutual reachability distance: d_mreach(a,b) = max(core(a), core(b), d(a,b)/alpha)
///
/// For Euclidean metric, uses an optimized path:
/// - Computes distances on-the-fly (no O(n²) memory allocation)
/// - Works with squared distances to avoid sqrt in the inner loop
/// - Uses raw slice access for maximum throughput
pub fn prim_mst(
    data: &ArrayView2<f64>,
    core_distances: &ArrayView1<f64>,
    metric: &Metric,
    alpha: f64,
) -> Vec<MstEdge> {
    prim_mst_seeded(data, core_distances, metric, alpha, None)
}

pub fn prim_mst_seeded(
    data: &ArrayView2<f64>,
    core_distances: &ArrayView1<f64>,
    metric: &Metric,
    alpha: f64,
    nn_indices: Option<&[usize]>,
) -> Vec<MstEdge> {
    let n = data.nrows();
    if n <= 1 {
        return vec![];
    }

    match metric {
        Metric::Euclidean if alpha == 1.0 => prim_mst_euclidean_fast(data, core_distances, nn_indices),
        Metric::Euclidean => prim_mst_euclidean_alpha(data, core_distances, alpha),
        _ => prim_mst_generic(data, core_distances, metric, alpha),
    }
}

/// Highly optimized Prim's MST for Euclidean metric with alpha=1.0 (the common case).
///
/// Key optimizations:
/// - No O(n²) matrix allocation — distances computed on-the-fly
/// - Raw slice indexing for data access (avoids ndarray view overhead)
/// - Compact active-set tracking (no branch misprediction from in_tree checks)
/// - Early-exit pruning: skip distance computation when core distances dominate
/// - Lazy sqrt: avoid sqrt when core distance is the mutual reachability result
/// - SIMD-accelerated squared Euclidean distance
fn prim_mst_euclidean_fast(
    data: &ArrayView2<f64>,
    core_distances: &ArrayView1<f64>,
    _nn_indices: Option<&[usize]>,
) -> Vec<MstEdge> {
    let n = data.nrows();
    let dim = data.ncols();

    // Get contiguous data slice for fast access
    let data_contiguous = data.as_standard_layout();
    let data_slice = data_contiguous
        .as_slice()
        .expect("data should be contiguous after as_standard_layout");

    let core_dists = core_distances
        .as_slice()
        .expect("core_distances should be contiguous");

    // Precompute squared core distances for fast comparisons.
    let core_dists_sq: Vec<f64> = core_dists.iter().map(|&d| d * d).collect();

    // min_weight_sq[j] = squared mutual reachability distance from j to nearest tree node
    let mut min_weight_sq = vec![f64::INFINITY; n];
    let mut nearest = vec![0usize; n];
    let mut edges = Vec::with_capacity(n - 1);

    // Active set: indices of nodes not yet in the tree
    let mut active: Vec<usize> = (1..n).collect();

    // Initialize from node 0
    let core_0_sq = core_dists_sq[0];
    for &j in &active {
        let d_sq = squared_euclidean(data_slice, 0, j, dim);
        let mr_sq = f64::max(f64::max(core_0_sq, core_dists_sq[j]), d_sq);
        min_weight_sq[j] = mr_sq;
        nearest[j] = 0;
    }

    for _ in 0..(n - 1) {
        if active.is_empty() {
            break;
        }

        // Find the active node with minimum squared weight
        let mut best_pos = 0;
        let mut best_sq = unsafe { *min_weight_sq.get_unchecked(*active.get_unchecked(0)) };
        let mut best_idx = unsafe { *active.get_unchecked(0) };
        for (pos, &j) in active.iter().enumerate().skip(1) {
            let w = unsafe { *min_weight_sq.get_unchecked(j) };
            if w < best_sq || (w == best_sq && j < best_idx) {
                best_sq = w;
                best_pos = pos;
                best_idx = j;
            }
        }

        let min_idx = best_idx;

        edges.push(MstEdge {
            u: nearest[min_idx],
            v: min_idx,
            weight: best_sq.sqrt(),
        });

        active.swap_remove(best_pos);

        // Periodically sort active set for cache-friendly data access
        if active.len() > 64 && active.len() % 128 == 0 {
            active.sort_unstable();
        }

        // Update min weights from the newly added node.
        let core_i_sq = unsafe { *core_dists_sq.get_unchecked(min_idx) };
        for &j in &active {
            let mw_sq_j = unsafe { *min_weight_sq.get_unchecked(j) };
            if core_i_sq >= mw_sq_j {
                continue;
            }
            let cd_sq_j = unsafe { *core_dists_sq.get_unchecked(j) };
            if cd_sq_j >= mw_sq_j {
                continue;
            }
            let d_sq = squared_euclidean(data_slice, min_idx, j, dim);
            if d_sq >= mw_sq_j {
                continue;
            }
            let mr_sq = f64::max(f64::max(core_i_sq, cd_sq_j), d_sq);
            if mr_sq < mw_sq_j {
                unsafe {
                    *min_weight_sq.get_unchecked_mut(j) = mr_sq;
                    *nearest.get_unchecked_mut(j) = min_idx;
                }
            }
        }
    }

    edges
}

/// Optimized Prim's for Euclidean with non-unit alpha.
fn prim_mst_euclidean_alpha(
    data: &ArrayView2<f64>,
    core_distances: &ArrayView1<f64>,
    alpha: f64,
) -> Vec<MstEdge> {
    let n = data.nrows();
    let dim = data.ncols();

    let data_contiguous = data.as_standard_layout();
    let data_slice = data_contiguous
        .as_slice()
        .expect("data should be contiguous");

    let core_dists = core_distances
        .as_slice()
        .expect("core_distances should be contiguous");

    let mut min_weight = vec![f64::INFINITY; n];
    let mut nearest = vec![0usize; n];
    let mut edges = Vec::with_capacity(n - 1);
    let mut active: Vec<usize> = (1..n).collect();

    let core_0 = core_dists[0];
    for &j in &active {
        let d = squared_euclidean(data_slice, 0, j, dim).sqrt();
        let scaled = d / alpha;
        min_weight[j] = f64::max(core_0, f64::max(core_dists[j], scaled));
        nearest[j] = 0;
    }

    for _ in 0..(n - 1) {
        if active.is_empty() {
            break;
        }

        let mut best_pos = 0;
        let mut best_val = min_weight[active[0]];
        let mut best_idx = active[0];
        for (pos, &j) in active.iter().enumerate().skip(1) {
            let w = min_weight[j];
            if w < best_val || (w == best_val && j < best_idx) {
                best_val = w;
                best_pos = pos;
                best_idx = j;
            }
        }

        let min_idx = best_idx;
        edges.push(MstEdge {
            u: nearest[min_idx],
            v: min_idx,
            weight: best_val,
        });
        active.swap_remove(best_pos);

        let core_i = core_dists[min_idx];
        for &j in &active {
            let d = squared_euclidean(data_slice, min_idx, j, dim).sqrt();
            let scaled = d / alpha;
            let mr = f64::max(core_i, f64::max(core_dists[j], scaled));
            if mr < min_weight[j] {
                min_weight[j] = mr;
                nearest[j] = min_idx;
            }
        }
    }

    edges
}

/// Fused core distance + Prim's MST for Euclidean metric with alpha=1.0.
///
/// Computes all pairwise squared distances, extracts core distances via kNN heaps,
/// then runs Prim's using cached O(1) lookups. Uses GEMM (X@X.T) for high-dim
/// or point-by-point SIMD for low-dim distance computation.
pub fn fused_core_and_prim(
    data: &ArrayView2<f64>,
    min_samples: usize,
) -> (ndarray::Array1<f64>, Vec<MstEdge>) {
    let n = data.nrows();
    let dim = data.ncols();
    let k = min_samples.min(n); // k-th nearest including self

    if n <= 1 {
        return (ndarray::Array1::zeros(n), vec![]);
    }

    // Phase 1: Compute pairwise squared distances and extract core distances.
    // For dim >= 16, GEMM (X@X.T) is faster than point-by-point SIMD due to
    // cache-blocked matrix multiply. For lower dims, SIMD is faster.
    let (gram_matrix, norms_sq, core_dists_sq) = if dim >= 16 {
        fused_phase1_gemm(data, n, k)
    } else {
        fused_phase1_simd(data, n, dim, k)
    };

    let core_distances = ndarray::Array1::from_iter(core_dists_sq.iter().map(|&d| d.sqrt()));

    // Phase 2: Prim's MST using cached distances.
    let edges = fused_prim_cached(&gram_matrix, &norms_sq, &core_dists_sq, n);

    (core_distances, edges)
}

/// Phase 1 using GEMM: compute Gram matrix X@X.T and derive distances.
/// dist²(i,j) = ||x_i||² + ||x_j||² - 2*(x_i · x_j)
fn fused_phase1_gemm(
    data: &ArrayView2<f64>,
    n: usize,
    k: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    // Compute Gram matrix (all dot products) via cache-blocked matmul.
    // into_raw_vec() avoids a 200MB copy by consuming the Array2.
    let data_owned = data.to_owned();
    let gram = data_owned.dot(&data_owned.t());
    let gram_slice = gram.as_slice().unwrap();

    // Extract squared norms from diagonal
    let norms_sq: Vec<f64> = (0..n).map(|i| gram_slice[i * n + i]).collect();

    // Extract core distances using a single reusable kNN heap per row.
    // Sequential row access is cache-friendly for the Gram matrix.
    let heap_k = if k > 1 { k - 1 } else { 0 };
    let mut core_dists_sq = vec![0.0f64; n];

    if heap_k > 0 {
        let mut heap = crate::knn_heap::KnnHeap::new(heap_k);
        for i in 0..n {
            heap.clear();
            let ni = norms_sq[i];
            let row_off = i * n;
            for j in 0..n {
                if i == j {
                    continue;
                }
                let d_sq = unsafe {
                    (ni + *norms_sq.get_unchecked(j)
                        - 2.0 * *gram_slice.get_unchecked(row_off + j))
                    .max(0.0)
                };
                heap.push(d_sq, j);
            }
            core_dists_sq[i] = heap.max_dist_sq();
            if core_dists_sq[i] == f64::INFINITY {
                core_dists_sq[i] = 0.0;
            }
        }
    }

    // Consume the Array2 to get its backing Vec without copying
    let gram_vec = gram.into_raw_vec_and_offset().0;
    (gram_vec, norms_sq, core_dists_sq)
}

/// Phase 1 using point-by-point SIMD distance computation.
fn fused_phase1_simd(
    data: &ArrayView2<f64>,
    n: usize,
    dim: usize,
    k: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let data_contiguous = data.as_standard_layout();
    let data_slice = data_contiguous.as_slice().unwrap();

    // Compute squared norms for Prim's phase
    let norms_sq: Vec<f64> = (0..n).map(|i| {
        let off = i * dim;
        let mut s = 0.0;
        for d in 0..dim {
            let v = unsafe { *data_slice.get_unchecked(off + d) };
            s += v * v;
        }
        s
    }).collect();

    // Build gram matrix (dot products) and kNN heaps simultaneously
    let mut gram = vec![0.0f64; n * n];
    let heap_k = if k > 1 { k - 1 } else { 0 };
    let mut heaps: Vec<crate::knn_heap::KnnHeap> =
        (0..n).map(|_| crate::knn_heap::KnnHeap::new(heap_k)).collect();

    // Set diagonal (self dot products)
    for i in 0..n {
        gram[i * n + i] = norms_sq[i];
    }

    for i in 0..n {
        let off_i = i * n;
        let ni = norms_sq[i];
        for j in (i + 1)..n {
            let d_sq = squared_euclidean(data_slice, i, j, dim);
            // dot(i,j) = (norms_sq[i] + norms_sq[j] - d_sq) / 2
            let dot_ij = (ni + norms_sq[j] - d_sq) * 0.5;
            unsafe {
                *gram.get_unchecked_mut(off_i + j) = dot_ij;
                *gram.get_unchecked_mut(j * n + i) = dot_ij;
            }
            if heap_k > 0 {
                heaps[i].push(d_sq, j);
                heaps[j].push(d_sq, i);
            }
        }
    }

    let mut core_dists_sq = vec![0.0f64; n];
    for i in 0..n {
        core_dists_sq[i] = heaps[i].max_dist_sq();
        if core_dists_sq[i] == f64::INFINITY {
            core_dists_sq[i] = 0.0;
        }
    }

    (gram, norms_sq, core_dists_sq)
}

/// Prim's MST using cached Gram matrix for O(1) distance lookups.
/// dist²(i,j) = norms_sq[i] + norms_sq[j] - 2*gram[i*n+j]
fn fused_prim_cached(
    gram: &[f64],
    norms_sq: &[f64],
    core_dists_sq: &[f64],
    n: usize,
) -> Vec<MstEdge> {
    let mut min_weight_sq = vec![f64::INFINITY; n];
    let mut nearest = vec![0usize; n];
    let mut edges = Vec::with_capacity(n - 1);
    let mut active: Vec<usize> = (1..n).collect();

    // Initialize from node 0
    let core_0_sq = core_dists_sq[0];
    let n0 = norms_sq[0];
    for &j in &active {
        let d_sq = (n0 + norms_sq[j] - 2.0 * gram[j]).max(0.0);
        let mr_sq = f64::max(f64::max(core_0_sq, core_dists_sq[j]), d_sq);
        min_weight_sq[j] = mr_sq;
        nearest[j] = 0;
    }

    for _ in 0..(n - 1) {
        if active.is_empty() {
            break;
        }

        // Find minimum
        let mut best_pos = 0;
        let mut best_sq = unsafe { *min_weight_sq.get_unchecked(*active.get_unchecked(0)) };
        let mut best_idx = unsafe { *active.get_unchecked(0) };
        for (pos, &j) in active.iter().enumerate().skip(1) {
            let w = unsafe { *min_weight_sq.get_unchecked(j) };
            if w < best_sq || (w == best_sq && j < best_idx) {
                best_sq = w;
                best_pos = pos;
                best_idx = j;
            }
        }

        let min_idx = best_idx;
        edges.push(MstEdge {
            u: nearest[min_idx],
            v: min_idx,
            weight: best_sq.sqrt(),
        });

        active.swap_remove(best_pos);

        if active.len() > 64 && active.len() % 128 == 0 {
            active.sort_unstable();
        }

        // Update from newly added node using cached distances
        let core_i_sq = core_dists_sq[min_idx];
        let ni = norms_sq[min_idx];
        let row_offset = min_idx * n;
        for &j in &active {
            let mw_sq_j = unsafe { *min_weight_sq.get_unchecked(j) };
            if core_i_sq >= mw_sq_j {
                continue;
            }
            let cd_sq_j = unsafe { *core_dists_sq.get_unchecked(j) };
            if cd_sq_j >= mw_sq_j {
                continue;
            }
            let d_sq = (ni + unsafe { *norms_sq.get_unchecked(j) }
                - 2.0 * unsafe { *gram.get_unchecked(row_offset + j) }).max(0.0);
            if d_sq >= mw_sq_j {
                continue;
            }
            let mr_sq = f64::max(f64::max(core_i_sq, cd_sq_j), d_sq);
            if mr_sq < mw_sq_j {
                unsafe {
                    *min_weight_sq.get_unchecked_mut(j) = mr_sq;
                    *nearest.get_unchecked_mut(j) = min_idx;
                }
            }
        }
    }

    edges
}

/// Compute squared Euclidean distance between points i and j using raw slice access.
#[inline(always)]
fn squared_euclidean(data: &[f64], i: usize, j: usize, dim: usize) -> f64 {
    crate::simd_distance::squared_euclidean_flat(data, i, j, dim)
}


/// Generic Prim's MST for non-Euclidean metrics. Precomputes the full matrix.
fn prim_mst_generic(
    data: &ArrayView2<f64>,
    core_distances: &ArrayView1<f64>,
    metric: &Metric,
    alpha: f64,
) -> Vec<MstEdge> {
    let n = data.nrows();

    let mr_matrix = compute_mutual_reachability_matrix(data, core_distances, metric, alpha);

    let mut in_tree = vec![false; n];
    let mut min_weight = vec![f64::INFINITY; n];
    let mut nearest = vec![0usize; n];
    let mut edges = Vec::with_capacity(n - 1);

    in_tree[0] = true;

    for j in 1..n {
        min_weight[j] = mr_matrix[j];
        nearest[j] = 0;
    }

    for _ in 0..(n - 1) {
        let mut min_val = f64::INFINITY;
        let mut min_idx = 0;
        for j in 0..n {
            if !in_tree[j] && min_weight[j] < min_val {
                min_val = min_weight[j];
                min_idx = j;
            }
        }

        if min_val == f64::INFINITY {
            for j in 0..n {
                if !in_tree[j] {
                    min_idx = j;
                    break;
                }
            }
        }

        edges.push(MstEdge {
            u: nearest[min_idx],
            v: min_idx,
            weight: min_weight[min_idx],
        });

        in_tree[min_idx] = true;

        let row_offset = min_idx * n;
        for j in 0..n {
            if !in_tree[j] {
                let d = mr_matrix[row_offset + j];
                if d < min_weight[j] {
                    min_weight[j] = d;
                    nearest[j] = min_idx;
                }
            }
        }
    }

    edges
}

/// Compute the full mutual reachability distance matrix as a flat Vec.
fn compute_mutual_reachability_matrix(
    data: &ArrayView2<f64>,
    core_distances: &ArrayView1<f64>,
    metric: &Metric,
    alpha: f64,
) -> Vec<f64> {
    let n = data.nrows();
    let mut mr = vec![0.0f64; n * n];

    match metric {
        Metric::Precomputed => {
            for i in 0..n {
                let core_i = core_distances[i];
                for j in (i + 1)..n {
                    let raw = data[[i, j]];
                    let scaled = if alpha != 1.0 { raw / alpha } else { raw };
                    let d = f64::max(core_i, f64::max(core_distances[j], scaled));
                    mr[i * n + j] = d;
                    mr[j * n + i] = d;
                }
            }
        }
        _ => {
            for i in 0..n {
                let row_i = data.row(i);
                let core_i = core_distances[i];
                for j in (i + 1)..n {
                    let raw = distance::compute_distance(&row_i, &data.row(j), metric);
                    let scaled = if alpha != 1.0 { raw / alpha } else { raw };
                    let d = f64::max(core_i, f64::max(core_distances[j], scaled));
                    mr[i * n + j] = d;
                    mr[j * n + i] = d;
                }
            }
        }
    }

    mr
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_prim_simple() {
        let data = array![[0.0], [1.0], [5.0]];
        let core_distances = array![1.0, 1.0, 4.0];
        let edges = prim_mst(
            &data.view(),
            &core_distances.view(),
            &Metric::Euclidean,
            1.0,
        );
        assert_eq!(edges.len(), 2);
    }

    #[test]
    fn test_prim_single_point() {
        let data = array![[0.0]];
        let core_distances = array![0.0];
        let edges = prim_mst(
            &data.view(),
            &core_distances.view(),
            &Metric::Euclidean,
            1.0,
        );
        assert_eq!(edges.len(), 0);
    }
}
