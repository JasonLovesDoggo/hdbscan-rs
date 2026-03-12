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
    // MR² = max(core_i², core_j², d²) — working entirely in squared space
    // eliminates ALL sqrt calls from the hot loop.
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
        // MR² = max(core_0², core_j², d²)
        let mr_sq = f64::max(f64::max(core_0_sq, core_dists_sq[j]), d_sq);
        min_weight_sq[j] = mr_sq;
        nearest[j] = 0;
    }

    for _ in 0..(n - 1) {
        if active.is_empty() {
            break;
        }

        // Find the active node with minimum squared weight (prefer smaller index on ties).
        // sqrt is monotonic, so min(MR²) ≡ min(MR).
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

        // Only take sqrt here, when creating the actual edge
        edges.push(MstEdge {
            u: nearest[min_idx],
            v: min_idx,
            weight: best_sq.sqrt(),
        });

        // Remove from active set (swap-remove is O(1))
        active.swap_remove(best_pos);

        // Periodically sort active set for cache-friendly data access
        if active.len() > 64 && active.len() % 128 == 0 {
            active.sort_unstable();
        }

        // Update min weights from the newly added node.
        // Safety: all indices in active are < n, and all arrays are length n.
        let core_i_sq = unsafe { *core_dists_sq.get_unchecked(min_idx) };
        for &j in &active {
            let mw_sq_j = unsafe { *min_weight_sq.get_unchecked(j) };
            // Fast check: if core_i² >= current best², can't improve
            if core_i_sq >= mw_sq_j {
                continue;
            }
            // Also check core_j²
            let cd_sq_j = unsafe { *core_dists_sq.get_unchecked(j) };
            if cd_sq_j >= mw_sq_j {
                continue;
            }
            let d_sq = squared_euclidean(data_slice, min_idx, j, dim);
            if d_sq >= mw_sq_j {
                continue;
            }
            // MR² = max(core_i², core_j², d²)
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
