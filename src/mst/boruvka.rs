use crate::kdtree::KdTree;
use crate::params::Metric;
use crate::types::MstEdge;
use crate::union_find::UnionFind;
use ndarray::{ArrayView1, ArrayView2};

/// Build MST using Boruvka's algorithm with KD-tree acceleration.
///
/// For Euclidean metric, achieves approximately O(n log² n) time complexity.
///
/// Key optimizations:
/// - Skip interior points whose core distance exceeds the component's current best
/// - Smart initial search radius to avoid redundant KD-tree queries
/// - Component-level best tracking to prune early
pub fn boruvka_mst(
    data: &ArrayView2<f64>,
    core_distances: &ArrayView1<f64>,
    kdtree: &KdTree,
    _metric: &Metric,
    alpha: f64,
) -> Vec<MstEdge> {
    let n = data.nrows();
    if n <= 1 {
        return vec![];
    }

    let core_dists = core_distances.as_slice().expect("core_distances contiguous");
    let points: Vec<Vec<f64>> = (0..n).map(|i| data.row(i).to_vec()).collect();

    let mut uf = UnionFind::new(n);
    let mut edges = Vec::with_capacity(n - 1);
    let mut n_components = n;

    // Per-point search radius cache: how many KD-tree neighbors to search.
    // Grows as components merge (interior points need to look farther).
    let mut search_k: Vec<usize> = vec![3; n];

    while n_components > 1 {
        // For each component, track the cheapest cross-component edge
        // cheapest[comp_root] = Some((mr_dist, from, to))
        let mut cheapest: Vec<Option<(f64, usize, usize)>> = vec![None; n];

        // Sort points by core distance (ascending) within each component.
        // Points with smallest core distance are most likely to produce
        // the cheapest cross-component edge, so process them first.
        // This allows us to set the component's best early and skip
        // interior points.
        let mut point_order: Vec<usize> = (0..n).collect();
        point_order.sort_unstable_by(|&a, &b| {
            core_dists[a].partial_cmp(&core_dists[b]).unwrap()
        });

        for &i in &point_order {
            let comp_i = uf.find(i);
            let core_i = core_dists[i];

            // Get component's current best MR distance
            let comp_best = cheapest[comp_i].map_or(f64::INFINITY, |(w, _, _)| w);

            // Key pruning: if this point's core distance alone >= component best,
            // then MR(i,j) >= core_i >= comp_best for ALL j. Skip this point.
            if core_i >= comp_best {
                continue;
            }

            // Search KD-tree neighbors with expanding radius.
            let mut best_mr = comp_best;
            let mut k = search_k[i];

            loop {
                let actual_k = k.min(n);
                let neighbors = kdtree.query_knn(&points[i], actual_k);

                let mut found_cross = false;
                let mut all_same_component = true;

                for &(raw_dist, j) in &neighbors {
                    if i == j {
                        continue;
                    }
                    // If raw distance >= best MR, no farther neighbor can improve
                    if raw_dist >= best_mr {
                        found_cross = true; // Effectively found our stopping point
                        break;
                    }

                    let comp_j = uf.find(j);
                    if comp_j == comp_i {
                        continue;
                    }
                    all_same_component = false;

                    // Compute mutual reachability distance
                    let scaled = if alpha != 1.0 { raw_dist / alpha } else { raw_dist };
                    let mr = f64::max(core_i, f64::max(core_dists[j], scaled));

                    if mr < best_mr {
                        best_mr = mr;
                        cheapest[comp_i] = Some((mr, i, j));
                    }
                    found_cross = true;
                }

                // Can we stop searching?
                if found_cross || actual_k >= n {
                    // Cache the search radius for this point
                    search_k[i] = k;
                    break;
                }

                if all_same_component {
                    // All neighbors so far are in the same component.
                    // Double k to look farther.
                    k = (k * 2).min(n);
                } else {
                    break;
                }
            }
        }

        // Merge cheapest edges for each component
        let mut merged_any = false;

        // Collect edges to merge, preferring deterministic ordering
        let mut merge_candidates: Vec<(f64, usize, usize, usize)> = Vec::new(); // (weight, from, to, comp)
        for root in 0..n {
            if uf.find(root) != root {
                continue;
            }
            if let Some((weight, from, to)) = cheapest[root] {
                merge_candidates.push((weight, from, to, root));
            }
        }
        // Sort by weight for deterministic merging
        merge_candidates.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap()
                .then_with(|| a.1.cmp(&b.1))
                .then_with(|| a.2.cmp(&b.2))
        });

        for (weight, from, to, _) in merge_candidates {
            let ca = uf.find(from);
            let cb = uf.find(to);
            if ca != cb {
                edges.push(MstEdge {
                    u: from,
                    v: to,
                    weight,
                });
                uf.union(from, to);
                n_components -= 1;
                merged_any = true;
            }
        }

        if !merged_any {
            // Fallback: brute-force remaining (rare edge case)
            let remaining = brute_force_remaining(&mut uf, data, core_distances, alpha);
            for edge in remaining {
                if n_components <= 1 {
                    break;
                }
                let ca = uf.find(edge.u);
                let cb = uf.find(edge.v);
                if ca != cb {
                    uf.union(edge.u, edge.v);
                    n_components -= 1;
                    edges.push(edge);
                }
            }
            break;
        }
    }

    edges
}

fn brute_force_remaining(
    uf: &mut UnionFind,
    data: &ArrayView2<f64>,
    core_distances: &ArrayView1<f64>,
    alpha: f64,
) -> Vec<MstEdge> {
    let n = data.nrows();
    let mut candidate_edges = Vec::new();

    for i in 0..n {
        for j in (i + 1)..n {
            let ci = uf.find(i);
            let cj = uf.find(j);
            if ci != cj {
                let raw_dist = crate::distance::euclidean::euclidean_distance(
                    &data.row(i),
                    &data.row(j),
                );
                let scaled = if alpha != 1.0 { raw_dist / alpha } else { raw_dist };
                let mr = f64::max(
                    core_distances[i],
                    f64::max(core_distances[j], scaled),
                );
                candidate_edges.push(MstEdge {
                    u: i,
                    v: j,
                    weight: mr,
                });
            }
        }
    }

    candidate_edges.sort_by(|a, b| a.weight.partial_cmp(&b.weight).unwrap());
    candidate_edges
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core_distance;
    use ndarray::array;

    #[test]
    fn test_boruvka_matches_prim() {
        let data = array![
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [0.1, 0.1],
            [10.0, 10.0],
            [10.1, 10.0],
            [10.0, 10.1],
            [10.1, 10.1],
        ];

        let cd = core_distance::compute_core_distances(&data.view(), &Metric::Euclidean, 3);
        let kdtree = KdTree::build(&data.view());

        let boruvka_edges = boruvka_mst(
            &data.view(),
            &cd.view(),
            &kdtree,
            &Metric::Euclidean,
            1.0,
        );

        let prim_edges = crate::mst::prim::prim_mst(
            &data.view(),
            &cd.view(),
            &Metric::Euclidean,
            1.0,
        );

        assert_eq!(boruvka_edges.len(), 7);
        assert_eq!(prim_edges.len(), 7);

        let boruvka_total: f64 = boruvka_edges.iter().map(|e| e.weight).sum();
        let prim_total: f64 = prim_edges.iter().map(|e| e.weight).sum();
        assert!(
            (boruvka_total - prim_total).abs() < 1e-10,
            "Weight mismatch: boruvka={} prim={}",
            boruvka_total,
            prim_total,
        );
    }

    #[test]
    fn test_boruvka_larger() {
        let n = 50;
        let mut data = ndarray::Array2::zeros((n, 2));
        for i in 0..n {
            data[[i, 0]] = (i as f64) * 0.1;
            data[[i, 1]] = ((i * 7) % 13) as f64;
        }

        let cd = core_distance::compute_core_distances(&data.view(), &Metric::Euclidean, 5);
        let kdtree = KdTree::build(&data.view());

        let boruvka_edges = boruvka_mst(
            &data.view(),
            &cd.view(),
            &kdtree,
            &Metric::Euclidean,
            1.0,
        );

        let prim_edges = crate::mst::prim::prim_mst(
            &data.view(),
            &cd.view(),
            &Metric::Euclidean,
            1.0,
        );

        assert_eq!(boruvka_edges.len(), n - 1);

        let boruvka_total: f64 = boruvka_edges.iter().map(|e| e.weight).sum();
        let prim_total: f64 = prim_edges.iter().map(|e| e.weight).sum();
        assert!(
            (boruvka_total - prim_total).abs() < 1e-8,
            "Weight mismatch: boruvka={} prim={}",
            boruvka_total,
            prim_total,
        );
    }
}
