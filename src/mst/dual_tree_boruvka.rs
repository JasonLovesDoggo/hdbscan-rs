//! Dual-tree Boruvka MST for Euclidean mutual reachability distance.
//!
//! Achieves O(n log n) expected time by traversing pairs of spatial tree nodes
//! and pruning entire subtree pairs when no cross-component edge can improve
//! the current best.
//!
//! Generic over tree type: works with both BoundedKdTree (low dimensions)
//! and BallTree (medium-to-high dimensions).
//!
//! Key optimizations:
//! - Per-node component caching for O(1) same-component pruning
//! - Per-node min core distance for tighter MR lower bounds
//! - Lazy sqrt: avoid sqrt when core distances dominate
//! - Closer-child-first traversal for tighter bounds earlier
//! - Core distance shortcut: skip points whose core distance exceeds component best

use crate::spatial_tree::{SpatialNode, SpatialTree};
use crate::types::MstEdge;
use crate::union_find::UnionFind;
use ndarray::ArrayView1;

/// Per-component nearest cross-component neighbor.
/// Stores squared MR distance to avoid sqrt in the inner loop.
/// mr_dist is computed lazily (sqrt of mr_dist_sq) only at edge collection time.
#[derive(Clone, Copy)]
struct ComponentBest {
    mr_dist_sq: f64,
    from: usize,
    to: usize,
}

/// Build MST using dual-tree Boruvka algorithm.
/// Generic over any SpatialTree implementation.
/// Build MST using dual-tree Boruvka algorithm.
/// Generic over any SpatialTree implementation.
///
/// `all_nn_flat`: optional flat array of ALL k neighbors per point. Layout: point i's neighbors
/// are at `all_nn_flat[i*k_neighbors..(i+1)*k_neighbors]`. If provided, enables richer KNN
/// pre-merging that reduces initial component count by 30-60%.
/// `k_neighbors`: number of neighbors per point in all_nn_flat.
/// `nn_indices`: fallback single nearest neighbor per point (used if all_nn_flat is None).
pub fn dual_tree_boruvka_mst<T: SpatialTree + Sync>(
    tree: &T,
    core_distances: &ArrayView1<f64>,
    alpha: f64,
    nn_indices: Option<&[usize]>,
) -> Vec<MstEdge> {
    dual_tree_boruvka_mst_full(tree, core_distances, alpha, nn_indices, None, 0)
}

/// Full version with all-kNN pre-merging support.
pub fn dual_tree_boruvka_mst_full<T: SpatialTree + Sync>(
    tree: &T,
    core_distances: &ArrayView1<f64>,
    alpha: f64,
    nn_indices: Option<&[usize]>,
    all_nn_flat: Option<&[usize]>,
    k_neighbors: usize,
) -> Vec<MstEdge> {
    let n = tree.n();
    if n <= 1 {
        return vec![];
    }

    let core_dists = core_distances
        .as_slice()
        .expect("core distances contiguous");

    // Precompute squared core distances to avoid repeated multiplications in inner loop
    let core_dists_sq: Vec<f64> = core_dists.iter().map(|&d| d * d).collect();

    let mut uf = UnionFind::new(n);
    let mut edges = Vec::with_capacity(n - 1);
    let mut n_components = n;

    let mut component_best: Vec<ComponentBest> = vec![
        ComponentBest {
            mr_dist_sq: f64::INFINITY,
            from: 0,
            to: 0,
        };
        n
    ];

    // KNN pre-merging: seed from kNN neighbors AND merge components.
    // This is the fast-hdbscan approach: instead of just setting initial bounds,
    // actually merge components via union-find before the first tree traversal.
    // Reduces initial component count by 30-60%, enabling much more same-component pruning.
    {
        // Inline closure to process a single kNN edge
        let mut process_edge = |i: usize, j: usize| {
            if j == i || j >= n {
                return;
            }
            let core_max_sq = f64::max(core_dists_sq[i], core_dists_sq[j]);
            let d_sq = tree.dist_sq(i, j);
            let mr_sq = if alpha == 1.0 {
                f64::max(core_max_sq, d_sq)
            } else {
                let dist = d_sq.sqrt();
                let scaled = dist / alpha;
                let core_max = core_max_sq.sqrt();
                let mr = f64::max(core_max, scaled);
                mr * mr
            };
            if mr_sq < component_best[i].mr_dist_sq {
                component_best[i] = ComponentBest {
                    mr_dist_sq: mr_sq,
                    from: i,
                    to: j,
                };
            }
            if mr_sq < component_best[j].mr_dist_sq {
                component_best[j] = ComponentBest {
                    mr_dist_sq: mr_sq,
                    from: j,
                    to: i,
                };
            }
        };

        // Use all k neighbors if available, otherwise fall back to single nearest
        if let Some(all_nn) = all_nn_flat {
            for i in 0..n {
                let base = i * k_neighbors;
                for ki in 0..k_neighbors {
                    let j = all_nn[base + ki];
                    process_edge(i, j);
                }
            }
        } else if let Some(nn) = nn_indices {
            for i in 0..n {
                process_edge(i, nn[i]);
            }
        }

        // Boruvka-style merge using kNN edges: for each component, merge cheapest edge.
        // Repeat until no more merges possible from kNN data alone.
        loop {
            let mut knn_merge_edges: Vec<(f64, usize, usize)> = Vec::new();
            for i in 0..n {
                if uf.find(i) != i {
                    continue;
                }
                let best = &component_best[i];
                if best.mr_dist_sq < f64::INFINITY {
                    let ca = uf.find(best.from);
                    let cb = uf.find(best.to);
                    if ca != cb {
                        knn_merge_edges.push((best.mr_dist_sq, best.from, best.to));
                    }
                }
            }
            if knn_merge_edges.is_empty() {
                break;
            }
            knn_merge_edges.sort_by(|a, b| {
                a.0.partial_cmp(&b.0)
                    .unwrap()
                    .then_with(|| a.1.cmp(&b.1))
                    .then_with(|| a.2.cmp(&b.2))
            });

            let mut merged_any = false;
            for &(mr_sq, from, to) in &knn_merge_edges {
                let ca = uf.find(from);
                let cb = uf.find(to);
                if ca != cb {
                    edges.push(MstEdge {
                        u: from,
                        v: to,
                        weight: mr_sq.sqrt(),
                    });
                    uf.union(from, to);
                    n_components -= 1;
                    merged_any = true;
                }
            }
            if !merged_any {
                break;
            }

            // Re-propagate component_best after merging: some edges may now be
            // cross-component with the new component structure
            for i in 0..n {
                let comp = uf.find(i);
                if comp != i {
                    if component_best[i].mr_dist_sq < component_best[comp].mr_dist_sq {
                        let best = component_best[i];
                        let cf = uf.find(best.from);
                        let ct = uf.find(best.to);
                        if cf != ct {
                            component_best[comp] = best;
                        }
                    }
                } else {
                    // Check if our best edge is now same-component
                    let best = &component_best[i];
                    if best.mr_dist_sq < f64::INFINITY {
                        let cf = uf.find(best.from);
                        let ct = uf.find(best.to);
                        if cf == ct {
                            component_best[i].mr_dist_sq = f64::INFINITY;
                        }
                    }
                }
            }
        }
    }

    // Precompute minimum core distance per tree node (static, computed once)
    let mut node_min_core = vec![f64::INFINITY; tree.nodes().len()];
    precompute_min_core(tree, 0, core_dists, &mut node_min_core);

    // Precompute squared min core distances for fast pruning
    let node_min_core_sq: Vec<f64> = node_min_core.iter().map(|&d| d * d).collect();

    // Per-node component cache (recomputed each round)
    let mut node_component = vec![usize::MAX; tree.nodes().len()];

    // Per-point component cache (avoids repeated uf.find() during traversal)
    let mut point_component = vec![0usize; n];

    let mut merge_edges: Vec<(f64, usize, usize)> = Vec::new();
    while n_components > 1 {
        // Cache component IDs for all points (avoids repeated uf.find() in traversal)
        for i in 0..n {
            point_component[i] = uf.find(i);
        }

        // Invalidate same-component edges and propagate valid bounds to roots.
        for i in 0..n {
            if point_component[i] != i {
                // Not a component root — propagate to root if better
                let comp = point_component[i];
                if component_best[i].mr_dist_sq < component_best[comp].mr_dist_sq {
                    let best = component_best[i];
                    let comp_from = point_component[best.from];
                    let comp_to = point_component[best.to];
                    if comp_from != comp_to {
                        component_best[comp] = best;
                    }
                }
                continue;
            }
            let best = &component_best[i];
            if best.mr_dist_sq < f64::INFINITY {
                let comp_from = point_component[best.from];
                let comp_to = point_component[best.to];
                if comp_from == comp_to {
                    component_best[i].mr_dist_sq = f64::INFINITY;
                }
            }
        }

        // Compute per-node component labels for O(1) same-component pruning.
        // With KNN pre-merging, components exist from round 0, so always compute.
        compute_node_components_cached(tree, 0, &point_component, &mut node_component);

        // Dual-tree traversal: split query tree at root for parallel processing.
        // Each thread gets its own component_best clone and processes a query subtree
        // against the full reference tree. Results are merged after all threads join.
        if !tree.nodes().is_empty() {
            let nodes = tree.nodes();
            let root = &nodes[0];
            #[cfg(not(target_arch = "wasm32"))]
            let n_threads = std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(1);
            #[cfg(target_arch = "wasm32")]
            let n_threads = 1usize;

            if n_threads <= 1 || n < 256 || root.is_leaf() {
                // Single-threaded path
                dual_tree_search(
                    tree,
                    0,
                    0,
                    core_dists,
                    &core_dists_sq,
                    &node_min_core_sq,
                    &point_component,
                    &mut component_best,
                    &node_component,
                    alpha,
                );
            } else {
                #[cfg(not(target_arch = "wasm32"))]
                {
                    // Adaptive parallelism: scale subtree count based on component density.
                    // With many components (early rounds), more subtrees = more parallel work.
                    // With few components (late rounds), pruning is effective so less parallelism needed.
                    let avg_points_per_component = n / n_components.max(1);
                    let target_subtrees = if avg_points_per_component < 10 {
                        // Many small components: split aggressively
                        (n_threads * 2).min(16)
                    } else if avg_points_per_component < 100 {
                        n_threads.min(8)
                    } else {
                        // Few large components: less splitting, good pruning
                        n_threads.min(4)
                    };
                    let mut query_roots = Vec::new();
                    collect_query_subtrees(tree, 0, 0, target_subtrees, &mut query_roots);

                    let cb_ref = &component_best;
                    let cd_ref: &[f64] = core_dists;
                    let cdsq_ref: &[f64] = &core_dists_sq;
                    let nmcsq_ref: &[f64] = &node_min_core_sq;
                    let pc_ref: &[usize] = &point_component;
                    let nc_ref: &[usize] = &node_component;

                    let thread_bests: Vec<Vec<ComponentBest>> = std::thread::scope(|s| {
                        let handles: Vec<_> = query_roots
                            .iter()
                            .map(|&qnode| {
                                let mut local_best = cb_ref.clone();
                                s.spawn(move || {
                                    dual_tree_search(
                                        tree,
                                        qnode,
                                        0,
                                        cd_ref,
                                        cdsq_ref,
                                        nmcsq_ref,
                                        pc_ref,
                                        &mut local_best,
                                        nc_ref,
                                        alpha,
                                    );
                                    local_best
                                })
                            })
                            .collect();
                        handles.into_iter().map(|h| h.join().unwrap()).collect()
                    });

                    // Merge: for each component, take the best across all threads
                    for thread_best in &thread_bests {
                        for i in 0..n {
                            if thread_best[i].mr_dist_sq < component_best[i].mr_dist_sq {
                                component_best[i] = thread_best[i];
                            }
                        }
                    }
                }
            }
        }

        // Collect and merge cheapest edges
        let mut merged_any = false;
        merge_edges.clear();

        for i in 0..n {
            if uf.find(i) != i {
                continue;
            }
            let best = &component_best[i];
            if best.mr_dist_sq < f64::INFINITY {
                merge_edges.push((best.mr_dist_sq.sqrt(), best.from, best.to));
            }
        }

        // Sort for deterministic merging
        merge_edges.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap()
                .then_with(|| a.1.cmp(&b.1))
                .then_with(|| a.2.cmp(&b.2))
        });

        for &(weight, from, to) in &merge_edges {
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
            break;
        }
    }

    edges
}

fn precompute_min_core<T: SpatialTree>(
    tree: &T,
    node_idx: usize,
    core_dists: &[f64],
    node_min_core: &mut [f64],
) {
    let nodes = tree.nodes();
    if node_idx >= nodes.len() {
        return;
    }

    let node = &nodes[node_idx];

    if node.is_leaf() {
        let mut min_core = f64::INFINITY;
        let sorted = tree.sorted_indices();
        for &idx in &sorted[node.idx_start()..node.idx_end()] {
            if core_dists[idx] < min_core {
                min_core = core_dists[idx];
            }
        }
        node_min_core[node_idx] = min_core;
    } else {
        let mut min_core = f64::INFINITY;
        let left = node.left();
        let right = node.right();
        if left < nodes.len() {
            precompute_min_core(tree, left, core_dists, node_min_core);
            min_core = f64::min(min_core, node_min_core[left]);
        }
        if right < nodes.len() {
            precompute_min_core(tree, right, core_dists, node_min_core);
            min_core = f64::min(min_core, node_min_core[right]);
        }
        node_min_core[node_idx] = min_core;
    }
}

/// Compute per-node component labels using cached point components.
/// If all points in a node share the same component, store that component ID.
/// Otherwise store usize::MAX (mixed).
fn compute_node_components_cached<T: SpatialTree>(
    tree: &T,
    node_idx: usize,
    point_component: &[usize],
    node_component: &mut [usize],
) -> usize {
    let nodes = tree.nodes();
    if node_idx >= nodes.len() {
        return usize::MAX;
    }

    let node = &nodes[node_idx];
    let sorted = tree.sorted_indices();

    if node.is_leaf() {
        let first = point_component[sorted[node.idx_start()]];
        let all_same = sorted[node.idx_start()..node.idx_end()]
            .iter()
            .all(|&idx| point_component[idx] == first);
        let comp = if all_same { first } else { usize::MAX };
        node_component[node_idx] = comp;
        comp
    } else {
        let left = node.left();
        let right = node.right();
        let left_comp = if left < nodes.len() {
            compute_node_components_cached(tree, left, point_component, node_component)
        } else {
            usize::MAX
        };
        // Short-circuit: if left subtree has mixed components, parent is also mixed
        let comp = if left_comp == usize::MAX {
            // Still need to recurse right so its node_component values are set
            if right < nodes.len() {
                compute_node_components_cached(tree, right, point_component, node_component);
            }
            usize::MAX
        } else {
            let right_comp = if right < nodes.len() {
                compute_node_components_cached(tree, right, point_component, node_component)
            } else {
                usize::MAX
            };
            if left_comp == right_comp {
                left_comp
            } else {
                usize::MAX
            }
        };
        node_component[node_idx] = comp;
        comp
    }
}

/// Core dual-tree traversal with per-component tracking.
#[allow(clippy::too_many_arguments, clippy::only_used_in_recursion)]
fn dual_tree_search<T: SpatialTree>(
    tree: &T,
    query_node: usize,
    ref_node: usize,
    core_dists: &[f64],
    core_dists_sq: &[f64],
    node_min_core_sq: &[f64],
    point_component: &[usize],
    component_best: &mut [ComponentBest],
    node_component: &[usize],
    alpha: f64,
) {
    let nodes = tree.nodes();
    if query_node >= nodes.len() || ref_node >= nodes.len() {
        return;
    }

    // === Pruning 1: O(1) same-component check via cached node components ===
    let q_comp = node_component[query_node];
    let r_comp = node_component[ref_node];
    if q_comp != usize::MAX && q_comp == r_comp {
        return;
    }

    // === Pruning 2: MR distance lower bound (squared) ===
    let min_dist_sq = tree.min_dist_sq_node_to_node(query_node, ref_node);
    let min_core_max_sq = f64::max(node_min_core_sq[query_node], node_min_core_sq[ref_node]);
    // MR_lower² = max(min_core_max², min_dist_sq) when alpha == 1.0
    let mr_lower_sq = if alpha == 1.0 {
        f64::max(min_core_max_sq, min_dist_sq)
    } else {
        let min_dist = min_dist_sq.sqrt();
        let min_core_max = min_core_max_sq.sqrt();
        let mr_lower = f64::max(min_core_max, min_dist / alpha);
        mr_lower * mr_lower
    };

    // Check if this lower bound can improve ANY component in the query node.
    let q_node = &nodes[query_node];
    let sorted = tree.sorted_indices();
    let mut can_prune = true;
    for &idx in &sorted[q_node.idx_start()..q_node.idx_end()] {
        let comp = point_component[idx];
        if mr_lower_sq < component_best[comp].mr_dist_sq {
            can_prune = false;
            break;
        }
    }
    if can_prune {
        return;
    }

    let r_node = &nodes[ref_node];

    // === Base case: both leaves ===
    if q_node.is_leaf() && r_node.is_leaf() {
        let dim = tree.dim();

        // Use tree-ordered data for sequential memory access when available.
        // Prefer f32 copy for half bandwidth (distances accumulated to f64).
        let tree_ordered = tree.tree_data();
        let tree_ordered_f32 = tree.tree_data_f32();
        let orig_data = tree.data();

        for pos_q in q_node.idx_start()..q_node.idx_end() {
            let qi = sorted[pos_q];
            let comp_q = point_component[qi];
            let core_q_sq = core_dists_sq[qi];
            let mut best_q_sq = component_best[comp_q].mr_dist_sq;

            // Core distance shortcut: can't beat current best (squared comparison)
            if core_q_sq >= best_q_sq {
                continue;
            }

            for pos_r in r_node.idx_start()..r_node.idx_end() {
                let ri = sorted[pos_r];
                let comp_r = point_component[ri];
                if comp_q == comp_r {
                    continue;
                }

                let core_max_sq = f64::max(core_q_sq, core_dists_sq[ri]);

                // If core_max² already exceeds both component bests, skip distance
                let best_r_sq = component_best[comp_r].mr_dist_sq;
                if core_max_sq >= best_q_sq && core_max_sq >= best_r_sq {
                    continue;
                }

                // Use f32 tree data for half bandwidth in higher dims (>= 8) where
                // memory bandwidth matters. Low dims (2D) don't benefit and f32
                // precision can change tie-breaking.
                let d_sq = if dim >= 8 {
                    if let Some(td32) = tree_ordered_f32 {
                        crate::simd_distance::squared_euclidean_flat_f32(td32, pos_q, pos_r, dim)
                    } else if let Some(td) = tree_ordered {
                        crate::simd_distance::squared_euclidean_flat(td, pos_q, pos_r, dim)
                    } else {
                        crate::simd_distance::squared_euclidean_flat(orig_data, qi, ri, dim)
                    }
                } else if let Some(td) = tree_ordered {
                    crate::simd_distance::squared_euclidean_flat(td, pos_q, pos_r, dim)
                } else {
                    crate::simd_distance::squared_euclidean_flat(orig_data, qi, ri, dim)
                };
                // MR² = max(core_max², d²) when alpha == 1.0
                let mr_sq = if alpha == 1.0 {
                    f64::max(core_max_sq, d_sq)
                } else {
                    let dist = d_sq.sqrt();
                    let scaled = dist / alpha;
                    let core_max = core_max_sq.sqrt();
                    let mr = f64::max(core_max, scaled);
                    mr * mr
                };

                if mr_sq < component_best[comp_q].mr_dist_sq {
                    component_best[comp_q] = ComponentBest {
                        mr_dist_sq: mr_sq,
                        from: qi,
                        to: ri,
                    };
                    best_q_sq = mr_sq; // Tighten bound for subsequent ri iterations
                }
                if mr_sq < component_best[comp_r].mr_dist_sq {
                    component_best[comp_r] = ComponentBest {
                        mr_dist_sq: mr_sq,
                        from: ri,
                        to: qi,
                    };
                }
            }
        }
        return;
    }

    // === Recursive case ===
    // Split the larger node. Visit closer child first for better pruning.
    if q_node.is_leaf() || (!r_node.is_leaf() && r_node.count() > q_node.count()) {
        // Split the reference node
        let (first, second) = closer_child_first(tree, query_node, ref_node);

        if first < nodes.len() {
            dual_tree_search(
                tree,
                query_node,
                first,
                core_dists,
                core_dists_sq,
                node_min_core_sq,
                point_component,
                component_best,
                node_component,
                alpha,
            );
        }
        if second < nodes.len() {
            dual_tree_search(
                tree,
                query_node,
                second,
                core_dists,
                core_dists_sq,
                node_min_core_sq,
                point_component,
                component_best,
                node_component,
                alpha,
            );
        }
    } else {
        // Split the query node
        let q_left = q_node.left();
        let q_right = q_node.right();
        if q_left < nodes.len() {
            dual_tree_search(
                tree,
                q_left,
                ref_node,
                core_dists,
                core_dists_sq,
                node_min_core_sq,
                point_component,
                component_best,
                node_component,
                alpha,
            );
        }
        if q_right < nodes.len() {
            dual_tree_search(
                tree,
                q_right,
                ref_node,
                core_dists,
                core_dists_sq,
                node_min_core_sq,
                point_component,
                component_best,
                node_component,
                alpha,
            );
        }
    }
}

/// Collect query subtree roots for parallel processing.
/// Splits the query tree to `target_count` subtrees (approximately).
#[cfg(not(target_arch = "wasm32"))]
fn collect_query_subtrees<T: SpatialTree>(
    tree: &T,
    node_idx: usize,
    depth: usize,
    target_count: usize,
    result: &mut Vec<usize>,
) {
    let nodes = tree.nodes();
    if node_idx >= nodes.len() {
        return;
    }
    let node = &nodes[node_idx];
    // Stop splitting when we have enough subtrees or hit a leaf
    if result.len() + 1 >= target_count || node.is_leaf() || depth >= 4 {
        result.push(node_idx);
        return;
    }
    let left = node.left();
    let right = node.right();
    if left < nodes.len() {
        collect_query_subtrees(tree, left, depth + 1, target_count, result);
    }
    if right < nodes.len() {
        collect_query_subtrees(tree, right, depth + 1, target_count, result);
    }
}

/// Determine which child of ref_node is closer to query_node, return (closer, farther).
fn closer_child_first<T: SpatialTree>(
    tree: &T,
    query_node: usize,
    ref_node: usize,
) -> (usize, usize) {
    let r = &tree.nodes()[ref_node];
    let r_left = r.left();
    let r_right = r.right();
    let nodes = tree.nodes();

    let dist_left = if r_left < nodes.len() {
        tree.min_dist_sq_node_to_node(query_node, r_left)
    } else {
        f64::INFINITY
    };
    let dist_right = if r_right < nodes.len() {
        tree.min_dist_sq_node_to_node(query_node, r_right)
    } else {
        f64::INFINITY
    };

    if dist_left <= dist_right {
        (r_left, r_right)
    } else {
        (r_right, r_left)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ball_tree::BallTree;
    use crate::core_distance;
    use crate::kdtree_bounded::BoundedKdTree;
    use crate::params::Metric;
    use ndarray::array;

    #[test]
    fn test_dual_tree_boruvka_simple() {
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
        let tree = BoundedKdTree::build(&data.view());

        let dtb_edges = dual_tree_boruvka_mst(&tree, &cd.view(), 1.0, None);
        let prim_edges =
            crate::mst::prim::prim_mst(&data.view(), &cd.view(), &Metric::Euclidean, 1.0);

        assert_eq!(dtb_edges.len(), 7);

        let dtb_total: f64 = dtb_edges.iter().map(|e| e.weight).sum();
        let prim_total: f64 = prim_edges.iter().map(|e| e.weight).sum();
        assert!(
            (dtb_total - prim_total).abs() < 1e-10,
            "Weight mismatch: dual_tree={} prim={}",
            dtb_total,
            prim_total,
        );
    }

    #[test]
    fn test_dual_tree_boruvka_200pts() {
        let n = 200;
        let mut data = ndarray::Array2::zeros((n, 2));
        for i in 0..n {
            let cluster = i / 50;
            let cx = (cluster % 2) as f64 * 20.0;
            let cy = (cluster / 2) as f64 * 20.0;
            let seed = (i as u64).wrapping_mul(6364136223846793005).wrapping_add(1);
            let offset_x = ((seed >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 2.0;
            let seed2 = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let offset_y = ((seed2 >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 2.0;
            data[[i, 0]] = cx + offset_x;
            data[[i, 1]] = cy + offset_y;
        }

        let cd = core_distance::compute_core_distances(&data.view(), &Metric::Euclidean, 5);
        let tree = BoundedKdTree::build(&data.view());

        let dtb_edges = dual_tree_boruvka_mst(&tree, &cd.view(), 1.0, None);
        let prim_edges =
            crate::mst::prim::prim_mst(&data.view(), &cd.view(), &Metric::Euclidean, 1.0);

        assert_eq!(dtb_edges.len(), n - 1);

        let dtb_total: f64 = dtb_edges.iter().map(|e| e.weight).sum();
        let prim_total: f64 = prim_edges.iter().map(|e| e.weight).sum();
        assert!(
            (dtb_total - prim_total).abs() < 1e-6,
            "Weight mismatch: dual_tree={:.6} prim={:.6}",
            dtb_total,
            prim_total,
        );
    }

    #[test]
    fn test_ball_tree_boruvka_simple() {
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
        let tree = BallTree::build(&data.view());

        let dtb_edges = dual_tree_boruvka_mst(&tree, &cd.view(), 1.0, None);
        let prim_edges =
            crate::mst::prim::prim_mst(&data.view(), &cd.view(), &Metric::Euclidean, 1.0);

        assert_eq!(dtb_edges.len(), 7);

        let dtb_total: f64 = dtb_edges.iter().map(|e| e.weight).sum();
        let prim_total: f64 = prim_edges.iter().map(|e| e.weight).sum();
        assert!(
            (dtb_total - prim_total).abs() < 1e-10,
            "Ball tree weight mismatch: dual_tree={} prim={}",
            dtb_total,
            prim_total,
        );
    }

    #[test]
    fn test_ball_tree_boruvka_200pts() {
        let n = 200;
        let mut data = ndarray::Array2::zeros((n, 2));
        for i in 0..n {
            let cluster = i / 50;
            let cx = (cluster % 2) as f64 * 20.0;
            let cy = (cluster / 2) as f64 * 20.0;
            let seed = (i as u64).wrapping_mul(6364136223846793005).wrapping_add(1);
            let offset_x = ((seed >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 2.0;
            let seed2 = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let offset_y = ((seed2 >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 2.0;
            data[[i, 0]] = cx + offset_x;
            data[[i, 1]] = cy + offset_y;
        }

        let cd = core_distance::compute_core_distances(&data.view(), &Metric::Euclidean, 5);
        let ball_tree = BallTree::build(&data.view());
        let kd_tree = BoundedKdTree::build(&data.view());

        let ball_edges = dual_tree_boruvka_mst(&ball_tree, &cd.view(), 1.0, None);
        let kd_edges = dual_tree_boruvka_mst(&kd_tree, &cd.view(), 1.0, None);

        assert_eq!(ball_edges.len(), n - 1);

        let ball_total: f64 = ball_edges.iter().map(|e| e.weight).sum();
        let kd_total: f64 = kd_edges.iter().map(|e| e.weight).sum();
        assert!(
            (ball_total - kd_total).abs() < 1e-6,
            "Ball vs KD weight mismatch: ball={:.6} kd={:.6}",
            ball_total,
            kd_total,
        );
    }

    #[test]
    fn test_ball_tree_boruvka_high_dim() {
        // Test in 50D to verify ball tree works in higher dimensions
        let n = 200;
        let dim = 50;
        let mut data = ndarray::Array2::zeros((n, dim));
        for i in 0..n {
            let cluster = i / 50;
            for d in 0..dim {
                let center = (cluster as f64) * 20.0;
                let seed = ((i * dim + d) as u64)
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1);
                let offset = ((seed >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 2.0;
                data[[i, d]] = center + offset;
            }
        }

        let cd = core_distance::compute_core_distances(&data.view(), &Metric::Euclidean, 5);
        let tree = BallTree::build(&data.view());

        let dtb_edges = dual_tree_boruvka_mst(&tree, &cd.view(), 1.0, None);
        let prim_edges =
            crate::mst::prim::prim_mst(&data.view(), &cd.view(), &Metric::Euclidean, 1.0);

        assert_eq!(dtb_edges.len(), n - 1);

        let dtb_total: f64 = dtb_edges.iter().map(|e| e.weight).sum();
        let prim_total: f64 = prim_edges.iter().map(|e| e.weight).sum();
        assert!(
            (dtb_total - prim_total).abs() < 1e-4,
            "50D ball tree weight mismatch: dual_tree={:.6} prim={:.6}",
            dtb_total,
            prim_total,
        );
    }
}
