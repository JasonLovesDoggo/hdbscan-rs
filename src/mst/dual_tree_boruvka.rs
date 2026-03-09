//! Dual-tree Boruvka MST for Euclidean mutual reachability distance.
//!
//! Achieves O(n log n) expected time by traversing pairs of KD-tree nodes
//! and pruning entire subtree pairs when no cross-component edge can improve
//! the current best.
//!
//! Key optimizations:
//! - Per-node component caching for O(1) same-component pruning
//! - Per-node min core distance for tighter MR lower bounds
//! - Lazy sqrt: avoid sqrt when core distances dominate
//! - Closer-child-first traversal for tighter bounds earlier
//! - Core distance shortcut: skip points whose core distance exceeds component best

use crate::kdtree_bounded::{BoundedKdTree, NO_CHILD};
use crate::types::MstEdge;
use crate::union_find::UnionFind;
use ndarray::ArrayView1;

/// Per-component nearest cross-component neighbor.
#[derive(Clone, Copy)]
struct ComponentBest {
    mr_dist: f64,
    from: usize,
    to: usize,
}

/// Build MST using dual-tree Boruvka algorithm.
pub fn dual_tree_boruvka_mst(
    tree: &BoundedKdTree,
    core_distances: &ArrayView1<f64>,
    alpha: f64,
) -> Vec<MstEdge> {
    let n = tree.n;
    if n <= 1 {
        return vec![];
    }

    let core_dists = core_distances.as_slice().expect("core distances contiguous");

    let mut uf = UnionFind::new(n);
    let mut edges = Vec::with_capacity(n - 1);
    let mut n_components = n;

    let mut component_best: Vec<ComponentBest> = vec![
        ComponentBest {
            mr_dist: f64::INFINITY,
            from: 0,
            to: 0,
        };
        n
    ];

    // Precompute minimum core distance per tree node (static, computed once)
    let mut node_min_core = vec![f64::INFINITY; tree.nodes.len()];
    precompute_min_core(tree, 0, core_dists, &mut node_min_core);

    // Per-node component cache (recomputed each round)
    let mut node_component = vec![usize::MAX; tree.nodes.len()];

    // Per-point component cache (avoids repeated uf.find() during traversal)
    let mut point_component = vec![0usize; n];

    while n_components > 1 {
        // Reset component bests
        for best in component_best.iter_mut() {
            best.mr_dist = f64::INFINITY;
        }

        // Cache component IDs for all points (avoids repeated uf.find() in traversal)
        for i in 0..n {
            point_component[i] = uf.find(i);
        }

        // Compute per-node component labels for O(1) same-component pruning
        compute_node_components_cached(tree, 0, &point_component, &mut node_component);

        // Dual-tree traversal from root × root
        if !tree.nodes.is_empty() {
            dual_tree_search(
                tree,
                0,
                0,
                core_dists,
                &node_min_core,
                &point_component,
                &mut component_best,
                &node_component,
                alpha,
            );
        }

        // Collect and merge cheapest edges
        let mut merged_any = false;
        let mut merge_edges: Vec<(f64, usize, usize)> = Vec::new();

        for i in 0..n {
            if uf.find(i) != i {
                continue;
            }
            let best = &component_best[i];
            if best.mr_dist < f64::INFINITY {
                merge_edges.push((best.mr_dist, best.from, best.to));
            }
        }

        // Sort for deterministic merging
        merge_edges.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap()
                .then_with(|| a.1.cmp(&b.1))
                .then_with(|| a.2.cmp(&b.2))
        });

        for (weight, from, to) in merge_edges {
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

fn precompute_min_core(
    tree: &BoundedKdTree,
    node_idx: usize,
    core_dists: &[f64],
    node_min_core: &mut [f64],
) {
    if node_idx == NO_CHILD {
        return;
    }

    let node = &tree.nodes[node_idx];

    if node.is_leaf {
        let mut min_core = f64::INFINITY;
        for &idx in &tree.sorted_indices[node.idx_start..node.idx_end] {
            if core_dists[idx] < min_core {
                min_core = core_dists[idx];
            }
        }
        node_min_core[node_idx] = min_core;
    } else {
        let mut min_core = f64::INFINITY;
        if node.left != NO_CHILD {
            precompute_min_core(tree, node.left, core_dists, node_min_core);
            min_core = f64::min(min_core, node_min_core[node.left]);
        }
        if node.right != NO_CHILD {
            precompute_min_core(tree, node.right, core_dists, node_min_core);
            min_core = f64::min(min_core, node_min_core[node.right]);
        }
        node_min_core[node_idx] = min_core;
    }
}

/// Compute per-node component labels using cached point components.
/// If all points in a node share the same component, store that component ID.
/// Otherwise store usize::MAX (mixed).
fn compute_node_components_cached(
    tree: &BoundedKdTree,
    node_idx: usize,
    point_component: &[usize],
    node_component: &mut [usize],
) -> usize {
    if node_idx == NO_CHILD {
        return usize::MAX;
    }

    let node = &tree.nodes[node_idx];

    if node.is_leaf {
        let first = point_component[tree.sorted_indices[node.idx_start]];
        let all_same = tree.sorted_indices[node.idx_start..node.idx_end]
            .iter()
            .all(|&idx| point_component[idx] == first);
        let comp = if all_same { first } else { usize::MAX };
        node_component[node_idx] = comp;
        comp
    } else {
        let left_comp = if node.left != NO_CHILD {
            compute_node_components_cached(tree, node.left, point_component, node_component)
        } else {
            usize::MAX
        };
        let right_comp = if node.right != NO_CHILD {
            compute_node_components_cached(tree, node.right, point_component, node_component)
        } else {
            usize::MAX
        };

        let comp = if left_comp != usize::MAX && left_comp == right_comp {
            left_comp
        } else {
            usize::MAX
        };
        node_component[node_idx] = comp;
        comp
    }
}

/// Core dual-tree traversal with per-component tracking.
fn dual_tree_search(
    tree: &BoundedKdTree,
    query_node: usize,
    ref_node: usize,
    core_dists: &[f64],
    node_min_core: &[f64],
    point_component: &[usize],
    component_best: &mut [ComponentBest],
    node_component: &[usize],
    alpha: f64,
) {
    if query_node == NO_CHILD || ref_node == NO_CHILD {
        return;
    }

    // === Pruning 1: O(1) same-component check via cached node components ===
    let q_comp = node_component[query_node];
    let r_comp = node_component[ref_node];
    if q_comp != usize::MAX && q_comp == r_comp {
        return;
    }

    // === Pruning 2: MR distance lower bound ===
    let min_dist_sq = tree.min_dist_sq_node_to_node(query_node, ref_node);
    let min_dist = min_dist_sq.sqrt();
    let min_dist_scaled = if alpha != 1.0 {
        min_dist / alpha
    } else {
        min_dist
    };

    let mr_lower = f64::max(
        node_min_core[query_node],
        f64::max(node_min_core[ref_node], min_dist_scaled),
    );

    // Check if this lower bound can improve ANY component in the query node.
    let q_node = &tree.nodes[query_node];
    let mut can_prune = true;
    for &idx in &tree.sorted_indices[q_node.idx_start..q_node.idx_end] {
        let comp = point_component[idx];
        if mr_lower < component_best[comp].mr_dist {
            can_prune = false;
            break;
        }
    }
    if can_prune {
        return;
    }

    let r_node = &tree.nodes[ref_node];

    // === Base case: both leaves ===
    if q_node.is_leaf && r_node.is_leaf {
        let q_points = &tree.sorted_indices[q_node.idx_start..q_node.idx_end];
        let r_points = &tree.sorted_indices[r_node.idx_start..r_node.idx_end];

        for &qi in q_points {
            let comp_q = point_component[qi];
            let core_q = core_dists[qi];

            // Core distance shortcut: can't beat current best
            if core_q >= component_best[comp_q].mr_dist {
                continue;
            }

            for &ri in r_points {
                let comp_r = point_component[ri];
                if comp_q == comp_r {
                    continue;
                }

                let core_r = core_dists[ri];
                let core_max = f64::max(core_q, core_r);

                // Lazy sqrt: avoid sqrt when core distances dominate
                let d_sq = tree.dist_sq(qi, ri);
                let mr = if alpha == 1.0 && d_sq <= core_max * core_max {
                    core_max
                } else {
                    let dist = d_sq.sqrt();
                    let scaled = if alpha != 1.0 { dist / alpha } else { dist };
                    f64::max(core_max, scaled)
                };

                if mr < component_best[comp_q].mr_dist {
                    component_best[comp_q] = ComponentBest {
                        mr_dist: mr,
                        from: qi,
                        to: ri,
                    };
                }
                if mr < component_best[comp_r].mr_dist {
                    component_best[comp_r] = ComponentBest {
                        mr_dist: mr,
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
    if q_node.is_leaf || (!r_node.is_leaf && r_node.count > q_node.count) {
        // Split the reference node
        let (first, second) = closer_child_first(tree, query_node, ref_node);

        if first != NO_CHILD {
            dual_tree_search(tree, query_node, first, core_dists, node_min_core, point_component, component_best, node_component, alpha);
        }
        if second != NO_CHILD {
            dual_tree_search(tree, query_node, second, core_dists, node_min_core, point_component, component_best, node_component, alpha);
        }
    } else {
        // Split the query node
        if q_node.left != NO_CHILD {
            dual_tree_search(tree, q_node.left, ref_node, core_dists, node_min_core, point_component, component_best, node_component, alpha);
        }
        if q_node.right != NO_CHILD {
            dual_tree_search(tree, q_node.right, ref_node, core_dists, node_min_core, point_component, component_best, node_component, alpha);
        }
    }
}

/// Determine which child of ref_node is closer to query_node, return (closer, farther).
fn closer_child_first(
    tree: &BoundedKdTree,
    query_node: usize,
    ref_node: usize,
) -> (usize, usize) {
    let r = &tree.nodes[ref_node];
    let dist_left = if r.left != NO_CHILD {
        tree.min_dist_sq_node_to_node(query_node, r.left)
    } else {
        f64::INFINITY
    };
    let dist_right = if r.right != NO_CHILD {
        tree.min_dist_sq_node_to_node(query_node, r.right)
    } else {
        f64::INFINITY
    };

    if dist_left <= dist_right {
        (r.left, r.right)
    } else {
        (r.right, r.left)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core_distance;
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

        let dtb_edges = dual_tree_boruvka_mst(&tree, &cd.view(), 1.0);
        let prim_edges = crate::mst::prim::prim_mst(
            &data.view(),
            &cd.view(),
            &Metric::Euclidean,
            1.0,
        );

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

        let dtb_edges = dual_tree_boruvka_mst(&tree, &cd.view(), 1.0);
        let prim_edges = crate::mst::prim::prim_mst(
            &data.view(),
            &cd.view(),
            &Metric::Euclidean,
            1.0,
        );

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
}
