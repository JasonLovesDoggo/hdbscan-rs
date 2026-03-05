use crate::types::{CondensedTreeEdge, SingleLinkageMerge};
use crate::union_find::UnionFind;

struct HierarchyNode {
    left_child: usize,
    right_child: usize,
    lambda: f64,
    size: usize,
}

/// Build a condensed tree from a single-linkage dendrogram.
///
/// The condensed tree prunes the hierarchy: when a split occurs and one side
/// has fewer than `min_cluster_size` points, those points "fall out" as noise
/// rather than forming a new cluster.
///
/// Cluster IDs start at `n_points` (root = n_points for the first virtual cluster).
///
/// Matches sklearn's `_condense_tree` behavior: when a small subtree falls out,
/// each point is emitted at the lambda of the hierarchy node where that point
/// actually separated, not the parent's lambda.
pub fn build_condensed_tree(
    merges: &[SingleLinkageMerge],
    n_points: usize,
    min_cluster_size: usize,
) -> Vec<CondensedTreeEdge> {
    if merges.is_empty() {
        return vec![];
    }

    let n_merges = merges.len();
    let mut edges = Vec::new();

    // Replay union-find to reconstruct the binary hierarchy tree.
    let mut root_to_node: Vec<usize> = (0..n_points).collect();
    let mut uf = UnionFind::new(n_points);
    let mut hierarchy = Vec::with_capacity(n_merges);

    for (i, merge) in merges.iter().enumerate() {
        let root_left = uf.find(merge.left);
        let root_right = uf.find(merge.right);

        let left_node = root_to_node[root_left];
        let right_node = root_to_node[root_right];

        let left_size = node_size(&hierarchy, left_node, n_points);
        let right_size = node_size(&hierarchy, right_node, n_points);

        hierarchy.push(HierarchyNode {
            left_child: left_node,
            right_child: right_node,
            lambda: if merge.distance > 0.0 {
                1.0 / merge.distance
            } else {
                f64::INFINITY
            },
            size: left_size + right_size,
        });

        uf.union(merge.left, merge.right);
        let new_root = uf.find(merge.left);
        root_to_node[new_root] = n_points + i;
    }

    // Walk the hierarchy top-down to build the condensed tree.
    let mut next_cluster = n_points;
    let mut condensed_id = vec![0usize; n_merges];

    let root_idx = n_merges - 1;
    condensed_id[root_idx] = next_cluster;
    next_cluster += 1;

    // Stack entries: (hierarchy_node_idx, current_condensed_cluster_id)
    let mut stack = vec![(root_idx, condensed_id[root_idx])];

    while let Some((node_idx, current_cluster)) = stack.pop() {
        let parent_lambda = hierarchy[node_idx].lambda;
        let left = hierarchy[node_idx].left_child;
        let right = hierarchy[node_idx].right_child;

        let left_size = node_size(&hierarchy, left, n_points);
        let right_size = node_size(&hierarchy, right, n_points);

        let left_big = left_size >= min_cluster_size;
        let right_big = right_size >= min_cluster_size;

        match (left_big, right_big) {
            (true, true) => {
                // Real split: both children become new clusters
                process_big_child(
                    left, left_size, n_points, current_cluster, parent_lambda,
                    &mut condensed_id, &mut next_cluster, &mut edges, &mut stack,
                );
                process_big_child(
                    right, right_size, n_points, current_cluster, parent_lambda,
                    &mut condensed_id, &mut next_cluster, &mut edges, &mut stack,
                );
            }
            (true, false) => {
                // Right is too small: emit its points at their actual lambda
                emit_fallout_subtree(
                    &hierarchy, right, n_points, current_cluster, parent_lambda, &mut edges,
                );
                // Left continues as current_cluster (collapse)
                if left >= n_points {
                    condensed_id[left - n_points] = current_cluster;
                    stack.push((left - n_points, current_cluster));
                }
            }
            (false, true) => {
                emit_fallout_subtree(
                    &hierarchy, left, n_points, current_cluster, parent_lambda, &mut edges,
                );
                if right >= n_points {
                    condensed_id[right - n_points] = current_cluster;
                    stack.push((right - n_points, current_cluster));
                }
            }
            (false, false) => {
                emit_fallout_subtree(
                    &hierarchy, left, n_points, current_cluster, parent_lambda, &mut edges,
                );
                emit_fallout_subtree(
                    &hierarchy, right, n_points, current_cluster, parent_lambda, &mut edges,
                );
            }
        }
    }

    edges
}

fn node_size(hierarchy: &[HierarchyNode], node: usize, n_points: usize) -> usize {
    if node < n_points {
        1
    } else {
        hierarchy[node - n_points].size
    }
}

#[allow(clippy::too_many_arguments)]
fn process_big_child(
    child: usize,
    child_size: usize,
    n_points: usize,
    parent_cluster: usize,
    lambda_val: f64,
    condensed_id: &mut [usize],
    next_cluster: &mut usize,
    edges: &mut Vec<CondensedTreeEdge>,
    stack: &mut Vec<(usize, usize)>,
) {
    if child >= n_points {
        let cluster_id = *next_cluster;
        *next_cluster += 1;
        condensed_id[child - n_points] = cluster_id;
        edges.push(CondensedTreeEdge {
            parent: parent_cluster,
            child: cluster_id,
            lambda_val,
            child_size,
        });
        stack.push((child - n_points, cluster_id));
    } else {
        edges.push(CondensedTreeEdge {
            parent: parent_cluster,
            child,
            lambda_val,
            child_size: 1,
        });
    }
}

/// Emit point-level fallout edges for a subtree that is too small to be a cluster.
///
/// Unlike the naive approach of emitting all points at the parent's lambda,
/// this walks the subtree and emits each point at the lambda of the internal
/// hierarchy node where that point is located. This matches sklearn's behavior
/// where `_bfs_from_hierarchy` traverses the subtree preserving internal lambdas.
fn emit_fallout_subtree(
    hierarchy: &[HierarchyNode],
    node: usize,
    n_points: usize,
    parent_cluster: usize,
    fallout_lambda: f64,
    edges: &mut Vec<CondensedTreeEdge>,
) {
    if node < n_points {
        // Single point: falls out at the fallout_lambda
        edges.push(CondensedTreeEdge {
            parent: parent_cluster,
            child: node,
            lambda_val: fallout_lambda,
            child_size: 1,
        });
    } else {
        // Internal hierarchy node within the small subtree.
        // This node split at its own lambda. Both children
        // fall out into the parent cluster at max(fallout_lambda, node.lambda).
        // sklearn behavior: each sub-node's points fall out at the lambda of
        // the node where they appear, which is the max of the path lambdas.
        let h = &hierarchy[node - n_points];
        let child_lambda = f64::max(fallout_lambda, h.lambda);
        emit_fallout_subtree(hierarchy, h.left_child, n_points, parent_cluster, child_lambda, edges);
        emit_fallout_subtree(hierarchy, h.right_child, n_points, parent_cluster, child_lambda, edges);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SingleLinkageMerge;

    #[test]
    fn test_condensed_tree_basic() {
        let merges = vec![
            SingleLinkageMerge { left: 0, right: 1, distance: 1.0, size: 2 },
            SingleLinkageMerge { left: 2, right: 3, distance: 1.0, size: 2 },
            SingleLinkageMerge { left: 4, right: 5, distance: 1.0, size: 2 },
            SingleLinkageMerge { left: 0, right: 2, distance: 5.0, size: 4 },
            SingleLinkageMerge { left: 0, right: 4, distance: 10.0, size: 6 },
        ];
        let edges = build_condensed_tree(&merges, 6, 2);
        assert!(!edges.is_empty());
    }

    #[test]
    fn test_condensed_tree_empty() {
        let edges = build_condensed_tree(&[], 0, 2);
        assert!(edges.is_empty());
    }

    #[test]
    fn test_fallout_preserves_internal_lambda() {
        // 5 points: 0,1,2 form a cluster, 3,4 are a small subtree
        // Merges:
        //   0-1 at dist=1 (lambda=1)
        //   3-4 at dist=2 (lambda=0.5)
        //   0-2 at dist=3 (lambda=0.333)
        //   {0,1,2}-{3,4} at dist=10 (lambda=0.1)
        let merges = vec![
            SingleLinkageMerge { left: 0, right: 1, distance: 1.0, size: 2 },
            SingleLinkageMerge { left: 3, right: 4, distance: 2.0, size: 2 },
            SingleLinkageMerge { left: 0, right: 2, distance: 3.0, size: 3 },
            SingleLinkageMerge { left: 0, right: 3, distance: 10.0, size: 5 },
        ];
        let edges = build_condensed_tree(&merges, 5, 3);

        // Points 3 and 4 should fall out. Their internal split was at lambda=0.5.
        // The fallout happens at the parent split lambda=0.1.
        // Since 0.5 > 0.1, points 3 and 4 should be emitted at lambda=0.5
        // (their internal separation lambda).
        let fallout_3 = edges.iter().find(|e| e.child == 3).unwrap();
        let fallout_4 = edges.iter().find(|e| e.child == 4).unwrap();
        assert!(
            (fallout_3.lambda_val - 0.5).abs() < 1e-10,
            "Point 3 should fall out at lambda=0.5, got {}",
            fallout_3.lambda_val
        );
        assert!(
            (fallout_4.lambda_val - 0.5).abs() < 1e-10,
            "Point 4 should fall out at lambda=0.5, got {}",
            fallout_4.lambda_val
        );
    }
}
