use crate::types::CondensedTreeEdge;
use crate::union_find::UnionFind;
use std::collections::{HashMap, HashSet};

/// Assign flat cluster labels from selected clusters and the condensed tree.
///
/// Uses a union-find approach matching sklearn's `_do_labelling`:
/// - Union all edges where child is NOT a selected cluster
/// - Each point's component root determines its cluster label
/// - Special handling for allow_single_cluster with threshold filtering
///
/// Returns labels indexed by point index.
pub fn assign_labels(
    condensed_tree: &[CondensedTreeEdge],
    selected_clusters: &HashSet<usize>,
    n_points: usize,
    allow_single_cluster: bool,
    cluster_selection_epsilon: f64,
) -> Vec<i32> {
    if selected_clusters.is_empty() {
        return vec![-1; n_points];
    }

    // Build a mapping from selected cluster IDs to sequential labels 0..k
    let mut sorted_selected: Vec<usize> = selected_clusters.iter().copied().collect();
    sorted_selected.sort_unstable();
    let cluster_to_label: HashMap<usize, i32> = sorted_selected
        .iter()
        .enumerate()
        .map(|(i, &c)| (c, i as i32))
        .collect();

    let root_cluster = condensed_tree.iter().map(|e| e.parent).min().unwrap();

    // Find max node ID for union-find sizing
    let max_parent = condensed_tree.iter().map(|e| e.parent).max().unwrap_or(0);
    let max_child = condensed_tree.iter().map(|e| e.child).max().unwrap_or(0);
    let uf_size = max_parent.max(max_child) + 1;

    let mut uf = UnionFind::new(uf_size);

    // Union all edges where child is NOT a selected cluster
    for edge in condensed_tree {
        if !selected_clusters.contains(&edge.child) {
            uf.union(edge.parent, edge.child);
        }
    }

    // Build point lambda lookup (each point appears exactly once as a child)
    let mut point_lambda: HashMap<usize, f64> = HashMap::new();
    for edge in condensed_tree {
        if edge.child < n_points {
            point_lambda.insert(edge.child, edge.lambda_val);
        }
    }

    // For single cluster + allow_single_cluster: compute threshold
    let single_cluster_threshold = if selected_clusters.len() == 1
        && allow_single_cluster
        && selected_clusters.contains(&root_cluster)
    {
        if cluster_selection_epsilon != 0.0 {
            Some(1.0 / cluster_selection_epsilon)
        } else {
            // Threshold = max lambda among edges whose parent == root_cluster
            let max_lambda = condensed_tree
                .iter()
                .filter(|e| e.parent == root_cluster)
                .map(|e| e.lambda_val)
                .fold(0.0f64, f64::max);
            Some(max_lambda)
        }
    } else {
        None
    };

    let mut labels = vec![-1i32; n_points];

    for point in 0..n_points {
        let cluster = uf.find(point);
        if cluster != root_cluster {
            // Point is in a non-root selected cluster
            if let Some(&label) = cluster_to_label.get(&cluster) {
                labels[point] = label;
            }
            // else: noise (component root is not a selected cluster)
        } else if let Some(threshold) = single_cluster_threshold {
            // Single cluster case: check if point's lambda meets threshold
            let lambda = point_lambda.get(&point).copied().unwrap_or(0.0);
            if lambda >= threshold {
                if let Some(&label) = cluster_to_label.get(&root_cluster) {
                    labels[point] = label;
                }
            }
            // else: noise
        }
        // else: point is in root component but root is not selected -> noise
    }

    labels
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_noise() {
        let labels = assign_labels(&[], &HashSet::new(), 5, false, 0.0);
        assert_eq!(labels, vec![-1, -1, -1, -1, -1]);
    }
}
