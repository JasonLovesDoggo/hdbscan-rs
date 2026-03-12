use crate::types::CondensedTreeEdge;
use crate::union_find::UnionFind;
use std::collections::HashSet;

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

    let root_cluster = condensed_tree.iter().map(|e| e.parent).min().unwrap();

    // Find max node ID for union-find sizing
    let max_parent = condensed_tree.iter().map(|e| e.parent).max().unwrap_or(0);
    let max_child = condensed_tree.iter().map(|e| e.child).max().unwrap_or(0);
    let uf_size = max_parent.max(max_child) + 1;

    // Vec-based cluster-to-label mapping
    let mut cluster_to_label = vec![-1i32; uf_size];
    for (i, &c) in sorted_selected.iter().enumerate() {
        cluster_to_label[c] = i as i32;
    }

    let mut uf = UnionFind::new(uf_size);

    // Union all edges where child is NOT a selected cluster
    for edge in condensed_tree {
        if !selected_clusters.contains(&edge.child) {
            uf.union(edge.parent, edge.child);
        }
    }

    // Build point lambda lookup
    let mut point_lambda = vec![0.0f64; n_points];
    for edge in condensed_tree {
        if edge.child < n_points {
            point_lambda[edge.child] = edge.lambda_val;
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
            let label = cluster_to_label[cluster];
            if label >= 0 {
                labels[point] = label;
            }
        } else if let Some(threshold) = single_cluster_threshold {
            let lambda = point_lambda[point];
            if lambda >= threshold {
                let label = cluster_to_label[root_cluster];
                if label >= 0 {
                    labels[point] = label;
                }
            }
        }
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
