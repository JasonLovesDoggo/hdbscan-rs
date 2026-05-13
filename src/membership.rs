use crate::types::CondensedTreeEdge;
use std::collections::HashSet;

/// Compute membership probabilities for each point.
///
/// For a point in cluster C:
///   prob = (lambda_point - lambda_birth(C)) / (lambda_max(C) - lambda_birth(C))
///
/// Noise points get probability 0.0.
pub fn compute_probabilities(
    condensed_tree: &[CondensedTreeEdge],
    selected_clusters: &HashSet<usize>,
    labels: &[i32],
    n_points: usize,
) -> Vec<f64> {
    if selected_clusters.is_empty() {
        return vec![0.0; n_points];
    }

    let mut sorted_selected: Vec<usize> = selected_clusters.iter().copied().collect();
    sorted_selected.sort_unstable();

    let max_node = condensed_tree
        .iter()
        .map(|edge| edge.parent.max(edge.child))
        .max()
        .unwrap_or(0);
    let node_count = max_node + 1;
    let mut is_selected = vec![false; node_count];
    for &cluster in &sorted_selected {
        if cluster < node_count {
            is_selected[cluster] = true;
        }
    }

    // Compute birth lambda and max lambda for each selected cluster.
    let mut birth_lambda = vec![0.0; node_count];
    let mut birth_seen = vec![false; node_count];
    let mut max_lambda = vec![0.0; node_count];

    for edge in condensed_tree {
        if edge.child >= n_points
            && is_selected[edge.child]
            && (!birth_seen[edge.child] || edge.lambda_val < birth_lambda[edge.child])
        {
            birth_lambda[edge.child] = edge.lambda_val;
            birth_seen[edge.child] = true;
        }
    }

    // For root cluster, birth lambda is 0
    if let Some(&root) = sorted_selected.first() {
        if !birth_seen[root] {
            birth_lambda[root] = 0.0;
            birth_seen[root] = true;
        }
    }

    // Max lambda: the maximum lambda at which any point falls out of this cluster
    // (or any descendant that's been absorbed into it)
    // We track the max lambda of point edges whose parent is the cluster or an ancestor
    // that maps to it.

    // Build the "effective cluster" for each cluster node in the condensed tree
    // (walk up to find the nearest selected ancestor)
    let mut cluster_parent = vec![usize::MAX; node_count];
    let mut cluster_present = vec![false; node_count];
    for edge in condensed_tree {
        cluster_present[edge.parent] = true;
        if edge.child >= n_points {
            cluster_parent[edge.child] = edge.parent;
            cluster_present[edge.child] = true;
        }
    }

    let mut effective_cluster = vec![usize::MAX; node_count];
    for &c in &sorted_selected {
        effective_cluster[c] = c;
    }

    // For non-selected clusters, walk up to find selected ancestor
    for c in n_points..node_count {
        if cluster_present[c] && effective_cluster[c] == usize::MAX {
            let mut current = c;
            while cluster_parent[current] != usize::MAX {
                let parent = cluster_parent[current];
                let ec = effective_cluster[parent];
                if ec != usize::MAX {
                    effective_cluster[c] = ec;
                    break;
                }
                current = parent;
            }
        }
    }

    // Compute max lambda per selected cluster from point edges
    for edge in condensed_tree {
        if edge.child < n_points {
            let ec = effective_cluster[edge.parent];
            if ec != usize::MAX && edge.lambda_val.is_finite() && edge.lambda_val > max_lambda[ec] {
                max_lambda[ec] = edge.lambda_val;
            }
        }
    }

    // Find the lambda at which each point enters its cluster
    let mut point_lambda = vec![0.0; n_points];
    for edge in condensed_tree {
        if edge.child < n_points && effective_cluster[edge.parent] != usize::MAX {
            // Use the highest lambda entry for this point in this effective cluster
            if edge.lambda_val >= point_lambda[edge.child] {
                point_lambda[edge.child] = edge.lambda_val;
            }
        }
    }

    let mut probabilities = vec![0.0; n_points];

    for point in 0..n_points {
        if labels[point] < 0 {
            continue;
        }
        let cluster = sorted_selected[labels[point] as usize];
        let bl = birth_lambda[cluster];
        let ml = max_lambda[cluster];
        let pl = point_lambda[point];

        let range = ml - bl;
        if range > 0.0 && range.is_finite() {
            probabilities[point] = ((pl - bl) / range).clamp(0.0, 1.0);
        } else {
            probabilities[point] = 1.0;
        }
    }

    probabilities
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::CondensedTreeEdge;

    #[test]
    fn test_all_noise_probabilities() {
        let probs = compute_probabilities(&[], &HashSet::new(), &[-1, -1], 2);
        assert_eq!(probs, vec![0.0, 0.0]);
    }

    #[test]
    fn non_root_selected_cluster_keeps_birth_lambda() {
        let condensed = vec![
            CondensedTreeEdge {
                parent: 4,
                child: 3,
                lambda_val: 2.0,
                child_size: 2,
            },
            CondensedTreeEdge {
                parent: 3,
                child: 0,
                lambda_val: 3.0,
                child_size: 1,
            },
            CondensedTreeEdge {
                parent: 3,
                child: 1,
                lambda_val: 4.0,
                child_size: 1,
            },
        ];
        let selected = HashSet::from([3]);
        let labels = vec![0, 0];

        let probs = compute_probabilities(&condensed, &selected, &labels, 2);

        assert_eq!(probs, vec![0.5, 1.0]);
    }
}
