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

    let max_id = condensed_tree
        .iter()
        .map(|e| e.parent.max(e.child))
        .max()
        .unwrap_or(0);
    let n_ids = max_id + 1;

    // Birth lambda and max lambda for each cluster, indexed by cluster ID
    let mut birth_lambda = vec![f64::INFINITY; n_ids];
    let mut max_lambda = vec![0.0f64; n_ids];
    let mut cluster_parent = vec![usize::MAX; n_ids];

    for edge in condensed_tree {
        if edge.child >= n_points {
            cluster_parent[edge.child] = edge.parent;
            if selected_clusters.contains(&edge.child) && edge.lambda_val < birth_lambda[edge.child]
            {
                birth_lambda[edge.child] = edge.lambda_val;
            }
        }
    }

    // For root cluster, birth lambda is 0
    if let Some(&root) = sorted_selected.first() {
        if birth_lambda[root] == f64::INFINITY {
            birth_lambda[root] = 0.0;
        }
    }

    // Build effective cluster mapping (nearest selected ancestor)
    let mut effective_cluster = vec![usize::MAX; n_ids];
    for &c in selected_clusters {
        effective_cluster[c] = c;
    }
    // Process in ascending order so parent mappings are resolved first (root has lowest ID)
    for c in n_points..n_ids {
        if effective_cluster[c] == usize::MAX {
            let parent = cluster_parent[c];
            if parent != usize::MAX && effective_cluster[parent] != usize::MAX {
                effective_cluster[c] = effective_cluster[parent];
            }
        }
    }
    // Second pass for deeper nesting
    for c in n_points..n_ids {
        if effective_cluster[c] == usize::MAX {
            let mut current = c;
            while cluster_parent[current] != usize::MAX {
                current = cluster_parent[current];
                if effective_cluster[current] != usize::MAX {
                    effective_cluster[c] = effective_cluster[current];
                    break;
                }
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
    let mut point_lambda = vec![f64::NEG_INFINITY; n_points];
    for edge in condensed_tree {
        if edge.child < n_points {
            let ec = effective_cluster[edge.parent];
            if ec != usize::MAX && edge.lambda_val >= point_lambda[edge.child] {
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
        let bl = if bl == f64::INFINITY { 0.0 } else { bl };
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

    #[test]
    fn test_all_noise_probabilities() {
        let probs = compute_probabilities(&[], &HashSet::new(), &[-1, -1], 2);
        assert_eq!(probs, vec![0.0, 0.0]);
    }
}
