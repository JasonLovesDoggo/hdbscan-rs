use crate::types::CondensedTreeEdge;
use std::collections::{HashMap, HashSet};

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

    // Build mapping from sequential label to cluster ID
    let mut sorted_selected: Vec<usize> = selected_clusters.iter().copied().collect();
    sorted_selected.sort_unstable();

    // Compute birth lambda and max lambda for each selected cluster
    let mut birth_lambda: HashMap<usize, f64> = HashMap::new();
    let mut max_lambda: HashMap<usize, f64> = HashMap::new();

    // Birth lambda: the lambda at which this cluster was created (edge from parent to this cluster)
    for edge in condensed_tree {
        if edge.child >= n_points && selected_clusters.contains(&edge.child) {
            birth_lambda
                .entry(edge.child)
                .and_modify(|v| {
                    if edge.lambda_val < *v {
                        *v = edge.lambda_val;
                    }
                })
                .or_insert(edge.lambda_val);
        }
    }

    // For root cluster, birth lambda is 0
    if let Some(&root) = sorted_selected.first() {
        birth_lambda.entry(root).or_insert(0.0);
    }

    // Max lambda: the maximum lambda at which any point falls out of this cluster
    // (or any descendant that's been absorbed into it)
    // We track the max lambda of point edges whose parent is the cluster or an ancestor
    // that maps to it.

    // Build the "effective cluster" for each cluster node in the condensed tree
    // (walk up to find the nearest selected ancestor)
    let mut cluster_parent: HashMap<usize, usize> = HashMap::new();
    for edge in condensed_tree {
        if edge.child >= n_points {
            cluster_parent.insert(edge.child, edge.parent);
        }
    }

    let mut effective_cluster: HashMap<usize, usize> = HashMap::new();
    for &c in selected_clusters {
        effective_cluster.insert(c, c);
    }

    // For non-selected clusters, walk up to find selected ancestor
    let all_clusters: HashSet<usize> = condensed_tree
        .iter()
        .flat_map(|e| {
            let mut v = vec![e.parent];
            if e.child >= n_points {
                v.push(e.child);
            }
            v
        })
        .collect();

    for &c in &all_clusters {
        if !effective_cluster.contains_key(&c) {
            let mut current = c;
            while let Some(&parent) = cluster_parent.get(&current) {
                if let Some(&ec) = effective_cluster.get(&parent) {
                    effective_cluster.insert(c, ec);
                    break;
                }
                current = parent;
            }
        }
    }

    // Compute max lambda per selected cluster from point edges
    for edge in condensed_tree {
        if edge.child < n_points {
            if let Some(&ec) = effective_cluster.get(&edge.parent) {
                let current_max = max_lambda.entry(ec).or_insert(0.0_f64);
                if edge.lambda_val.is_finite() && edge.lambda_val > *current_max {
                    *current_max = edge.lambda_val;
                }
            }
        }
    }

    // Find the lambda at which each point enters its cluster
    let mut point_lambda: HashMap<usize, f64> = HashMap::new();
    let mut point_cluster_id: HashMap<usize, usize> = HashMap::new();
    for edge in condensed_tree {
        if edge.child < n_points {
            if let Some(&ec) = effective_cluster.get(&edge.parent) {
                // Use the highest lambda entry for this point in this effective cluster
                let current = point_lambda.get(&edge.child).copied().unwrap_or(f64::NEG_INFINITY);
                if edge.lambda_val >= current {
                    point_lambda.insert(edge.child, edge.lambda_val);
                    point_cluster_id.insert(edge.child, ec);
                }
            }
        }
    }

    let mut probabilities = vec![0.0; n_points];

    for point in 0..n_points {
        if labels[point] < 0 {
            continue; // noise
        }
        let cluster = sorted_selected[labels[point] as usize];
        let bl = *birth_lambda.get(&cluster).unwrap_or(&0.0);
        let ml = *max_lambda.get(&cluster).unwrap_or(&0.0);
        let pl = *point_lambda.get(&point).unwrap_or(&0.0);

        let range = ml - bl;
        if range > 0.0 && range.is_finite() {
            let prob = ((pl - bl) / range).clamp(0.0, 1.0);
            probabilities[point] = prob;
        } else {
            // All points at same lambda or infinite
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
