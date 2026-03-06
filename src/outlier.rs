use crate::types::CondensedTreeEdge;
use std::collections::{HashMap, HashSet};

/// Compute GLOSH (Global-Local Outlier Scores from Hierarchies) outlier scores.
///
/// For each point: score = (lambda_max_cluster - lambda_point) / lambda_max_cluster
///
/// Where lambda_max_cluster is the maximum lambda of the cluster the point belongs to
/// (or would belong to, for noise points).
///
/// Scores are in [0, 1], where higher = more outlier-like.
pub fn compute_outlier_scores(
    condensed_tree: &[CondensedTreeEdge],
    n_points: usize,
) -> Vec<f64> {
    if condensed_tree.is_empty() {
        return vec![0.0; n_points];
    }

    // For GLOSH, we use ALL clusters in the condensed tree (not just selected ones).
    // Each point's outlier score is relative to the cluster subtree it belongs to.

    // Find max lambda for each cluster (considering all descendant points)
    let mut cluster_parent: HashMap<usize, usize> = HashMap::new();
    let mut max_lambda: HashMap<usize, f64> = HashMap::new();

    for edge in condensed_tree {
        if edge.child >= n_points {
            cluster_parent.insert(edge.child, edge.parent);
        }
    }

    // Find which cluster each point most deeply belongs to
    let mut point_parent: HashMap<usize, usize> = HashMap::new();
    let mut point_lambda: HashMap<usize, f64> = HashMap::new();

    for edge in condensed_tree {
        if edge.child < n_points {
            let current = point_lambda.get(&edge.child).copied().unwrap_or(f64::NEG_INFINITY);
            if edge.lambda_val >= current {
                point_parent.insert(edge.child, edge.parent);
                point_lambda.insert(edge.child, edge.lambda_val);
            }
        }
    }

    // Compute max lambda per cluster from point fallouts
    for edge in condensed_tree {
        if edge.child < n_points && edge.lambda_val.is_finite() {
            let current = max_lambda.entry(edge.parent).or_insert(0.0_f64);
            if edge.lambda_val > *current {
                *current = edge.lambda_val;
            }
        }
    }

    // Propagate max lambda up the tree
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

    // Sort clusters by ID descending (leaves first) for bottom-up propagation
    let mut sorted_clusters: Vec<usize> = all_clusters.iter().copied().collect();
    sorted_clusters.sort_unstable_by(|a, b| b.cmp(a));

    for &cluster in &sorted_clusters {
        if let Some(&parent) = cluster_parent.get(&cluster) {
            let child_max = *max_lambda.get(&cluster).unwrap_or(&0.0);
            let parent_max = max_lambda.entry(parent).or_insert(0.0_f64);
            if child_max > *parent_max {
                *parent_max = child_max;
            }
        }
    }

    // Compute scores
    let mut scores = vec![0.0; n_points];

    for point in 0..n_points {
        if let Some(&parent) = point_parent.get(&point) {
            let pl = *point_lambda.get(&point).unwrap_or(&0.0);
            let ml = *max_lambda.get(&parent).unwrap_or(&0.0);

            if ml > 0.0 && ml.is_finite() {
                scores[point] = ((ml - pl) / ml).clamp(0.0, 1.0);
            }
        }
    }

    scores
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_tree_outliers() {
        let scores = compute_outlier_scores(&[], 3);
        assert_eq!(scores, vec![0.0, 0.0, 0.0]);
    }
}
