use crate::types::CondensedTreeEdge;

/// Compute GLOSH (Global-Local Outlier Scores from Hierarchies) outlier scores.
///
/// For each point: score = (lambda_max_cluster - lambda_point) / lambda_max_cluster
///
/// Where lambda_max_cluster is the maximum lambda of the cluster the point belongs to
/// (or would belong to, for noise points).
///
/// Scores are in [0, 1], where higher = more outlier-like.
pub fn compute_outlier_scores(condensed_tree: &[CondensedTreeEdge], n_points: usize) -> Vec<f64> {
    if condensed_tree.is_empty() {
        return vec![0.0; n_points];
    }

    // Find max cluster ID to size Vec-based lookups
    let max_id = condensed_tree
        .iter()
        .map(|e| e.parent.max(e.child))
        .max()
        .unwrap_or(0);
    let n_ids = max_id + 1;

    // cluster_parent[c] = parent of cluster c (0 = no parent)
    let mut cluster_parent = vec![usize::MAX; n_ids];
    let mut max_lambda = vec![0.0f64; n_ids];

    for edge in condensed_tree {
        if edge.child >= n_points {
            cluster_parent[edge.child] = edge.parent;
        }
    }

    // Find which cluster each point most deeply belongs to
    let mut point_parent = vec![usize::MAX; n_points];
    let mut point_lambda = vec![f64::NEG_INFINITY; n_points];

    for edge in condensed_tree {
        if edge.child < n_points && edge.lambda_val >= point_lambda[edge.child] {
            point_parent[edge.child] = edge.parent;
            point_lambda[edge.child] = edge.lambda_val;
        }
    }

    // Compute max lambda per cluster from point fallouts
    for edge in condensed_tree {
        if edge.child < n_points && edge.lambda_val.is_finite() && edge.lambda_val > max_lambda[edge.parent] {
            max_lambda[edge.parent] = edge.lambda_val;
        }
    }

    // Collect and sort clusters by ID descending (leaves first) for bottom-up propagation
    let mut sorted_clusters: Vec<usize> = (n_points..n_ids)
        .filter(|&c| cluster_parent[c] != usize::MAX || c == n_points)
        .collect();
    sorted_clusters.sort_unstable_by(|a, b| b.cmp(a));

    for &cluster in &sorted_clusters {
        let parent = cluster_parent[cluster];
        if parent != usize::MAX && max_lambda[cluster] > max_lambda[parent] {
            max_lambda[parent] = max_lambda[cluster];
        }
    }

    // Compute scores
    let mut scores = vec![0.0; n_points];

    for point in 0..n_points {
        let parent = point_parent[point];
        if parent != usize::MAX {
            let pl = point_lambda[point];
            let ml = max_lambda[parent];
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
