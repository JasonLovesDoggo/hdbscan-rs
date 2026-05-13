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

    let node_count = n_ids;
    let mut cluster_parent = vec![usize::MAX; node_count];
    let mut cluster_present = vec![false; node_count];
    let mut max_lambda = vec![0.0; node_count];

    for edge in condensed_tree {
        if edge.parent < node_count {
            cluster_present[edge.parent] = true;
        }
        if edge.child >= n_points {
            cluster_parent[edge.child] = edge.parent;
            cluster_present[edge.child] = true;
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
        if edge.child < n_points && edge.lambda_val.is_finite() {
            let current = &mut max_lambda[edge.parent];
            if edge.lambda_val > *current {
                *current = edge.lambda_val;
            }
        }
    }

    // Propagate max lambda up the tree
    for cluster in (n_points..node_count).rev() {
        if cluster_present[cluster] {
            let parent = cluster_parent[cluster];
            if parent != usize::MAX {
                let child_max = max_lambda[cluster];
                let parent_max = &mut max_lambda[parent];
                if child_max > *parent_max {
                    *parent_max = child_max;
                }
            }
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
