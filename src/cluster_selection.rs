use crate::params::ClusterSelectionMethod;
use crate::types::CondensedTreeEdge;
use std::collections::{HashMap, HashSet};

/// Result of cluster selection: which condensed tree cluster IDs are selected.
pub struct ClusterSelectionResult {
    /// Set of selected cluster IDs (condensed tree IDs, >= n_points)
    pub selected_clusters: HashSet<usize>,
}

/// Select clusters from the condensed tree.
pub fn select_clusters(
    condensed_tree: &[CondensedTreeEdge],
    n_points: usize,
    method: ClusterSelectionMethod,
    cluster_selection_epsilon: f64,
    allow_single_cluster: bool,
) -> ClusterSelectionResult {
    if condensed_tree.is_empty() {
        return ClusterSelectionResult {
            selected_clusters: HashSet::new(),
        };
    }

    // Identify all cluster nodes (IDs >= n_points) that appear as parents or cluster children
    let mut all_clusters: HashSet<usize> = HashSet::new();
    let mut children_of: HashMap<usize, Vec<usize>> = HashMap::new(); // cluster -> cluster children
    let mut cluster_birth_lambda: HashMap<usize, f64> = HashMap::new();

    for edge in condensed_tree {
        all_clusters.insert(edge.parent);
        if edge.child >= n_points {
            all_clusters.insert(edge.child);
            children_of
                .entry(edge.parent)
                .or_default()
                .push(edge.child);
            // A cluster's birth lambda is the lambda at which it splits from its parent
            cluster_birth_lambda
                .entry(edge.child)
                .and_modify(|v| {
                    if edge.lambda_val < *v {
                        *v = edge.lambda_val;
                    }
                })
                .or_insert(edge.lambda_val);
        }
    }

    // Root cluster birth lambda is 0
    let root = *all_clusters.iter().min().unwrap_or(&n_points);
    cluster_birth_lambda.entry(root).or_insert(0.0);

    // Find leaf clusters (no cluster children)
    let leaf_clusters: HashSet<usize> = all_clusters
        .iter()
        .filter(|c| !children_of.contains_key(c))
        .copied()
        .collect();

    let selected = match method {
        ClusterSelectionMethod::Eom => {
            eom_selection(condensed_tree, n_points, &all_clusters, &children_of, &leaf_clusters, allow_single_cluster)
        }
        ClusterSelectionMethod::Leaf => leaf_clusters.clone(),
    };

    // Apply epsilon merging
    let selected = if cluster_selection_epsilon > 0.0 {
        apply_epsilon_merging(
            &selected,
            condensed_tree,
            n_points,
            &children_of,
            &cluster_birth_lambda,
            cluster_selection_epsilon,
        )
    } else {
        selected
    };

    ClusterSelectionResult {
        selected_clusters: selected,
    }
}

/// EOM (Excess of Mass) cluster selection.
/// Maximizes total cluster stability.
///
/// Matches sklearn's implementation:
/// - Stability is computed as sum of (lambda - birth_lambda(parent)) * child_size
///   for ALL edges (both point and cluster edges).
/// - Bottom-up pass compares own stability vs sum of children's (propagated) stability.
/// - Root is excluded from selection when allow_single_cluster is false.
fn eom_selection(
    condensed_tree: &[CondensedTreeEdge],
    n_points: usize,
    all_clusters: &HashSet<usize>,
    children_of: &HashMap<usize, Vec<usize>>,
    _leaf_clusters: &HashSet<usize>,
    allow_single_cluster: bool,
) -> HashSet<usize> {
    let root = *all_clusters.iter().min().unwrap_or(&n_points);

    // Compute birth lambda for each node (point or cluster).
    // In sklearn: births[child] = edge.value for each edge, then births[root] = 0.
    // Each child appears exactly once as a child in the condensed tree.
    let mut births: HashMap<usize, f64> = HashMap::new();
    for edge in condensed_tree {
        births.insert(edge.child, edge.lambda_val);
    }
    births.insert(root, 0.0);

    // Compute stability for each cluster.
    // sklearn: stability[parent] += (lambda_val - births[parent]) * child_size
    // for ALL edges (both point-level and cluster-level).
    let mut stability: HashMap<usize, f64> = HashMap::new();
    for &c in all_clusters {
        stability.insert(c, 0.0);
    }

    for edge in condensed_tree {
        let parent = edge.parent;
        let bl = *births.get(&parent).unwrap_or(&0.0);
        let contribution = (edge.lambda_val - bl) * edge.child_size as f64;
        *stability.entry(parent).or_insert(0.0) += contribution;
    }

    // Build the node list for EOM processing.
    // sklearn: if allow_single_cluster, include all; otherwise exclude root.
    let mut node_list: Vec<usize> = if allow_single_cluster {
        all_clusters.iter().copied().collect()
    } else {
        all_clusters.iter().copied().filter(|&c| c != root).collect()
    };
    // Process in reverse topological order (highest ID = deepest first)
    node_list.sort_unstable_by(|a, b| b.cmp(a));

    // is_cluster tracks which nodes are selected (all start as true)
    let mut is_cluster: HashMap<usize, bool> = HashMap::new();
    for &c in &node_list {
        is_cluster.insert(c, true);
    }

    // Bottom-up pass: for each node, compare its stability to sum of children's stability.
    // If children win, set node to not-a-cluster and propagate children's stability up.
    // If node wins, deselect all descendants.
    // Note: stability dict is mutated in place (like sklearn).
    for &node in &node_list {
        if let Some(children) = children_of.get(&node) {
            let subtree_stability: f64 = children
                .iter()
                .map(|c| *stability.get(c).unwrap_or(&0.0))
                .sum();

            let own_stability = *stability.get(&node).unwrap_or(&0.0);

            if subtree_stability > own_stability {
                // Children are collectively better
                is_cluster.insert(node, false);
                stability.insert(node, subtree_stability);
            } else {
                // This node is better: deselect all descendants
                let descendants = bfs_descendants(node, children_of);
                for sub_node in descendants {
                    is_cluster.insert(sub_node, false);
                }
            }
        }
    }

    let selected: HashSet<usize> = is_cluster
        .iter()
        .filter(|(_, &v)| v)
        .map(|(&k, _)| k)
        .collect();

    // Handle allow_single_cluster edge case:
    // If only the root is selected and allow_single_cluster is false, return empty
    if !allow_single_cluster && selected.len() == 1 && selected.contains(&root) {
        return HashSet::new();
    }

    selected
}

/// BFS to find all descendants of a node in the cluster tree (excluding the node itself).
fn bfs_descendants(
    node: usize,
    children_of: &HashMap<usize, Vec<usize>>,
) -> Vec<usize> {
    let mut result = Vec::new();
    let mut queue = Vec::new();
    if let Some(children) = children_of.get(&node) {
        queue.extend(children.iter().copied());
    }
    while let Some(current) = queue.pop() {
        result.push(current);
        if let Some(children) = children_of.get(&current) {
            queue.extend(children.iter().copied());
        }
    }
    result
}

/// Apply epsilon merging: merge selected clusters whose split distance < epsilon.
fn apply_epsilon_merging(
    selected: &HashSet<usize>,
    _condensed_tree: &[CondensedTreeEdge],
    _n_points: usize,
    children_of: &HashMap<usize, Vec<usize>>,
    birth_lambda: &HashMap<usize, f64>,
    epsilon: f64,
) -> HashSet<usize> {
    let epsilon_lambda = if epsilon > 0.0 { 1.0 / epsilon } else { f64::INFINITY };
    let mut result = selected.clone();

    // For each selected cluster, if its birth lambda < epsilon_lambda,
    // it means the split happened at a distance > epsilon.
    // We want to merge clusters that split at distance < epsilon.
    // A cluster born at lambda means it split from parent at distance 1/lambda.
    // If 1/lambda < epsilon (i.e., lambda > epsilon_lambda), the split is too fine -> merge back.

    // Walk top-down: if a parent has children that are all selected,
    // and the children were born at lambda > epsilon_lambda, merge them.
    let mut changed = true;
    while changed {
        changed = false;
        let current = result.clone();
        for &cluster in &current {
            if let Some(children) = children_of.get(&cluster) {
                // Check if all children are selected and born at lambda > epsilon_lambda
                let all_children_selected_and_fine = children.iter().all(|c| {
                    result.contains(c)
                        && birth_lambda.get(c).copied().unwrap_or(0.0) > epsilon_lambda
                });
                if all_children_selected_and_fine && !children.is_empty() {
                    // Merge: select parent instead of children
                    for &child in children {
                        result.remove(&child);
                    }
                    result.insert(cluster);
                    changed = true;
                }
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_tree() {
        let result = select_clusters(&[], 0, ClusterSelectionMethod::Eom, 0.0, false);
        assert!(result.selected_clusters.is_empty());
    }
}
