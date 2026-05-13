use crate::params::ClusterSelectionMethod;
use crate::types::CondensedTreeEdge;
use std::collections::HashSet;

/// Result of cluster selection: which condensed tree cluster IDs are selected.
pub struct ClusterSelectionResult {
    /// Set of selected cluster IDs (condensed tree IDs, >= n_points)
    pub selected_clusters: HashSet<usize>,
    /// Stability value for each selected cluster (ordered by cluster label 0, 1, 2, ...)
    pub cluster_persistence: Vec<f64>,
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
            cluster_persistence: Vec::new(),
        };
    }

    // Cluster ids are dense enough to use id-indexed storage: points are
    // `0..n_points`, condensed clusters start at `n_points`, and each merge
    // introduces at most one new cluster id.
    let node_count = condensed_tree
        .iter()
        .map(|edge| edge.parent.max(edge.child))
        .max()
        .map_or(n_points + 1, |max_node| max_node + 1);
    let mut cluster_present = vec![false; node_count];
    let mut child_counts = vec![0usize; node_count];
    let mut cluster_birth_lambda = vec![0.0; node_count];
    let mut cluster_birth_seen = vec![false; node_count];

    for edge in condensed_tree {
        cluster_present[edge.parent] = true;
        if edge.child >= n_points {
            cluster_present[edge.child] = true;
            child_counts[edge.parent] += 1;
            // A cluster's birth lambda is the lambda at which it splits from its parent
            if !cluster_birth_seen[edge.child] || edge.lambda_val < cluster_birth_lambda[edge.child]
            {
                cluster_birth_lambda[edge.child] = edge.lambda_val;
                cluster_birth_seen[edge.child] = true;
            }
        }
    }

    // Root cluster birth lambda is 0
    let root = cluster_present
        .iter()
        .enumerate()
        .skip(n_points)
        .find_map(|(cluster, &present)| present.then_some(cluster))
        .unwrap_or(n_points);
    cluster_birth_lambda[root] = 0.0;
    let children_of = ClusterChildren::new(condensed_tree, n_points, child_counts);
    let all_clusters: Vec<usize> = cluster_present
        .iter()
        .enumerate()
        .skip(n_points)
        .filter_map(|(cluster, &present)| present.then_some(cluster))
        .collect();

    // Find leaf clusters (no cluster children)
    let leaf_clusters: HashSet<usize> = all_clusters
        .iter()
        .copied()
        .filter(|&cluster| children_of.children(cluster).is_empty())
        .collect();

    let selected = match method {
        ClusterSelectionMethod::Eom => eom_selection(
            condensed_tree,
            n_points,
            &all_clusters,
            &children_of,
            &leaf_clusters,
            allow_single_cluster,
        ),
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

    // Compute cluster_persistence for selected clusters.
    // persistence = 1/birth_lambda - 1/death_lambda for each selected cluster.
    // Death lambda is the max lambda of any edge with that cluster as parent.
    let mut cluster_death_lambda = vec![0.0; node_count];
    for edge in condensed_tree {
        let entry = &mut cluster_death_lambda[edge.parent];
        if edge.lambda_val > *entry {
            *entry = edge.lambda_val;
        }
    }

    // Sort selected clusters by ID for deterministic label assignment order
    let mut sorted_selected: Vec<usize> = selected.iter().copied().collect();
    sorted_selected.sort_unstable();

    let cluster_persistence: Vec<f64> = sorted_selected
        .iter()
        .map(|&c| {
            let birth = cluster_birth_lambda[c];
            let death = cluster_death_lambda[c];
            let inv_birth = if birth > 0.0 {
                1.0 / birth
            } else {
                f64::INFINITY
            };
            let inv_death = if death > 0.0 { 1.0 / death } else { 0.0 };
            (inv_birth - inv_death).max(0.0)
        })
        .collect();

    ClusterSelectionResult {
        selected_clusters: selected,
        cluster_persistence,
    }
}

struct ClusterChildren {
    offsets: Vec<usize>,
    children: Vec<usize>,
}

impl ClusterChildren {
    fn new(condensed_tree: &[CondensedTreeEdge], n_points: usize, mut counts: Vec<usize>) -> Self {
        let mut offsets = Vec::with_capacity(counts.len() + 1);
        offsets.push(0);
        for &count in &counts {
            offsets.push(offsets[offsets.len() - 1] + count);
        }

        let mut children = vec![0usize; offsets[offsets.len() - 1]];
        counts.copy_from_slice(&offsets[..offsets.len() - 1]);
        for edge in condensed_tree {
            if edge.child >= n_points {
                children[counts[edge.parent]] = edge.child;
                counts[edge.parent] += 1;
            }
        }

        ClusterChildren { offsets, children }
    }

    fn children(&self, cluster: usize) -> &[usize] {
        if cluster + 1 >= self.offsets.len() {
            return &[];
        }
        &self.children[self.offsets[cluster]..self.offsets[cluster + 1]]
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
    all_clusters: &[usize],
    children_of: &ClusterChildren,
    _leaf_clusters: &HashSet<usize>,
    allow_single_cluster: bool,
) -> HashSet<usize> {
    let root = all_clusters.iter().copied().min().unwrap_or(n_points);

    // Compute birth lambda for each node (point or cluster).
    // In sklearn: births[child] = edge.value for each edge, then births[root] = 0.
    // Each child appears exactly once as a child in the condensed tree.
    let node_count = condensed_tree
        .iter()
        .map(|edge| edge.parent.max(edge.child))
        .max()
        .map_or(n_points + 1, |max_node| max_node + 1);
    let mut births = vec![0.0; node_count];
    for edge in condensed_tree {
        births[edge.child] = edge.lambda_val;
    }
    births[root] = 0.0;

    // Compute stability for each cluster.
    // sklearn: stability[parent] += (lambda_val - births[parent]) * child_size
    // for ALL edges (both point-level and cluster-level).
    let mut stability = vec![0.0; node_count];

    for edge in condensed_tree {
        let parent = edge.parent;
        let bl = births[parent];
        let contribution = (edge.lambda_val - bl) * edge.child_size as f64;
        stability[parent] += contribution;
    }

    // Build the node list for EOM processing.
    // sklearn: if allow_single_cluster, include all; otherwise exclude root.
    let mut node_list: Vec<usize> = if allow_single_cluster {
        all_clusters.to_vec()
    } else {
        all_clusters
            .iter()
            .copied()
            .filter(|&c| c != root)
            .collect()
    };
    // Process in reverse topological order (highest ID = deepest first)
    node_list.sort_unstable_by(|a, b| b.cmp(a));

    // is_cluster tracks which nodes are selected (all start as true)
    let mut is_cluster = vec![false; node_count];
    for &c in &node_list {
        is_cluster[c] = true;
    }

    // Bottom-up pass: for each node, compare its stability to sum of children's stability.
    // If children win, set node to not-a-cluster and propagate children's stability up.
    // If node wins, deselect all descendants.
    // Note: stability dict is mutated in place (like sklearn).
    let mut descendant_stack = Vec::new();
    for &node in &node_list {
        let children = children_of.children(node);
        if !children.is_empty() {
            let subtree_stability: f64 = children.iter().map(|&c| stability[c]).sum();

            let own_stability = stability[node];

            if subtree_stability > own_stability {
                // Children are collectively better
                is_cluster[node] = false;
                stability[node] = subtree_stability;
            } else {
                // This node is better: deselect all descendants
                mark_descendants_not_cluster(
                    node,
                    children_of,
                    &mut is_cluster,
                    &mut descendant_stack,
                );
            }
        }
    }

    let selected: HashSet<usize> = node_list
        .iter()
        .copied()
        .filter(|&cluster| is_cluster[cluster])
        .collect();

    // Handle allow_single_cluster edge case:
    // If only the root is selected and allow_single_cluster is false, return empty
    if !allow_single_cluster && selected.len() == 1 && selected.contains(&root) {
        return HashSet::new();
    }

    selected
}

fn mark_descendants_not_cluster(
    node: usize,
    children_of: &ClusterChildren,
    is_cluster: &mut [bool],
    stack: &mut Vec<usize>,
) {
    stack.clear();
    stack.extend(children_of.children(node).iter().copied());
    while let Some(current) = stack.pop() {
        is_cluster[current] = false;
        stack.extend(children_of.children(current).iter().copied());
    }
}

/// Apply epsilon merging: merge selected clusters whose split distance < epsilon.
fn apply_epsilon_merging(
    selected: &HashSet<usize>,
    _condensed_tree: &[CondensedTreeEdge],
    _n_points: usize,
    children_of: &ClusterChildren,
    birth_lambda: &[f64],
    epsilon: f64,
) -> HashSet<usize> {
    let epsilon_lambda = if epsilon > 0.0 {
        1.0 / epsilon
    } else {
        f64::INFINITY
    };
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
            let children = children_of.children(cluster);
            if !children.is_empty() {
                // Check if all children are selected and born at lambda > epsilon_lambda
                let all_children_selected_and_fine = children.iter().all(|c| {
                    result.contains(c)
                        && birth_lambda.get(*c).copied().unwrap_or(0.0) > epsilon_lambda
                });
                if all_children_selected_and_fine {
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
