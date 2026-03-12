use crate::types::{MstEdge, SingleLinkageMerge};
use crate::union_find::UnionFind;
use ordered_float::OrderedFloat;

/// Convert a minimum spanning tree into a single-linkage dendrogram.
///
/// Sorts MST edges by weight and processes them with union-find.
/// Tied edges (equal weight) are batched together to avoid ordering artifacts.
pub fn mst_to_single_linkage(edges: &[MstEdge], n_points: usize) -> Vec<SingleLinkageMerge> {
    if edges.is_empty() {
        return vec![];
    }

    // Sort edges directly for cache-friendly access (avoids indirection through index array)
    let mut sorted_edges: Vec<MstEdge> = edges.to_vec();
    sorted_edges.sort_unstable_by(|ea, eb| {
        OrderedFloat(ea.weight)
            .cmp(&OrderedFloat(eb.weight))
            .then_with(|| ea.u.min(ea.v).cmp(&eb.u.min(eb.v)))
            .then_with(|| ea.u.max(ea.v).cmp(&eb.u.max(eb.v)))
    });

    let mut uf = UnionFind::new(n_points);
    let mut merges = Vec::with_capacity(edges.len());

    // Process edges in batches of equal weight (tied edge handling)
    let mut i = 0;
    while i < sorted_edges.len() {
        let current_weight = sorted_edges[i].weight;
        let batch_start = i;

        // Find end of batch with same weight
        while i < sorted_edges.len() && sorted_edges[i].weight == current_weight {
            i += 1;
        }

        // Process all edges in this batch
        for edge in &sorted_edges[batch_start..i] {
            let root_u = uf.find(edge.u);
            let root_v = uf.find(edge.v);

            if root_u != root_v {
                let size = uf.size_of(edge.u) + uf.size_of(edge.v);
                let (left, right) = if root_u < root_v {
                    (root_u, root_v)
                } else {
                    (root_v, root_u)
                };
                uf.union(root_u, root_v);
                merges.push(SingleLinkageMerge {
                    left,
                    right,
                    distance: edge.weight,
                    size,
                });
            }
        }
    }

    merges
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_linkage_basic() {
        let edges = vec![
            MstEdge {
                u: 0,
                v: 1,
                weight: 1.0,
            },
            MstEdge {
                u: 1,
                v: 2,
                weight: 2.0,
            },
            MstEdge {
                u: 2,
                v: 3,
                weight: 3.0,
            },
        ];
        let merges = mst_to_single_linkage(&edges, 4);
        assert_eq!(merges.len(), 3);
        assert_eq!(merges[0].distance, 1.0);
        assert_eq!(merges[0].size, 2);
        assert_eq!(merges[1].distance, 2.0);
        assert_eq!(merges[1].size, 3);
        assert_eq!(merges[2].distance, 3.0);
        assert_eq!(merges[2].size, 4);
    }

    #[test]
    fn test_tied_edges() {
        // All edges have the same weight
        let edges = vec![
            MstEdge {
                u: 0,
                v: 1,
                weight: 1.0,
            },
            MstEdge {
                u: 2,
                v: 3,
                weight: 1.0,
            },
            MstEdge {
                u: 1,
                v: 2,
                weight: 1.0,
            },
        ];
        let merges = mst_to_single_linkage(&edges, 4);
        assert_eq!(merges.len(), 3);
        // All merges at distance 1.0
        for m in &merges {
            assert_eq!(m.distance, 1.0);
        }
        // Final merge should have size 4
        assert_eq!(merges[2].size, 4);
    }
}
