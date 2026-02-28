/// A single edge in the minimum spanning tree.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MstEdge {
    pub u: usize,
    pub v: usize,
    pub weight: f64,
}

/// A single merge step in the single-linkage dendrogram.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SingleLinkageMerge {
    pub left: usize,
    pub right: usize,
    pub distance: f64,
    pub size: usize,
}

/// An edge in the condensed tree.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CondensedTreeEdge {
    pub parent: usize,
    pub child: usize,
    pub lambda_val: f64,
    pub child_size: usize,
}
