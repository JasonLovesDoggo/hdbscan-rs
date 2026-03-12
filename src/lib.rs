//! # hdbscan-rs
//!
//! A Rust implementation of the HDBSCAN (Hierarchical Density-Based Spatial
//! Clustering of Applications with Noise) algorithm, producing results
//! compatible with scikit-learn's HDBSCAN.
//!
//! ## Quick Start
//!
//! ```rust
//! use hdbscan_rs::{Hdbscan, HdbscanParams};
//! use ndarray::array;
//!
//! let data = array![
//!     [0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1], [0.05, 0.05],
//!     [10.0, 10.0], [10.1, 10.0], [10.0, 10.1], [10.1, 10.1], [10.05, 10.05],
//! ];
//!
//! let params = HdbscanParams { min_cluster_size: 3, ..Default::default() };
//! let mut hdbscan = Hdbscan::new(params);
//! let labels = hdbscan.fit_predict(&data.view()).unwrap();
//! ```

pub mod centers;
pub mod cluster_selection;
pub mod condensed_tree;
pub mod core_distance;
pub mod distance;
pub mod error;
pub mod hdbscan;
pub mod kdtree;
pub mod kdtree_bounded;
pub mod labels;
pub mod linkage;
pub mod membership;
pub mod mst;
pub mod outlier;
pub mod params;
pub mod prediction;
pub mod types;
pub mod union_find;

#[cfg(feature = "python")]
mod python;

// Re-exports
pub use error::HdbscanError;
pub use hdbscan::Hdbscan;
pub use params::{ClusterSelectionMethod, HdbscanBuilder, HdbscanParams, Metric, StoreCenters};
pub use types::{CondensedTreeEdge, MstEdge, SingleLinkageMerge};
