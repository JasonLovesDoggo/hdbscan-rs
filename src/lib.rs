//! # hdbscan-rs
//!
//! A Rust implementation of the HDBSCAN (Hierarchical Density-Based Spatial
//! Clustering of Applications with Noise) algorithm, producing results
//! compatible with scikit-learn's HDBSCAN.

pub mod core_distance;
pub mod distance;
pub mod error;
pub mod kdtree;
pub mod mst;
pub mod params;
pub mod types;
pub mod union_find;

// Re-exports
pub use error::HdbscanError;
pub use params::{HdbscanParams, Metric};
