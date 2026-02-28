use thiserror::Error;

#[derive(Debug, Error)]
pub enum HdbscanError {
    #[error("input data must have at least one point")]
    EmptyData,

    #[error("input data contains NaN or infinite values")]
    InvalidData,

    #[error("min_cluster_size must be at least 2, got {0}")]
    InvalidMinClusterSize(usize),

    #[error("min_samples must be at least 1, got {0}")]
    InvalidMinSamples(usize),

    #[error("min_samples ({min_samples}) exceeds number of points ({n_points})")]
    MinSamplesExceedsData {
        min_samples: usize,
        n_points: usize,
    },

    #[error("precomputed distance matrix must be square, got {rows}x{cols}")]
    NonSquareDistanceMatrix { rows: usize, cols: usize },

    #[error("precomputed distance matrix contains negative values")]
    NegativeDistances,

    #[error("model has not been fitted yet")]
    NotFitted,

    #[error("dimension mismatch: model fitted with {expected} features, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("Minkowski p must be >= 1.0, got {0}")]
    InvalidMinkowskiP(f64),
}
