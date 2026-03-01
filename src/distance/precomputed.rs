use crate::error::HdbscanError;
use ndarray::ArrayView2;

/// Validate a precomputed distance matrix.
pub fn validate_precomputed(data: &ArrayView2<f64>) -> Result<(), HdbscanError> {
    let (rows, cols) = data.dim();
    if rows != cols {
        return Err(HdbscanError::NonSquareDistanceMatrix { rows, cols });
    }
    for val in data.iter() {
        if val.is_nan() || val.is_infinite() {
            return Err(HdbscanError::InvalidData);
        }
        if *val < 0.0 {
            return Err(HdbscanError::NegativeDistances);
        }
    }
    Ok(())
}
