use crate::distance;
use crate::error::HdbscanError;
use crate::params::Metric;
use ndarray::{ArrayView1, ArrayView2, Array1};

/// Data cached from fitting, needed for approximate prediction.
pub struct PredictionData {
    /// The training data
    pub training_data: ndarray::Array2<f64>,
    /// Core distances of training points
    pub core_distances: Array1<f64>,
    /// Cluster labels from fit
    pub labels: Vec<i32>,
    /// Cluster membership probabilities from fit
    pub probabilities: Vec<f64>,
    /// Number of features
    pub n_features: usize,
    /// The metric used
    pub metric: Metric,
    /// min_samples used
    pub min_samples: usize,
}

/// Approximate prediction for new points.
///
/// For each new point:
/// 1. Find its nearest mutual-reachability neighbor in the training data
/// 2. Assign it to that neighbor's cluster
/// 3. Compute a membership probability based on the lambda at which it would join
pub fn approximate_predict(
    prediction_data: &PredictionData,
    points: &ArrayView2<f64>,
) -> Result<(Vec<i32>, Vec<f64>), HdbscanError> {
    let n_new = points.nrows();
    if n_new == 0 {
        return Ok((vec![], vec![]));
    }

    if points.ncols() != prediction_data.n_features {
        return Err(HdbscanError::DimensionMismatch {
            expected: prediction_data.n_features,
            got: points.ncols(),
        });
    }

    let training = &prediction_data.training_data;
    let core_dists = &prediction_data.core_distances;
    let n_train = training.nrows();

    let mut labels = Vec::with_capacity(n_new);
    let mut probabilities = Vec::with_capacity(n_new);

    for i in 0..n_new {
        let point = points.row(i);

        // Find nearest training point by mutual reachability distance
        let mut best_mr_dist = f64::INFINITY;
        let mut best_idx = 0;

        for j in 0..n_train {
            let raw_dist = compute_dist(&point, &training.row(j), &prediction_data.metric);
            let mr_dist = f64::max(core_dists[j], raw_dist);
            if mr_dist < best_mr_dist {
                best_mr_dist = mr_dist;
                best_idx = j;
            }
        }

        labels.push(prediction_data.labels[best_idx]);

        // Probability: based on the neighbor's probability, scaled by distance
        if prediction_data.labels[best_idx] >= 0 {
            let neighbor_prob = prediction_data.probabilities[best_idx];
            // Simple heuristic: if the mutual reachability distance is the same
            // as the core distance, the point is "inside" the cluster
            if best_mr_dist <= core_dists[best_idx] {
                probabilities.push(neighbor_prob);
            } else {
                // Scale down probability by ratio of core distance to mutual reachability distance
                let lambda_new = if best_mr_dist > 0.0 {
                    1.0 / best_mr_dist
                } else {
                    f64::INFINITY
                };
                let lambda_core = if core_dists[best_idx] > 0.0 {
                    1.0 / core_dists[best_idx]
                } else {
                    f64::INFINITY
                };
                if lambda_core.is_infinite() {
                    probabilities.push(neighbor_prob);
                } else {
                    let ratio = (lambda_new / lambda_core).min(1.0);
                    probabilities.push(neighbor_prob * ratio);
                }
            }
        } else {
            probabilities.push(0.0);
        }
    }

    Ok((labels, probabilities))
}

#[inline]
fn compute_dist(a: &ArrayView1<f64>, b: &ArrayView1<f64>, metric: &Metric) -> f64 {
    distance::compute_distance(a, b, metric)
}
