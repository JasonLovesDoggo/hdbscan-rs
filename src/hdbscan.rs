use ndarray::{Array2, ArrayView2};

use crate::centers;
use crate::cluster_selection;
use crate::condensed_tree;
use crate::core_distance;
use crate::distance::precomputed;
use crate::error::HdbscanError;
use crate::labels;
use crate::linkage;
use crate::membership;
use crate::mst;
use crate::outlier;
use crate::params::{HdbscanBuilder, HdbscanParams, Metric};
use crate::prediction::{self, PredictionData};
use crate::types::CondensedTreeEdge;

/// HDBSCAN clustering algorithm.
///
/// Produces results compatible with scikit-learn's HDBSCAN implementation.
pub struct Hdbscan {
    params: HdbscanParams,
    labels_: Option<Vec<i32>>,
    probabilities_: Option<Vec<f64>>,
    outlier_scores_: Option<Vec<f64>>,
    condensed_tree_: Option<Vec<CondensedTreeEdge>>,
    cluster_persistence_: Option<Vec<f64>>,
    centroids_: Option<Array2<f64>>,
    medoids_: Option<Array2<f64>>,
    prediction_data_: Option<PredictionData>,
}

impl Hdbscan {
    /// Create a new HDBSCAN instance with the given parameters.
    pub fn new(params: HdbscanParams) -> Self {
        Hdbscan {
            params,
            labels_: None,
            probabilities_: None,
            outlier_scores_: None,
            condensed_tree_: None,
            cluster_persistence_: None,
            centroids_: None,
            medoids_: None,
            prediction_data_: None,
        }
    }

    /// Create a builder for configuring HDBSCAN parameters.
    pub fn builder() -> HdbscanBuilder {
        HdbscanBuilder::new()
    }

    /// Fit the model to data.
    pub fn fit(&mut self, data: &ArrayView2<f64>) -> Result<(), HdbscanError> {
        self.params.validate()?;
        self.validate_data(data)?;

        let n_points = data.nrows();
        let min_samples = self.params.effective_min_samples();

        if min_samples > n_points {
            return Err(HdbscanError::MinSamplesExceedsData {
                min_samples,
                n_points,
            });
        }

        // Step 1+2: Compute core distances and build MST.
        let (core_distances, mst_edges) = self.compute_core_and_mst(data, min_samples);

        // Step 3: Build single-linkage tree
        let single_linkage = linkage::mst_to_single_linkage(&mst_edges, n_points);

        // Step 4: Build condensed tree
        let condensed = condensed_tree::build_condensed_tree(
            &single_linkage,
            n_points,
            self.params.min_cluster_size,
        );

        // Step 5: Select clusters
        let selection = cluster_selection::select_clusters(
            &condensed,
            n_points,
            self.params.cluster_selection_method,
            self.params.cluster_selection_epsilon,
            self.params.allow_single_cluster,
        );

        // Step 6: Assign labels
        let point_labels = labels::assign_labels(
            &condensed,
            &selection.selected_clusters,
            n_points,
            self.params.allow_single_cluster,
            self.params.cluster_selection_epsilon,
        );

        let n_clusters = if point_labels.is_empty() {
            0
        } else {
            (*point_labels.iter().max().unwrap_or(&-1) + 1).max(0) as usize
        };

        // Step 7: Compute probabilities
        let probs = membership::compute_probabilities(
            &condensed,
            &selection.selected_clusters,
            &point_labels,
            n_points,
        );

        // Step 8: Compute outlier scores
        let outlier_scores = outlier::compute_outlier_scores(&condensed, n_points);

        // Step 9: Compute centers if requested
        if let Some(store) = self.params.store_centers {
            if n_clusters > 0 {
                let (centroids, medoids) =
                    centers::compute_centers(data, &point_labels, n_clusters, store);
                self.centroids_ = centroids;
                self.medoids_ = medoids;
            }
        }

        // Step 10: Cache prediction data
        self.prediction_data_ = Some(PredictionData {
            training_data: data.to_owned(),
            core_distances,
            labels: point_labels.clone(),
            probabilities: probs.clone(),
            n_features: data.ncols(),
            metric: self.params.metric.clone(),
            min_samples,
        });

        self.labels_ = Some(point_labels);
        self.probabilities_ = Some(probs);
        self.outlier_scores_ = Some(outlier_scores);
        self.condensed_tree_ = Some(condensed);
        self.cluster_persistence_ = Some(selection.cluster_persistence);

        Ok(())
    }

    /// Fit the model and return cluster labels.
    pub fn fit_predict(&mut self, data: &ArrayView2<f64>) -> Result<Vec<i32>, HdbscanError> {
        self.fit(data)?;
        Ok(self.labels_.clone().unwrap())
    }

    /// Predict cluster labels for new points using approximate prediction.
    pub fn approximate_predict(
        &self,
        points: &ArrayView2<f64>,
    ) -> Result<(Vec<i32>, Vec<f64>), HdbscanError> {
        let pred_data = self
            .prediction_data_
            .as_ref()
            .ok_or(HdbscanError::NotFitted)?;
        prediction::approximate_predict(pred_data, points)
    }

    /// Get cluster labels (None if not fitted).
    pub fn labels(&self) -> Option<&[i32]> {
        self.labels_.as_deref()
    }

    /// Get membership probabilities (None if not fitted).
    pub fn probabilities(&self) -> Option<&[f64]> {
        self.probabilities_.as_deref()
    }

    /// Get GLOSH outlier scores (None if not fitted).
    pub fn outlier_scores(&self) -> Option<&[f64]> {
        self.outlier_scores_.as_deref()
    }

    /// Get the condensed tree (None if not fitted).
    pub fn condensed_tree(&self) -> Option<&[CondensedTreeEdge]> {
        self.condensed_tree_.as_deref()
    }

    /// Get cluster persistence values (None if not fitted).
    pub fn cluster_persistence(&self) -> Option<&[f64]> {
        self.cluster_persistence_.as_deref()
    }

    /// Get cluster centroids (None if not fitted or not requested).
    pub fn centroids(&self) -> Option<&Array2<f64>> {
        self.centroids_.as_ref()
    }

    /// Get cluster medoids (None if not fitted or not requested).
    pub fn medoids(&self) -> Option<&Array2<f64>> {
        self.medoids_.as_ref()
    }

    /// Compute core distances and MST, sharing tree construction where possible.
    fn compute_core_and_mst(
        &self,
        data: &ArrayView2<f64>,
        min_samples: usize,
    ) -> (ndarray::Array1<f64>, Vec<crate::types::MstEdge>) {
        use crate::ball_tree::BallTree;
        use crate::kdtree_bounded::BoundedKdTree;

        let n = data.nrows();
        let dim = data.ncols();
        let threshold = mst::dual_tree_threshold(dim);

        let use_prims = !matches!(self.params.metric, Metric::Euclidean) || n <= threshold;

        if use_prims {
            // Fused core+Prim's: compute all pairwise distances once (GEMM for high dim,
            // SIMD for low dim), extract core distances, then run Prim's with O(1) lookups.
            // Only worthwhile at dim > 16 where per-distance cost justifies caching.
            if matches!(self.params.metric, Metric::Euclidean)
                && self.params.alpha == 1.0
                && dim > 16
            {
                // High dim, small n: fused GEMM+Prim's (GEMM efficient, cached lookups)
                mst::prim::fused_core_and_prim(data, min_samples)
            } else {
                let (core_distances, nn_indices) = core_distance::compute_core_distances_with_nn(
                    data,
                    &self.params.metric,
                    min_samples,
                );
                let mst_edges = mst::auto_mst(
                    data,
                    &core_distances.view(),
                    &self.params.metric,
                    self.params.alpha,
                    Some(&nn_indices),
                );
                (core_distances, mst_edges)
            }
        } else if dim <= 16 {
            // Share bounded kd-tree between core distances and Boruvka MST
            let tree = BoundedKdTree::build(data);
            let (core_distances, nn_indices) =
                core_distance::compute_core_distances_with_bounded_kdtree(&tree, data, min_samples);
            let mst_edges = mst::dual_tree_boruvka::dual_tree_boruvka_mst(
                &tree,
                &core_distances.view(),
                self.params.alpha,
                Some(&nn_indices),
            );
            (core_distances, mst_edges)
        } else {
            // Share ball tree between core distances and Boruvka MST
            let tree = BallTree::build(data);
            let (core_distances, nn_indices) =
                core_distance::compute_core_distances_with_balltree(&tree, data, min_samples);
            let mst_edges = mst::dual_tree_boruvka::dual_tree_boruvka_mst(
                &tree,
                &core_distances.view(),
                self.params.alpha,
                Some(&nn_indices),
            );
            (core_distances, mst_edges)
        }
    }

    fn validate_data(&self, data: &ArrayView2<f64>) -> Result<(), HdbscanError> {
        if data.nrows() == 0 {
            return Err(HdbscanError::EmptyData);
        }

        match &self.params.metric {
            Metric::Precomputed => {
                precomputed::validate_precomputed(data)?;
            }
            _ => {
                // Check for NaN/Inf
                for val in data.iter() {
                    if val.is_nan() || val.is_infinite() {
                        return Err(HdbscanError::InvalidData);
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_basic_clustering() {
        // Two well-separated clusters
        let data = array![
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [0.1, 0.1],
            [0.05, 0.05],
            [10.0, 10.0],
            [10.1, 10.0],
            [10.0, 10.1],
            [10.1, 10.1],
            [10.05, 10.05],
        ];

        let params = HdbscanParams {
            min_cluster_size: 3,
            ..Default::default()
        };
        let mut hdbscan = Hdbscan::new(params);
        let labels = hdbscan.fit_predict(&data.view()).unwrap();

        // Should find 2 clusters
        let n_clusters = *labels.iter().max().unwrap() + 1;
        assert_eq!(n_clusters, 2, "Expected 2 clusters, got {}", n_clusters);

        // Points in first group should share a label
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[0], labels[2]);

        // Points in second group should share a label
        assert_eq!(labels[5], labels[6]);
        assert_eq!(labels[5], labels[7]);

        // The two groups should have different labels
        assert_ne!(labels[0], labels[5]);
    }

    #[test]
    fn test_all_noise_small_dataset() {
        // 4 scattered points with min_cluster_size=5 -> all noise
        let data = array![[0.0], [100.0], [200.0], [300.0]];
        let params = HdbscanParams {
            min_cluster_size: 5,
            min_samples: Some(2),
            ..Default::default()
        };
        let mut hdbscan = Hdbscan::new(params);
        let labels = hdbscan.fit_predict(&data.view()).unwrap();
        // Everything should be noise
        assert!(labels.iter().all(|&l| l == -1));
    }

    #[test]
    fn test_min_samples_exceeds_data() {
        let data = array![[0.0], [100.0]];
        let params = HdbscanParams {
            min_cluster_size: 5,
            ..Default::default()
        };
        let mut hdbscan = Hdbscan::new(params);
        assert!(matches!(
            hdbscan.fit(&data.view()),
            Err(HdbscanError::MinSamplesExceedsData { .. })
        ));
    }

    #[test]
    fn test_empty_data() {
        let data = Array2::<f64>::zeros((0, 2));
        let mut hdbscan = Hdbscan::new(HdbscanParams::default());
        assert!(hdbscan.fit(&data.view()).is_err());
    }

    #[test]
    fn test_nan_data() {
        let data = array![[1.0, f64::NAN], [2.0, 3.0]];
        let mut hdbscan = Hdbscan::new(HdbscanParams::default());
        assert!(hdbscan.fit(&data.view()).is_err());
    }

    #[test]
    fn test_not_fitted_predict() {
        let hdbscan = Hdbscan::new(HdbscanParams::default());
        let points = array![[1.0, 2.0]];
        assert!(hdbscan.approximate_predict(&points.view()).is_err());
    }

    #[test]
    fn test_probabilities_and_outlier_scores() {
        let data = array![
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [0.1, 0.1],
            [0.05, 0.05],
            [10.0, 10.0],
            [10.1, 10.0],
            [10.0, 10.1],
            [10.1, 10.1],
            [10.05, 10.05],
        ];

        let params = HdbscanParams {
            min_cluster_size: 3,
            ..Default::default()
        };
        let mut hdbscan = Hdbscan::new(params);
        hdbscan.fit(&data.view()).unwrap();

        let probs = hdbscan.probabilities().unwrap();
        assert_eq!(probs.len(), 10);
        // All probabilities should be in [0, 1]
        for &p in probs {
            assert!(p >= 0.0 && p <= 1.0);
        }

        let scores = hdbscan.outlier_scores().unwrap();
        assert_eq!(scores.len(), 10);
        for &s in scores {
            assert!(s >= 0.0 && s <= 1.0);
        }
    }

    #[test]
    fn test_builder() {
        let params = Hdbscan::builder()
            .min_cluster_size(10)
            .min_samples(5)
            .build()
            .unwrap();
        assert_eq!(params.min_cluster_size, 10);
        assert_eq!(params.min_samples, Some(5));
    }

    #[test]
    fn test_deterministic() {
        let data = array![
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [0.1, 0.1],
            [0.05, 0.05],
            [10.0, 10.0],
            [10.1, 10.0],
            [10.0, 10.1],
            [10.1, 10.1],
            [10.05, 10.05],
        ];

        let params = HdbscanParams {
            min_cluster_size: 3,
            ..Default::default()
        };

        let mut h1 = Hdbscan::new(params.clone());
        let labels1 = h1.fit_predict(&data.view()).unwrap();

        let mut h2 = Hdbscan::new(params);
        let labels2 = h2.fit_predict(&data.view()).unwrap();

        assert_eq!(labels1, labels2);
    }

    #[test]
    fn test_duplicate_points() {
        // All identical points
        let data = array![[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0],];
        let params = HdbscanParams {
            min_cluster_size: 3,
            ..Default::default()
        };
        let mut hdbscan = Hdbscan::new(params);
        // Should not panic
        let result = hdbscan.fit_predict(&data.view());
        assert!(result.is_ok());
    }
}
