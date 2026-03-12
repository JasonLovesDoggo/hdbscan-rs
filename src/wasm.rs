//! WebAssembly bindings for hdbscan-rs via wasm-bindgen.
//!
//! Accepts flat Float64Array data with explicit row/column counts.
//! All methods return typed arrays directly usable from JavaScript.

use ndarray::Array2;
use wasm_bindgen::prelude::*;

use crate::hdbscan::Hdbscan as RustHdbscan;
use crate::params::{ClusterSelectionMethod, HdbscanParams, Metric};

fn to_js_err(e: crate::error::HdbscanError) -> JsValue {
    JsValue::from_str(&e.to_string())
}

fn parse_metric(metric: &str, p: Option<f64>) -> Result<Metric, JsValue> {
    match metric {
        "euclidean" => Ok(Metric::Euclidean),
        "manhattan" | "l1" | "cityblock" => Ok(Metric::Manhattan),
        "cosine" => Ok(Metric::Cosine),
        "minkowski" => Ok(Metric::Minkowski(p.unwrap_or(2.0))),
        "precomputed" => Ok(Metric::Precomputed),
        _ => Err(JsValue::from_str(&format!(
            "Unknown metric: '{}'. Expected: euclidean, manhattan, cosine, minkowski, precomputed",
            metric
        ))),
    }
}

fn parse_method(method: &str) -> Result<ClusterSelectionMethod, JsValue> {
    match method {
        "eom" => Ok(ClusterSelectionMethod::Eom),
        "leaf" => Ok(ClusterSelectionMethod::Leaf),
        _ => Err(JsValue::from_str(&format!(
            "Unknown cluster_selection_method: '{}'. Expected 'eom' or 'leaf'",
            method
        ))),
    }
}

fn data_to_array2(
    data: &[f64],
    n_samples: usize,
    n_features: usize,
) -> Result<Array2<f64>, JsValue> {
    if data.len() != n_samples * n_features {
        return Err(JsValue::from_str(&format!(
            "Data length {} does not match {} x {} = {}",
            data.len(),
            n_samples,
            n_features,
            n_samples * n_features
        )));
    }
    Array2::from_shape_vec((n_samples, n_features), data.to_vec())
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// HDBSCAN clustering for WebAssembly.
///
/// @example
/// ```js
/// import init, { HDBSCAN } from 'hdbscan-rs';
/// await init();
///
/// const clusterer = new HDBSCAN(5);
/// const data = new Float64Array([0,0, 0,1, 1,0, 10,10, 10,11, 11,10]);
/// const labels = clusterer.fit_predict(data, 6, 2);
/// ```
#[wasm_bindgen]
#[allow(clippy::upper_case_acronyms)]
pub struct HDBSCAN {
    inner: RustHdbscan,
}

#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
impl HDBSCAN {
    /// Create a new HDBSCAN clusterer.
    ///
    /// @param min_cluster_size - Minimum cluster size (default: 5)
    /// @param min_samples - Min samples for core distance (default: min_cluster_size)
    /// @param metric - Distance metric: "euclidean", "manhattan", "cosine", "minkowski", "precomputed"
    /// @param p - Minkowski p parameter (only used when metric="minkowski")
    /// @param alpha - Distance scaling factor (default: 1.0)
    /// @param cluster_selection_epsilon - Epsilon for merging clusters (default: 0.0)
    /// @param cluster_selection_method - "eom" or "leaf" (default: "eom")
    /// @param allow_single_cluster - Allow single cluster result (default: false)
    #[wasm_bindgen(constructor)]
    pub fn new(
        min_cluster_size: Option<usize>,
        min_samples: Option<usize>,
        metric: Option<String>,
        p: Option<f64>,
        alpha: Option<f64>,
        cluster_selection_epsilon: Option<f64>,
        cluster_selection_method: Option<String>,
        allow_single_cluster: Option<bool>,
    ) -> Result<HDBSCAN, JsValue> {
        let metric_str = metric.as_deref().unwrap_or("euclidean");
        let method_str = cluster_selection_method.as_deref().unwrap_or("eom");

        let params = HdbscanParams {
            min_cluster_size: min_cluster_size.unwrap_or(5),
            min_samples,
            metric: parse_metric(metric_str, p)?,
            alpha: alpha.unwrap_or(1.0),
            cluster_selection_epsilon: cluster_selection_epsilon.unwrap_or(0.0),
            cluster_selection_method: parse_method(method_str)?,
            allow_single_cluster: allow_single_cluster.unwrap_or(false),
            store_centers: None,
        };
        params.validate().map_err(to_js_err)?;

        Ok(HDBSCAN {
            inner: RustHdbscan::new(params),
        })
    }

    /// Fit the model and return cluster labels as Int32Array.
    ///
    /// @param data - Flat Float64Array of shape [n_samples * n_features]
    /// @param n_samples - Number of data points
    /// @param n_features - Number of dimensions per point
    /// @returns Int32Array of cluster labels (-1 = noise)
    pub fn fit_predict(
        &mut self,
        data: &[f64],
        n_samples: usize,
        n_features: usize,
    ) -> Result<Vec<i32>, JsValue> {
        let arr = data_to_array2(data, n_samples, n_features)?;
        self.inner.fit_predict(&arr.view()).map_err(to_js_err)
    }

    /// Fit the model without returning labels.
    pub fn fit(
        &mut self,
        data: &[f64],
        n_samples: usize,
        n_features: usize,
    ) -> Result<(), JsValue> {
        let arr = data_to_array2(data, n_samples, n_features)?;
        self.inner.fit(&arr.view()).map_err(to_js_err)
    }

    /// Predict cluster labels for new points. Must call fit() first.
    ///
    /// @returns Object with `labels: Int32Array` and `probabilities: Float64Array`
    pub fn approximate_predict(
        &self,
        data: &[f64],
        n_samples: usize,
        n_features: usize,
    ) -> Result<JsValue, JsValue> {
        let arr = data_to_array2(data, n_samples, n_features)?;
        let (labels, probs) = self
            .inner
            .approximate_predict(&arr.view())
            .map_err(to_js_err)?;

        let obj = js_sys::Object::new();
        let labels_arr = js_sys::Int32Array::from(&labels[..]);
        let probs_arr = js_sys::Float64Array::from(&probs[..]);
        js_sys::Reflect::set(&obj, &"labels".into(), &labels_arr)?;
        js_sys::Reflect::set(&obj, &"probabilities".into(), &probs_arr)?;
        Ok(obj.into())
    }

    /// Get cluster labels after fitting. Returns null if not fitted.
    pub fn labels(&self) -> Option<Vec<i32>> {
        self.inner.labels().map(|l| l.to_vec())
    }

    /// Get membership probabilities after fitting. Returns null if not fitted.
    pub fn probabilities(&self) -> Option<Vec<f64>> {
        self.inner.probabilities().map(|p| p.to_vec())
    }

    /// Get GLOSH outlier scores after fitting. Returns null if not fitted.
    pub fn outlier_scores(&self) -> Option<Vec<f64>> {
        self.inner.outlier_scores().map(|s| s.to_vec())
    }

    /// Get cluster persistence values after fitting. Returns null if not fitted.
    pub fn cluster_persistence(&self) -> Option<Vec<f64>> {
        self.inner.cluster_persistence().map(|p| p.to_vec())
    }
}
