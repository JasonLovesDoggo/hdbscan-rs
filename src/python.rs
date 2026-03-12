//! Python bindings for hdbscan-rs via PyO3.

use ndarray::Array2;
use numpy::{PyArray1, PyArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::error::HdbscanError;
use crate::hdbscan::Hdbscan as RustHdbscan;
use crate::params::{ClusterSelectionMethod, HdbscanParams, Metric};

fn to_py_err(e: HdbscanError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn parse_metric(metric: &str, p: Option<f64>) -> PyResult<Metric> {
    match metric {
        "euclidean" => Ok(Metric::Euclidean),
        "manhattan" | "l1" | "cityblock" => Ok(Metric::Manhattan),
        "cosine" => Ok(Metric::Cosine),
        "minkowski" => Ok(Metric::Minkowski(p.unwrap_or(2.0))),
        "precomputed" => Ok(Metric::Precomputed),
        _ => Err(PyValueError::new_err(format!(
            "Unknown metric: '{}'. Expected one of: euclidean, manhattan, cosine, minkowski, precomputed",
            metric
        ))),
    }
}

/// Extract a 2-D f64 array from a numpy ndarray, accepting f32 or f64.
fn extract_f64_array(obj: &Bound<'_, pyo3::types::PyAny>) -> PyResult<Array2<f64>> {
    // Try f64 first (zero-copy)
    if let Ok(arr) = obj.downcast::<numpy::PyArray2<f64>>() {
        return Ok(arr.readonly().as_array().to_owned());
    }
    // Try f32 and upcast
    if let Ok(arr) = obj.downcast::<numpy::PyArray2<f32>>() {
        let view = arr.readonly();
        let view = view.as_array();
        let shape = (view.nrows(), view.ncols());
        let data: Vec<f64> = view.iter().map(|&v| v as f64).collect();
        return Ok(Array2::from_shape_vec(shape, data).unwrap());
    }
    Err(PyTypeError::new_err(
        "Expected a 2-D numpy array with dtype float32 or float64",
    ))
}

fn parse_cluster_selection_method(method: &str) -> PyResult<ClusterSelectionMethod> {
    match method {
        "eom" => Ok(ClusterSelectionMethod::Eom),
        "leaf" => Ok(ClusterSelectionMethod::Leaf),
        _ => Err(PyValueError::new_err(format!(
            "Unknown cluster_selection_method: '{}'. Expected 'eom' or 'leaf'",
            method
        ))),
    }
}

/// HDBSCAN clustering algorithm.
///
/// High-performance Rust implementation with results compatible with scikit-learn.
#[pyclass]
#[allow(clippy::upper_case_acronyms)]
pub struct HDBSCAN {
    inner: RustHdbscan,
    min_cluster_size: usize,
    min_samples: Option<usize>,
    metric: String,
    #[allow(dead_code)]
    p: Option<f64>,
    alpha: f64,
    cluster_selection_epsilon: f64,
    cluster_selection_method: String,
    allow_single_cluster: bool,
    n_features_in_: Option<usize>,
}

#[pymethods]
#[allow(clippy::too_many_arguments)]
impl HDBSCAN {
    #[new]
    #[pyo3(signature = (
        min_cluster_size = 5,
        min_samples = None,
        metric = "euclidean",
        p = None,
        alpha = 1.0,
        cluster_selection_epsilon = 0.0,
        cluster_selection_method = "eom",
        allow_single_cluster = false,
    ))]
    fn new(
        min_cluster_size: usize,
        min_samples: Option<usize>,
        metric: &str,
        p: Option<f64>,
        alpha: f64,
        cluster_selection_epsilon: f64,
        cluster_selection_method: &str,
        allow_single_cluster: bool,
    ) -> PyResult<Self> {
        let rust_metric = parse_metric(metric, p)?;
        let rust_method = parse_cluster_selection_method(cluster_selection_method)?;

        let params = HdbscanParams {
            min_cluster_size,
            min_samples,
            metric: rust_metric,
            alpha,
            cluster_selection_epsilon,
            cluster_selection_method: rust_method,
            allow_single_cluster,
            store_centers: None,
        };
        params.validate().map_err(to_py_err)?;

        Ok(HDBSCAN {
            inner: RustHdbscan::new(params),
            min_cluster_size,
            min_samples,
            metric: metric.to_string(),
            p,
            alpha,
            cluster_selection_epsilon,
            cluster_selection_method: cluster_selection_method.to_string(),
            allow_single_cluster,
            n_features_in_: None,
        })
    }

    /// Fit the model to data and return cluster labels.
    ///
    /// Parameters
    /// ----------
    /// X : numpy.ndarray of shape (n_samples, n_features)
    ///     Training data.
    ///
    /// Returns
    /// -------
    /// labels : numpy.ndarray of shape (n_samples,)
    ///     Cluster labels. -1 indicates noise.
    #[pyo3(signature = (X))]
    fn fit_predict<'py>(
        &mut self,
        py: Python<'py>,
        #[allow(non_snake_case)] X: &Bound<'py, pyo3::types::PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<i32>>> {
        let data = extract_f64_array(X)?;
        self.n_features_in_ = Some(data.ncols());
        let labels = self.inner.fit_predict(&data.view()).map_err(to_py_err)?;
        Ok(PyArray1::from_vec(py, labels))
    }

    /// Fit the model to data.
    ///
    /// Parameters
    /// ----------
    /// X : numpy.ndarray of shape (n_samples, n_features)
    ///     Training data.
    ///
    /// Returns
    /// -------
    /// self : HDBSCAN
    ///     Returns self for method chaining (sklearn convention).
    #[pyo3(signature = (X))]
    fn fit(
        slf: Py<Self>,
        #[allow(non_snake_case)] X: &Bound<'_, pyo3::types::PyAny>,
    ) -> PyResult<Py<Self>> {
        let data = extract_f64_array(X)?;
        Python::with_gil(|py| -> PyResult<()> {
            let mut this = slf.borrow_mut(py);
            this.n_features_in_ = Some(data.ncols());
            this.inner.fit(&data.view()).map_err(to_py_err)?;
            Ok(())
        })?;
        Ok(slf)
    }

    /// Predict cluster labels for new points.
    ///
    /// Parameters
    /// ----------
    /// X : numpy.ndarray of shape (n_samples, n_features)
    ///     New data points.
    ///
    /// Returns
    /// -------
    /// labels : numpy.ndarray of shape (n_samples,)
    ///     Predicted cluster labels.
    /// probabilities : numpy.ndarray of shape (n_samples,)
    ///     Prediction confidence.
    #[pyo3(signature = (X))]
    #[allow(clippy::type_complexity)]
    fn approximate_predict<'py>(
        &self,
        py: Python<'py>,
        #[allow(non_snake_case)] X: &Bound<'py, pyo3::types::PyAny>,
    ) -> PyResult<(Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<f64>>)> {
        let data = extract_f64_array(X)?;
        let (labels, probs) = self
            .inner
            .approximate_predict(&data.view())
            .map_err(to_py_err)?;
        Ok((
            PyArray1::from_vec(py, labels),
            PyArray1::from_vec(py, probs),
        ))
    }

    /// Cluster labels after fitting. -1 indicates noise.
    #[getter]
    fn labels_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i32>>> {
        match self.inner.labels() {
            Some(labels) => Ok(PyArray1::from_slice(py, labels)),
            None => Err(PyValueError::new_err("Model has not been fitted yet")),
        }
    }

    /// Membership probabilities after fitting.
    #[getter]
    fn probabilities_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        match self.inner.probabilities() {
            Some(probs) => Ok(PyArray1::from_slice(py, probs)),
            None => Err(PyValueError::new_err("Model has not been fitted yet")),
        }
    }

    /// GLOSH outlier scores after fitting.
    #[getter]
    fn outlier_scores_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        match self.inner.outlier_scores() {
            Some(scores) => Ok(PyArray1::from_slice(py, scores)),
            None => Err(PyValueError::new_err("Model has not been fitted yet")),
        }
    }

    /// Cluster persistence values after fitting.
    #[getter]
    fn cluster_persistence_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        match self.inner.cluster_persistence() {
            Some(p) => Ok(PyArray1::from_slice(py, p)),
            None => Err(PyValueError::new_err("Model has not been fitted yet")),
        }
    }

    /// Number of features seen during fit.
    #[getter]
    fn n_features_in_(&self) -> PyResult<usize> {
        match self.n_features_in_ {
            Some(n) => Ok(n),
            None => Err(PyValueError::new_err("Model has not been fitted yet")),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "HDBSCAN(min_cluster_size={}, min_samples={}, metric='{}', alpha={}, \
             cluster_selection_epsilon={}, cluster_selection_method='{}', \
             allow_single_cluster={})",
            self.min_cluster_size,
            match self.min_samples {
                Some(ms) => ms.to_string(),
                None => "None".to_string(),
            },
            self.metric,
            self.alpha,
            self.cluster_selection_epsilon,
            self.cluster_selection_method,
            self.allow_single_cluster,
        )
    }
}

/// High-performance HDBSCAN clustering in Rust.
#[pymodule]
pub fn hdbscan_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<HDBSCAN>()?;
    Ok(())
}
