use crate::error::HdbscanError;

/// Distance metric for HDBSCAN.
#[derive(Debug, Clone)]
pub enum Metric {
    Euclidean,
    Manhattan,
    Cosine,
    Minkowski(f64),
    Precomputed,
}

impl Default for Metric {
    fn default() -> Self {
        Metric::Euclidean
    }
}

/// Cluster selection method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClusterSelectionMethod {
    /// Excess of Mass (default in sklearn)
    Eom,
    /// Leaf clusters
    Leaf,
}

impl Default for ClusterSelectionMethod {
    fn default() -> Self {
        ClusterSelectionMethod::Eom
    }
}

/// What centers to store after fitting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreCenters {
    Centroid,
    Medoid,
    Both,
}

/// Parameters for HDBSCAN clustering.
#[derive(Debug, Clone)]
pub struct HdbscanParams {
    pub min_cluster_size: usize,
    pub min_samples: Option<usize>,
    pub metric: Metric,
    pub alpha: f64,
    pub cluster_selection_epsilon: f64,
    pub cluster_selection_method: ClusterSelectionMethod,
    pub allow_single_cluster: bool,
    pub store_centers: Option<StoreCenters>,
}

impl Default for HdbscanParams {
    fn default() -> Self {
        HdbscanParams {
            min_cluster_size: 5,
            min_samples: None,
            metric: Metric::default(),
            alpha: 1.0,
            cluster_selection_epsilon: 0.0,
            cluster_selection_method: ClusterSelectionMethod::default(),
            allow_single_cluster: false,
            store_centers: None,
        }
    }
}

impl HdbscanParams {
    pub fn validate(&self) -> Result<(), HdbscanError> {
        if self.min_cluster_size < 2 {
            return Err(HdbscanError::InvalidMinClusterSize(self.min_cluster_size));
        }
        if let Some(ms) = self.min_samples {
            if ms < 1 {
                return Err(HdbscanError::InvalidMinSamples(ms));
            }
        }
        if let Metric::Minkowski(p) = &self.metric {
            if *p < 1.0 || p.is_nan() {
                return Err(HdbscanError::InvalidMinkowskiP(*p));
            }
        }
        Ok(())
    }

    /// Effective min_samples (defaults to min_cluster_size).
    pub fn effective_min_samples(&self) -> usize {
        self.min_samples.unwrap_or(self.min_cluster_size)
    }
}

/// Builder for HdbscanParams.
pub struct HdbscanBuilder {
    params: HdbscanParams,
}

impl HdbscanBuilder {
    pub fn new() -> Self {
        HdbscanBuilder {
            params: HdbscanParams::default(),
        }
    }

    pub fn min_cluster_size(mut self, v: usize) -> Self {
        self.params.min_cluster_size = v;
        self
    }

    pub fn min_samples(mut self, v: usize) -> Self {
        self.params.min_samples = Some(v);
        self
    }

    pub fn metric(mut self, v: Metric) -> Self {
        self.params.metric = v;
        self
    }

    pub fn alpha(mut self, v: f64) -> Self {
        self.params.alpha = v;
        self
    }

    pub fn cluster_selection_epsilon(mut self, v: f64) -> Self {
        self.params.cluster_selection_epsilon = v;
        self
    }

    pub fn cluster_selection_method(mut self, v: ClusterSelectionMethod) -> Self {
        self.params.cluster_selection_method = v;
        self
    }

    pub fn allow_single_cluster(mut self, v: bool) -> Self {
        self.params.allow_single_cluster = v;
        self
    }

    pub fn store_centers(mut self, v: StoreCenters) -> Self {
        self.params.store_centers = Some(v);
        self
    }

    pub fn build(self) -> Result<HdbscanParams, HdbscanError> {
        self.params.validate()?;
        Ok(self.params)
    }
}

impl Default for HdbscanBuilder {
    fn default() -> Self {
        Self::new()
    }
}
