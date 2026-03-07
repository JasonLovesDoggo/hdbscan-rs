use hdbscan_rs::{Hdbscan, HdbscanError, HdbscanParams};
use ndarray::{array, Array2};
use std::collections::HashSet;

#[test]
fn test_all_identical_points() {
    let data = Array2::from_elem((10, 2), 1.0);
    let params = HdbscanParams {
        min_cluster_size: 3,
        ..Default::default()
    };
    let mut hdbscan = Hdbscan::new(params);
    let result = hdbscan.fit_predict(&data.view());
    assert!(result.is_ok());
}

#[test]
fn test_single_point_error() {
    // min_cluster_size=5, 1 point, min_samples defaults to 5 > 1
    let data = array![[1.0, 2.0]];
    let params = HdbscanParams {
        min_cluster_size: 5,
        ..Default::default()
    };
    let mut hdbscan = Hdbscan::new(params);
    assert!(hdbscan.fit(&data.view()).is_err());
}

#[test]
fn test_single_point_with_min_samples_1() {
    let data = array![[1.0, 2.0]];
    let params = HdbscanParams {
        min_cluster_size: 2,
        min_samples: Some(1),
        ..Default::default()
    };
    let mut hdbscan = Hdbscan::new(params);
    let labels = hdbscan.fit_predict(&data.view()).unwrap();
    assert_eq!(labels, vec![-1]);
}

#[test]
fn test_nan_values_error() {
    let data = array![[1.0, f64::NAN], [2.0, 3.0]];
    let mut hdbscan = Hdbscan::new(HdbscanParams::default());
    match hdbscan.fit(&data.view()) {
        Err(HdbscanError::InvalidData) => {}
        other => panic!("Expected InvalidData error, got {:?}", other),
    }
}

#[test]
fn test_inf_values_error() {
    let data = array![[1.0, f64::INFINITY], [2.0, 3.0]];
    let mut hdbscan = Hdbscan::new(HdbscanParams::default());
    match hdbscan.fit(&data.view()) {
        Err(HdbscanError::InvalidData) => {}
        other => panic!("Expected InvalidData error, got {:?}", other),
    }
}

#[test]
fn test_empty_data_error() {
    let data = Array2::<f64>::zeros((0, 3));
    let mut hdbscan = Hdbscan::new(HdbscanParams::default());
    match hdbscan.fit(&data.view()) {
        Err(HdbscanError::EmptyData) => {}
        other => panic!("Expected EmptyData error, got {:?}", other),
    }
}

#[test]
fn test_min_cluster_size_2() {
    let data = array![
        [0.0, 0.0],
        [0.01, 0.0],
        [0.0, 0.01],
        [10.0, 10.0],
        [10.01, 10.0],
        [10.0, 10.01],
    ];
    let params = HdbscanParams {
        min_cluster_size: 2,
        ..Default::default()
    };
    let mut hdbscan = Hdbscan::new(params);
    let labels = hdbscan.fit_predict(&data.view()).unwrap();
    let n_clusters = *labels.iter().max().unwrap() + 1;
    assert!(n_clusters >= 2, "Expected at least 2 clusters with min_cluster_size=2");
}

#[test]
fn test_invalid_min_cluster_size() {
    let params = HdbscanParams {
        min_cluster_size: 1,
        ..Default::default()
    };
    assert!(params.validate().is_err());
}

#[test]
fn test_invalid_min_cluster_size_zero() {
    let params = HdbscanParams {
        min_cluster_size: 0,
        ..Default::default()
    };
    assert!(params.validate().is_err());
}

#[test]
fn test_duplicates_within_clusters() {
    // Mix of unique and duplicate points
    let data = array![
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.1, 0.1],
        [0.1, 0.1],
        [10.0, 10.0],
        [10.0, 10.0],
        [10.0, 10.0],
        [10.1, 10.1],
        [10.1, 10.1],
    ];
    let params = HdbscanParams {
        min_cluster_size: 3,
        ..Default::default()
    };
    let mut hdbscan = Hdbscan::new(params);
    let result = hdbscan.fit_predict(&data.view());
    assert!(result.is_ok());
}

#[test]
fn test_precomputed_non_square_error() {
    let data = Array2::zeros((3, 4));
    let params = HdbscanParams {
        min_cluster_size: 2,
        metric: hdbscan_rs::Metric::Precomputed,
        ..Default::default()
    };
    let mut hdbscan = Hdbscan::new(params);
    match hdbscan.fit(&data.view()) {
        Err(HdbscanError::NonSquareDistanceMatrix { .. }) => {}
        other => panic!("Expected NonSquareDistanceMatrix error, got {:?}", other),
    }
}

#[test]
fn test_approximate_predict_not_fitted() {
    let hdbscan = Hdbscan::new(HdbscanParams::default());
    let points = array![[1.0, 2.0]];
    match hdbscan.approximate_predict(&points.view()) {
        Err(HdbscanError::NotFitted) => {}
        other => panic!("Expected NotFitted error, got {:?}", other),
    }
}

#[test]
fn test_approximate_predict_dimension_mismatch() {
    let data = array![
        [0.0, 0.0],
        [0.1, 0.0],
        [0.0, 0.1],
        [0.1, 0.1],
        [0.05, 0.05],
    ];
    let params = HdbscanParams {
        min_cluster_size: 2,
        ..Default::default()
    };
    let mut hdbscan = Hdbscan::new(params);
    hdbscan.fit(&data.view()).unwrap();

    let points = array![[1.0, 2.0, 3.0]]; // wrong dims
    match hdbscan.approximate_predict(&points.view()) {
        Err(HdbscanError::DimensionMismatch { .. }) => {}
        other => panic!("Expected DimensionMismatch error, got {:?}", other),
    }
}

/// Regression test inspired by tom-whitehead/hdbscan: small dataset where only
/// a root cluster exists + allow_single_cluster + epsilon should not panic.
#[test]
fn test_single_root_cluster_with_epsilon_no_panic() {
    let data = array![
        [1.0, 1.0],
        [1.1, 1.1],
        [1.2, 1.0],
        [10.0, 10.0], // outlier
    ];
    let params = HdbscanParams {
        min_cluster_size: 3,
        allow_single_cluster: true,
        cluster_selection_epsilon: 1.2,
        ..Default::default()
    };
    let mut hdbscan = Hdbscan::new(params);
    let labels = hdbscan.fit_predict(&data.view()).unwrap();
    assert_eq!(labels.len(), 4);
    // At least some points should be clustered
    let n_clusters: HashSet<_> = labels.iter().filter(|&&l| l >= 0).map(|&l| l).collect();
    assert!(n_clusters.len() >= 1, "Should find at least 1 cluster");
}

/// Epsilon merging: without epsilon we get multiple clusters, with large epsilon
/// they merge into fewer clusters.
#[test]
fn test_epsilon_merges_close_clusters() {
    // Two tight groups close together
    let data = array![
        [0.0, 0.0],
        [0.1, 0.0],
        [0.0, 0.1],
        [0.1, 0.1],
        [0.05, 0.05],
        [0.5, 0.0],
        [0.6, 0.0],
        [0.5, 0.1],
        [0.6, 0.1],
        [0.55, 0.05],
    ];

    // Without epsilon
    let params_no_eps = HdbscanParams {
        min_cluster_size: 3,
        min_samples: Some(2),
        cluster_selection_epsilon: 0.0,
        ..Default::default()
    };
    let mut hdb1 = Hdbscan::new(params_no_eps);
    let labels1 = hdb1.fit_predict(&data.view()).unwrap();
    let clusters1: HashSet<_> = labels1.iter().filter(|&&l| l >= 0).map(|&l| l).collect();

    // With large epsilon — should merge close clusters
    let params_eps = HdbscanParams {
        min_cluster_size: 3,
        min_samples: Some(2),
        cluster_selection_epsilon: 5.0,
        ..Default::default()
    };
    let mut hdb2 = Hdbscan::new(params_eps);
    let labels2 = hdb2.fit_predict(&data.view()).unwrap();
    let clusters2: HashSet<_> = labels2.iter().filter(|&&l| l >= 0).map(|&l| l).collect();

    // Large epsilon should result in same or fewer clusters
    assert!(
        clusters2.len() <= clusters1.len(),
        "Epsilon should merge clusters: without={} with={}",
        clusters1.len(),
        clusters2.len()
    );
}

/// Two points should be too few for default min_cluster_size=5 — all noise.
#[test]
fn test_two_points_all_noise() {
    let data = array![[0.0, 0.0], [1.0, 1.0]];
    let params = HdbscanParams {
        min_cluster_size: 5,
        min_samples: Some(2),
        ..Default::default()
    };
    let mut hdbscan = Hdbscan::new(params);
    let labels = hdbscan.fit_predict(&data.view()).unwrap();
    assert_eq!(labels, vec![-1, -1]);
}

/// Large number of dimensions should work correctly.
#[test]
fn test_high_dimensional_data() {
    let dim = 50;
    let n = 30;
    let mut data = Array2::zeros((n, dim));
    // Create two clusters in high-dimensional space
    for i in 0..15 {
        for d in 0..dim {
            data[[i, d]] = (i as f64 * 0.01) + (d as f64 * 0.001);
        }
    }
    for i in 15..30 {
        for d in 0..dim {
            data[[i, d]] = 100.0 + (i as f64 * 0.01) + (d as f64 * 0.001);
        }
    }
    let params = HdbscanParams {
        min_cluster_size: 5,
        ..Default::default()
    };
    let mut hdbscan = Hdbscan::new(params);
    let labels = hdbscan.fit_predict(&data.view()).unwrap();
    let n_clusters = labels.iter().filter(|&&l| l >= 0).collect::<HashSet<_>>().len();
    assert!(n_clusters >= 2, "Should find at least 2 clusters in high-dim data, got {}", n_clusters);
}
