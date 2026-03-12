use hdbscan_rs::{ClusterSelectionMethod, Hdbscan, HdbscanParams, Metric};
use ndarray::Array2;
use std::collections::HashMap;

/// Generate blobs dataset deterministically (no external dep needed).
fn make_blobs(n: usize, centers: &[[f64; 2]], seed: u64) -> Array2<f64> {
    let mut data = Array2::zeros((n, 2));
    let mut rng_state = seed;
    let per_center = n / centers.len();
    for (c_idx, center) in centers.iter().enumerate() {
        let start = c_idx * per_center;
        let end = if c_idx == centers.len() - 1 {
            n
        } else {
            start + per_center
        };
        for i in start..end {
            // Simple LCG pseudo-random
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u1 = (rng_state >> 33) as f64 / (1u64 << 31) as f64;
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u2 = (rng_state >> 33) as f64 / (1u64 << 31) as f64;
            // Box-Muller for Gaussian noise
            let z0 = (-2.0 * u1.max(1e-300).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u3 = (rng_state >> 33) as f64 / (1u64 << 31) as f64;
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u4 = (rng_state >> 33) as f64 / (1u64 << 31) as f64;
            let z1 = (-2.0 * u3.max(1e-300).ln()).sqrt() * (2.0 * std::f64::consts::PI * u4).cos();
            data[[i, 0]] = center[0] + z0 * 0.5;
            data[[i, 1]] = center[1] + z1 * 0.5;
        }
    }
    data
}

#[test]
fn test_labels_range_invariant() {
    // Labels should always be in [-1, n_clusters)
    let data = make_blobs(200, &[[0.0, 0.0], [10.0, 10.0], [20.0, 0.0]], 42);
    let params = HdbscanParams {
        min_cluster_size: 5,
        ..Default::default()
    };
    let mut hdbscan = Hdbscan::new(params);
    let labels = hdbscan.fit_predict(&data.view()).unwrap();

    let max_label = *labels.iter().max().unwrap();
    let min_label = *labels.iter().min().unwrap();
    assert!(min_label >= -1, "Labels should be >= -1, got {}", min_label);
    assert!(max_label >= 0, "Should have at least one cluster");

    // All labels should be -1 or in [0, max_label]
    for &l in &labels {
        assert!(l == -1 || (l >= 0 && l <= max_label));
    }
}

#[test]
fn test_probabilities_range_invariant() {
    let data = make_blobs(200, &[[0.0, 0.0], [10.0, 10.0]], 42);
    let params = HdbscanParams {
        min_cluster_size: 5,
        ..Default::default()
    };
    let mut hdbscan = Hdbscan::new(params);
    hdbscan.fit(&data.view()).unwrap();

    let probs = hdbscan.probabilities().unwrap();
    let labels = hdbscan.labels().unwrap();

    for i in 0..probs.len() {
        assert!(
            probs[i] >= 0.0,
            "Probability should be >= 0, got {}",
            probs[i]
        );
        assert!(
            probs[i] <= 1.0,
            "Probability should be <= 1, got {}",
            probs[i]
        );

        // Noise points should have probability 0
        if labels[i] == -1 {
            assert_eq!(probs[i], 0.0, "Noise point should have probability 0");
        }
    }
}

#[test]
fn test_outlier_scores_range_invariant() {
    let data = make_blobs(200, &[[0.0, 0.0], [10.0, 10.0]], 42);
    let params = HdbscanParams {
        min_cluster_size: 5,
        ..Default::default()
    };
    let mut hdbscan = Hdbscan::new(params);
    hdbscan.fit(&data.view()).unwrap();

    let scores = hdbscan.outlier_scores().unwrap();
    for &s in scores {
        assert!(s >= 0.0, "Outlier score should be >= 0, got {}", s);
        assert!(s <= 1.0, "Outlier score should be <= 1, got {}", s);
    }
}

#[test]
fn test_determinism_multiple_runs() {
    let data = make_blobs(300, &[[0.0, 0.0], [10.0, 10.0], [5.0, 15.0]], 42);

    for _ in 0..5 {
        let params = HdbscanParams {
            min_cluster_size: 5,
            ..Default::default()
        };
        let mut h1 = Hdbscan::new(params.clone());
        let l1 = h1.fit_predict(&data.view()).unwrap();

        let mut h2 = Hdbscan::new(params);
        let l2 = h2.fit_predict(&data.view()).unwrap();

        assert_eq!(l1, l2, "Results should be deterministic across runs");
    }
}

#[test]
fn test_eom_vs_leaf_both_valid() {
    let data = make_blobs(200, &[[0.0, 0.0], [10.0, 10.0]], 42);

    let mut h_eom = Hdbscan::new(HdbscanParams {
        min_cluster_size: 5,
        cluster_selection_method: ClusterSelectionMethod::Eom,
        ..Default::default()
    });
    let labels_eom = h_eom.fit_predict(&data.view()).unwrap();

    let mut h_leaf = Hdbscan::new(HdbscanParams {
        min_cluster_size: 5,
        cluster_selection_method: ClusterSelectionMethod::Leaf,
        ..Default::default()
    });
    let labels_leaf = h_leaf.fit_predict(&data.view()).unwrap();

    // Both should produce valid labels
    assert_eq!(labels_eom.len(), 200);
    assert_eq!(labels_leaf.len(), 200);

    // Both should find at least 1 cluster
    assert!(*labels_eom.iter().max().unwrap() >= 0);
    assert!(*labels_leaf.iter().max().unwrap() >= 0);
}

#[test]
fn test_min_cluster_size_2() {
    // Smallest valid min_cluster_size
    let data = make_blobs(100, &[[0.0, 0.0], [10.0, 10.0]], 42);
    let params = HdbscanParams {
        min_cluster_size: 2,
        ..Default::default()
    };
    let mut hdbscan = Hdbscan::new(params);
    let labels = hdbscan.fit_predict(&data.view()).unwrap();
    assert_eq!(labels.len(), 100);
    // With mcs=2, should find at least 2 clusters
    assert!(*labels.iter().max().unwrap() >= 1);
}

#[test]
fn test_high_min_cluster_size_all_noise() {
    // min_cluster_size > n should produce all noise
    let data = make_blobs(50, &[[0.0, 0.0], [10.0, 10.0]], 42);
    let params = HdbscanParams {
        min_cluster_size: 30,
        min_samples: Some(5),
        ..Default::default()
    };
    let mut hdbscan = Hdbscan::new(params);
    let labels = hdbscan.fit_predict(&data.view()).unwrap();
    // With such large mcs relative to cluster sizes, many/all points should be noise
    assert!(labels.iter().filter(|&&l| l == -1).count() > 0);
}

#[test]
fn test_varying_min_samples() {
    let data = make_blobs(200, &[[0.0, 0.0], [10.0, 10.0]], 42);

    for ms in [2, 5, 10, 20] {
        let params = HdbscanParams {
            min_cluster_size: 5,
            min_samples: Some(ms),
            ..Default::default()
        };
        let mut hdbscan = Hdbscan::new(params);
        let labels = hdbscan.fit_predict(&data.view()).unwrap();
        assert_eq!(labels.len(), 200);
        // All labels valid
        for &l in &labels {
            assert!(l >= -1);
        }
    }
}

#[test]
fn test_three_clusters_well_separated() {
    let data = make_blobs(300, &[[0.0, 0.0], [20.0, 20.0], [40.0, 0.0]], 42);
    let params = HdbscanParams {
        min_cluster_size: 10,
        ..Default::default()
    };
    let mut hdbscan = Hdbscan::new(params);
    let labels = hdbscan.fit_predict(&data.view()).unwrap();

    let n_clusters = *labels.iter().max().unwrap() + 1;
    assert_eq!(
        n_clusters, 3,
        "Should find exactly 3 clusters, got {}",
        n_clusters
    );
}

#[test]
fn test_close_clusters_may_merge() {
    // Two very close clusters - might merge into one
    let data = make_blobs(200, &[[0.0, 0.0], [1.0, 1.0]], 42);
    let params = HdbscanParams {
        min_cluster_size: 5,
        ..Default::default()
    };
    let mut hdbscan = Hdbscan::new(params);
    let labels = hdbscan.fit_predict(&data.view()).unwrap();
    assert_eq!(labels.len(), 200);
    // Should find some clusters (close clusters might split or merge)
    let n_clusters = *labels.iter().max().unwrap() + 1;
    assert!(
        n_clusters >= 1,
        "Should find at least 1 cluster, got {}",
        n_clusters
    );
}

#[test]
fn test_approximate_predict_consistency() {
    let data = make_blobs(200, &[[0.0, 0.0], [10.0, 10.0]], 42);
    let params = HdbscanParams {
        min_cluster_size: 5,
        ..Default::default()
    };
    let mut hdbscan = Hdbscan::new(params);
    hdbscan.fit(&data.view()).unwrap();

    // Predict on training data — should mostly match original labels
    let (pred_labels, pred_probs) = hdbscan.approximate_predict(&data.view()).unwrap();
    assert_eq!(pred_labels.len(), 200);
    assert_eq!(pred_probs.len(), 200);

    let orig_labels = hdbscan.labels().unwrap();
    let mut matches = 0;
    for i in 0..200 {
        if pred_labels[i] == orig_labels[i] {
            matches += 1;
        }
    }
    // At least 80% of predictions should match (approximate prediction isn't exact)
    let match_rate = matches as f64 / 200.0;
    assert!(
        match_rate > 0.80,
        "Approximate predict should match > 80% of training labels, got {:.1}%",
        match_rate * 100.0
    );
}

#[test]
fn test_condensed_tree_not_empty() {
    let data = make_blobs(100, &[[0.0, 0.0], [10.0, 10.0]], 42);
    let params = HdbscanParams {
        min_cluster_size: 5,
        ..Default::default()
    };
    let mut hdbscan = Hdbscan::new(params);
    hdbscan.fit(&data.view()).unwrap();

    let tree = hdbscan.condensed_tree().unwrap();
    assert!(!tree.is_empty(), "Condensed tree should not be empty");

    // All edges should have valid lambda
    for edge in tree {
        assert!(edge.lambda_val >= 0.0);
        assert!(edge.child_size >= 1);
    }
}

#[test]
fn test_manhattan_metric() {
    let data = make_blobs(100, &[[0.0, 0.0], [10.0, 10.0]], 42);
    let params = HdbscanParams {
        min_cluster_size: 5,
        metric: Metric::Manhattan,
        ..Default::default()
    };
    let mut hdbscan = Hdbscan::new(params);
    let labels = hdbscan.fit_predict(&data.view()).unwrap();
    assert_eq!(labels.len(), 100);
    let n_clusters = *labels.iter().max().unwrap() + 1;
    assert!(
        n_clusters >= 2,
        "Manhattan metric should find at least 2 clusters"
    );
}

#[test]
fn test_precomputed_symmetric() {
    // Build a small precomputed distance matrix
    let points = vec![
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
    let n = points.len();
    let mut dist_matrix = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let dx = points[i][0] - points[j][0];
            let dy = points[i][1] - points[j][1];
            let d: f64 = dx * dx + dy * dy;
            dist_matrix[[i, j]] = d.sqrt();
        }
    }

    let params = HdbscanParams {
        min_cluster_size: 3,
        metric: Metric::Precomputed,
        ..Default::default()
    };
    let mut hdbscan = Hdbscan::new(params);
    let labels = hdbscan.fit_predict(&dist_matrix.view()).unwrap();

    let n_clusters = *labels.iter().max().unwrap() + 1;
    assert_eq!(n_clusters, 2, "Precomputed should find 2 clusters");
}

#[test]
fn test_cluster_selection_epsilon() {
    let data = make_blobs(300, &[[0.0, 0.0], [3.0, 3.0], [20.0, 20.0]], 42);

    // Without epsilon: might find 3 clusters
    let params_no_eps = HdbscanParams {
        min_cluster_size: 10,
        cluster_selection_epsilon: 0.0,
        ..Default::default()
    };
    let mut h1 = Hdbscan::new(params_no_eps);
    let labels_no_eps = h1.fit_predict(&data.view()).unwrap();

    // With large epsilon: should merge close clusters
    let params_eps = HdbscanParams {
        min_cluster_size: 10,
        cluster_selection_epsilon: 5.0,
        ..Default::default()
    };
    let mut h2 = Hdbscan::new(params_eps);
    let labels_eps = h2.fit_predict(&data.view()).unwrap();

    let n_clusters_no_eps = *labels_no_eps.iter().max().unwrap() + 1;
    let n_clusters_eps = *labels_eps.iter().max().unwrap() + 1;

    // Epsilon should produce same or fewer clusters
    assert!(
        n_clusters_eps <= n_clusters_no_eps,
        "Epsilon should merge clusters: {} with eps vs {} without",
        n_clusters_eps,
        n_clusters_no_eps
    );
}

#[test]
fn test_large_n_500_correct_clusters() {
    let data = make_blobs(
        500,
        &[[0.0, 0.0], [15.0, 15.0], [30.0, 0.0], [-15.0, 15.0]],
        42,
    );
    let params = HdbscanParams {
        min_cluster_size: 10,
        ..Default::default()
    };
    let mut hdbscan = Hdbscan::new(params);
    let labels = hdbscan.fit_predict(&data.view()).unwrap();

    let n_clusters = *labels.iter().max().unwrap() + 1;
    assert_eq!(n_clusters, 4, "Should find 4 clusters, got {}", n_clusters);

    let noise_count = labels.iter().filter(|&&l| l == -1).count();
    assert!(
        noise_count < 50,
        "Should have few noise points, got {}",
        noise_count
    );
}

/// sklearn-contrib test: for every min_cluster_size from 2 to 100,
/// every found cluster must have at least min_cluster_size points.
#[test]
fn test_min_cluster_size_enforcement() {
    let data = make_blobs(200, &[[0.0, 0.0], [10.0, 10.0], [20.0, 0.0]], 42);

    for mcs in [2, 5, 10, 20, 50, 100] {
        let params = HdbscanParams {
            min_cluster_size: mcs,
            min_samples: Some(mcs.min(10)),
            ..Default::default()
        };
        let mut hdbscan = Hdbscan::new(params);
        let labels = hdbscan.fit_predict(&data.view()).unwrap();

        // Count points per cluster (ignoring noise)
        let mut cluster_counts: HashMap<i32, usize> = HashMap::new();
        for &l in &labels {
            if l >= 0 {
                *cluster_counts.entry(l).or_insert(0) += 1;
            }
        }

        for (&cluster_id, &count) in &cluster_counts {
            assert!(
                count >= mcs,
                "Cluster {} has {} points, less than min_cluster_size={}",
                cluster_id,
                count,
                mcs
            );
        }
    }
}

/// sklearn-contrib test: no clusters when min_cluster_size > n.
#[test]
fn test_no_clusters_when_mcs_exceeds_n() {
    let data = make_blobs(50, &[[0.0, 0.0], [10.0, 10.0]], 42);
    let params = HdbscanParams {
        min_cluster_size: 51, // larger than dataset
        min_samples: Some(5),
        ..Default::default()
    };
    let mut hdbscan = Hdbscan::new(params);
    let labels = hdbscan.fit_predict(&data.view()).unwrap();

    let n_clusters = labels.iter().filter(|&&l| l >= 0).count();
    assert_eq!(
        n_clusters, 0,
        "No clusters when min_cluster_size > n, got {} clustered points",
        n_clusters
    );
}

/// Labels should be consecutive integers starting from 0.
#[test]
fn test_labels_are_consecutive() {
    let data = make_blobs(300, &[[0.0, 0.0], [15.0, 15.0], [30.0, 0.0]], 42);
    let params = HdbscanParams {
        min_cluster_size: 10,
        ..Default::default()
    };
    let mut hdbscan = Hdbscan::new(params);
    let labels = hdbscan.fit_predict(&data.view()).unwrap();

    let max_label = *labels.iter().max().unwrap();
    if max_label >= 0 {
        // Every label from 0 to max_label should appear at least once
        for expected in 0..=max_label {
            assert!(
                labels.iter().any(|&l| l == expected),
                "Label {} missing — labels should be consecutive from 0 to {}",
                expected,
                max_label
            );
        }
    }
}
