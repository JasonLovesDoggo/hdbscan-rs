use hdbscan_rs::{ClusterSelectionMethod, Hdbscan, HdbscanParams, Metric};
use ndarray::{array, Array2};

fn make_two_clusters() -> Array2<f64> {
    array![
        // Cluster A (around origin)
        [0.0, 0.0],
        [0.1, 0.0],
        [0.0, 0.1],
        [0.1, 0.1],
        [0.05, 0.05],
        [0.02, 0.08],
        [0.08, 0.02],
        // Cluster B (around (10,10))
        [10.0, 10.0],
        [10.1, 10.0],
        [10.0, 10.1],
        [10.1, 10.1],
        [10.05, 10.05],
        [10.02, 10.08],
        [10.08, 10.02],
    ]
}

#[test]
fn test_two_clusters_eom() {
    let data = make_two_clusters();
    let params = HdbscanParams {
        min_cluster_size: 3,
        cluster_selection_method: ClusterSelectionMethod::Eom,
        ..Default::default()
    };
    let mut hdbscan = Hdbscan::new(params);
    let labels = hdbscan.fit_predict(&data.view()).unwrap();

    // Should find 2 clusters
    let n_clusters = *labels.iter().max().unwrap() + 1;
    assert_eq!(n_clusters, 2);

    // First 7 points in one cluster, last 7 in another
    let label_a = labels[0];
    let label_b = labels[7];
    assert_ne!(label_a, label_b);
    for i in 0..7 {
        assert_eq!(labels[i], label_a, "point {} should be in cluster A", i);
    }
    for i in 7..14 {
        assert_eq!(labels[i], label_b, "point {} should be in cluster B", i);
    }
}

#[test]
fn test_two_clusters_leaf() {
    let data = make_two_clusters();
    let params = HdbscanParams {
        min_cluster_size: 3,
        cluster_selection_method: ClusterSelectionMethod::Leaf,
        ..Default::default()
    };
    let mut hdbscan = Hdbscan::new(params);
    let labels = hdbscan.fit_predict(&data.view()).unwrap();

    // With leaf selection, should still find at least 2 clusters
    let n_clusters = *labels.iter().max().unwrap() + 1;
    assert!(n_clusters >= 2);
}

#[test]
fn test_two_clusters_precomputed() {
    let raw_data = make_two_clusters();
    let n = raw_data.nrows();

    // Compute pairwise euclidean distances
    let mut dist_matrix = Array2::zeros((n, n));
    for i in 0..n {
        for j in (i + 1)..n {
            let d: f64 = raw_data
                .row(i)
                .iter()
                .zip(raw_data.row(j).iter())
                .map(|(&a, &b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            dist_matrix[[i, j]] = d;
            dist_matrix[[j, i]] = d;
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
    assert_eq!(n_clusters, 2);
}

#[test]
fn test_probabilities_in_range() {
    let data = make_two_clusters();
    let params = HdbscanParams {
        min_cluster_size: 3,
        ..Default::default()
    };
    let mut hdbscan = Hdbscan::new(params);
    hdbscan.fit(&data.view()).unwrap();

    let probs = hdbscan.probabilities().unwrap();
    for &p in probs {
        assert!(p >= 0.0 && p <= 1.0, "probability {} out of range", p);
    }

    let scores = hdbscan.outlier_scores().unwrap();
    for &s in scores {
        assert!(s >= 0.0 && s <= 1.0, "outlier score {} out of range", s);
    }
}

#[test]
fn test_labels_valid_range() {
    let data = make_two_clusters();
    let params = HdbscanParams {
        min_cluster_size: 3,
        ..Default::default()
    };
    let mut hdbscan = Hdbscan::new(params);
    let labels = hdbscan.fit_predict(&data.view()).unwrap();

    let n_clusters = (*labels.iter().max().unwrap() + 1).max(0);
    for &l in &labels {
        assert!(l >= -1 && l < n_clusters, "label {} out of valid range", l);
    }
}
