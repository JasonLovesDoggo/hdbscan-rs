use hdbscan_rs::{ClusterSelectionMethod, Hdbscan, HdbscanParams, Metric};
use ndarray::Array2;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Deserialize)]
struct Fixture {
    name: String,
    params: FixtureParams,
    data: Vec<Vec<f64>>,
    n_points: usize,
    n_features: usize,
    expected_labels: Vec<i32>,
    expected_probabilities: Vec<f64>,
    n_clusters: usize,
    n_noise: usize,
    elapsed_ms: f64,
}

#[derive(Deserialize)]
struct FixtureParams {
    min_cluster_size: usize,
    #[serde(default)]
    min_samples: Option<usize>,
    #[serde(default = "default_metric")]
    metric: String,
    #[serde(default = "default_method")]
    cluster_selection_method: String,
    #[serde(default)]
    cluster_selection_epsilon: Option<f64>,
    #[serde(default)]
    allow_single_cluster: Option<bool>,
}

fn default_metric() -> String {
    "euclidean".to_string()
}
fn default_method() -> String {
    "eom".to_string()
}

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
}

fn load_fixture(name: &str) -> Fixture {
    let path = fixture_dir().join(format!("{}.json", name));
    let content = fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!("Failed to read fixture {}: {}", path.display(), e);
    });
    serde_json::from_str(&content).unwrap_or_else(|e| {
        panic!("Failed to parse fixture {}: {}", name, e);
    })
}

fn fixture_to_array(fixture: &Fixture) -> Array2<f64> {
    let mut arr = Array2::zeros((fixture.n_points, fixture.n_features));
    for (i, row) in fixture.data.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            arr[[i, j]] = val;
        }
    }
    arr
}

fn build_params(fp: &FixtureParams) -> HdbscanParams {
    let metric = match fp.metric.as_str() {
        "euclidean" => Metric::Euclidean,
        "manhattan" => Metric::Manhattan,
        "precomputed" => Metric::Precomputed,
        other => panic!("Unknown metric: {}", other),
    };
    let method = match fp.cluster_selection_method.as_str() {
        "eom" => ClusterSelectionMethod::Eom,
        "leaf" => ClusterSelectionMethod::Leaf,
        other => panic!("Unknown method: {}", other),
    };
    HdbscanParams {
        min_cluster_size: fp.min_cluster_size,
        min_samples: fp.min_samples,
        metric,
        cluster_selection_method: method,
        cluster_selection_epsilon: fp.cluster_selection_epsilon.unwrap_or(0.0),
        allow_single_cluster: fp.allow_single_cluster.unwrap_or(false),
        ..Default::default()
    }
}

/// Normalize labels so that they are assigned in order of first appearance.
/// This allows comparing labels from different implementations that may assign
/// different numeric IDs to the same clusters.
fn normalize_labels(labels: &[i32]) -> Vec<i32> {
    let mut mapping: HashMap<i32, i32> = HashMap::new();
    let mut next_id = 0i32;
    labels
        .iter()
        .map(|&l| {
            if l == -1 {
                -1
            } else {
                *mapping.entry(l).or_insert_with(|| {
                    let id = next_id;
                    next_id += 1;
                    id
                })
            }
        })
        .collect()
}

/// Compute the Adjusted Rand Index between two label vectors.
/// ARI = 1.0 means perfect agreement, 0.0 means random, negative means worse than random.
fn adjusted_rand_index(labels_a: &[i32], labels_b: &[i32]) -> f64 {
    let n = labels_a.len();
    assert_eq!(n, labels_b.len());
    if n == 0 {
        return 1.0;
    }

    // Build contingency table
    let mut a_map: HashMap<i32, usize> = HashMap::new();
    let mut b_map: HashMap<i32, usize> = HashMap::new();
    let mut pair_map: HashMap<(i32, i32), usize> = HashMap::new();

    for i in 0..n {
        *a_map.entry(labels_a[i]).or_insert(0) += 1;
        *b_map.entry(labels_b[i]).or_insert(0) += 1;
        *pair_map.entry((labels_a[i], labels_b[i])).or_insert(0) += 1;
    }

    let comb2 = |x: usize| -> f64 { (x as f64) * ((x as f64) - 1.0) / 2.0 };
    let n_c2 = comb2(n);
    if n_c2 == 0.0 {
        return 1.0;
    }

    let sum_nij_c2: f64 = pair_map.values().map(|&v| comb2(v)).sum();
    let sum_ai_c2: f64 = a_map.values().map(|&v| comb2(v)).sum();
    let sum_bj_c2: f64 = b_map.values().map(|&v| comb2(v)).sum();

    let expected = sum_ai_c2 * sum_bj_c2 / n_c2;
    let max_val = (sum_ai_c2 + sum_bj_c2) / 2.0;

    if max_val == expected {
        return 1.0;
    }

    (sum_nij_c2 - expected) / (max_val - expected)
}

struct CompareResult {
    fixture_name: String,
    n_points: usize,
    sklearn_clusters: usize,
    rust_clusters: usize,
    sklearn_noise: usize,
    rust_noise: usize,
    labels_match_exactly: bool,
    ari: f64,
    prob_rmse: f64,
    sklearn_ms: f64,
    rust_ms: f64,
    speedup: f64,
}

fn run_comparison(name: &str) -> CompareResult {
    let fixture = load_fixture(name);
    let data = fixture_to_array(&fixture);
    let params = build_params(&fixture.params);

    let mut hdbscan = Hdbscan::new(params);

    let start = Instant::now();
    let rust_labels = hdbscan.fit_predict(&data.view()).unwrap();
    let rust_ms = start.elapsed().as_secs_f64() * 1000.0;

    let rust_probs = hdbscan.probabilities().unwrap().to_vec();

    let norm_sklearn = normalize_labels(&fixture.expected_labels);
    let norm_rust = normalize_labels(&rust_labels);

    let labels_match = norm_sklearn == norm_rust;
    let ari = adjusted_rand_index(&fixture.expected_labels, &rust_labels);

    let rust_clusters = rust_labels.iter().filter(|&&l| l >= 0).map(|&l| l).max().map_or(0, |m| (m + 1) as usize);
    let rust_noise = rust_labels.iter().filter(|&&l| l == -1).count();

    // Probability RMSE (only for non-noise points that match cluster assignment)
    let mut prob_sse = 0.0;
    let mut prob_count = 0;
    for i in 0..fixture.n_points {
        if fixture.expected_labels[i] >= 0 && rust_labels[i] >= 0 {
            let diff = fixture.expected_probabilities[i] - rust_probs[i];
            prob_sse += diff * diff;
            prob_count += 1;
        }
    }
    let prob_rmse = if prob_count > 0 {
        (prob_sse / prob_count as f64).sqrt()
    } else {
        0.0
    };

    CompareResult {
        fixture_name: fixture.name,
        n_points: fixture.n_points,
        sklearn_clusters: fixture.n_clusters,
        rust_clusters,
        sklearn_noise: fixture.n_noise,
        rust_noise,
        labels_match_exactly: labels_match,
        ari,
        prob_rmse,
        sklearn_ms: fixture.elapsed_ms,
        rust_ms,
        speedup: fixture.elapsed_ms / rust_ms,
    }
}

// ===== Cross-implementation comparison tests =====

macro_rules! cross_impl_test {
    ($test_name:ident, $fixture:expr, $min_ari:expr) => {
        #[test]
        fn $test_name() {
            let result = run_comparison($fixture);
            println!(
                "[{}] sklearn: {} clusters/{} noise | rust: {} clusters/{} noise | exact={} ARI={:.4} prob_rmse={:.6} | sklearn {:.1}ms, rust {:.1}ms ({:.1}x)",
                result.fixture_name,
                result.sklearn_clusters, result.sklearn_noise,
                result.rust_clusters, result.rust_noise,
                result.labels_match_exactly, result.ari, result.prob_rmse,
                result.sklearn_ms, result.rust_ms, result.speedup,
            );
            assert!(
                result.ari >= $min_ari,
                "ARI {:.4} below minimum {:.4} for {}",
                result.ari, $min_ari, result.fixture_name,
            );
            // Probabilities should be in valid range
            assert!(result.prob_rmse.is_finite());
        }
    };
}

// --- Two moons ---
cross_impl_test!(cross_moons_mcs5_eom, "moons_n200_mcs5_eom", 0.95);
cross_impl_test!(cross_moons_mcs10_eom, "moons_n200_mcs10_eom", 0.95);
cross_impl_test!(cross_moons_mcs15_eom, "moons_n200_mcs15_eom", 0.95);
cross_impl_test!(cross_moons_mcs5_leaf, "moons_n200_mcs5_leaf", 0.85);
cross_impl_test!(cross_moons_mcs10_leaf, "moons_n200_mcs10_leaf", 0.90);
cross_impl_test!(cross_moons_mcs15_leaf, "moons_n200_mcs15_leaf", 0.90);
cross_impl_test!(cross_moons_mcs10_ms5_eom, "moons_n200_mcs10_ms5_eom", 0.95);

// --- Blobs ---
cross_impl_test!(cross_blobs_mcs5_eom, "blobs_n300_c4_mcs5_eom", 0.95);
cross_impl_test!(cross_blobs_mcs10_eom, "blobs_n300_c4_mcs10_eom", 0.95);
cross_impl_test!(cross_blobs_mcs20_eom, "blobs_n300_c4_mcs20_eom", 0.95);
cross_impl_test!(cross_blobs_mcs10_leaf, "blobs_n300_c4_mcs10_leaf", 0.95);
cross_impl_test!(cross_blobs_mcs10_eps1, "blobs_n300_c4_mcs10_eps1", 0.95);

// --- Circles ---
cross_impl_test!(cross_circles_mcs5_eom, "circles_n300_mcs5_eom", 0.95);
cross_impl_test!(cross_circles_mcs10_eom, "circles_n300_mcs10_eom", 0.95);
cross_impl_test!(cross_circles_mcs5_leaf, "circles_n300_mcs5_leaf", 0.90);
cross_impl_test!(cross_circles_mcs10_leaf, "circles_n300_mcs10_leaf", 0.90);

// --- Duplicates (petal-clustering #70) ---
cross_impl_test!(cross_duplicates, "duplicates_n80_mcs5_eom", 0.95);

// --- All identical ---
#[test]
fn cross_all_identical() {
    let result = run_comparison("all_identical_n30_mcs5");
    println!("[all_identical] rust: {} clusters, {} noise", result.rust_clusters, result.rust_noise);
    assert_eq!(result.rust_noise, result.n_points, "All identical points should be noise");
}

// --- Single cluster ---
cross_impl_test!(cross_single_no_allow, "single_blob_n100_mcs5_no_allow", 0.95);
cross_impl_test!(cross_single_allow, "single_blob_n100_mcs5_allow", 0.95);

// --- Varying density ---
cross_impl_test!(cross_varying_eom, "varying_density_n200_mcs10_eom", 0.95);
cross_impl_test!(cross_varying_leaf, "varying_density_n200_mcs10_leaf", 0.90);

// --- Manhattan ---
cross_impl_test!(cross_manhattan, "blobs_n150_c3_manhattan_mcs5", 0.95);

// --- Precomputed ---
cross_impl_test!(cross_precomputed, "precomputed_blobs_n100_c3_mcs5", 0.95);

// ===== Performance comparison tests =====

macro_rules! perf_test {
    ($test_name:ident, $fixture:expr) => {
        #[test]
        fn $test_name() {
            let result = run_comparison($fixture);
            println!(
                "[PERF {}] n={} | sklearn {:.1}ms | rust {:.1}ms | {:.2}x {}",
                result.fixture_name, result.n_points,
                result.sklearn_ms, result.rust_ms, result.speedup,
                if result.speedup >= 1.0 { "faster" } else { "slower" },
            );
            println!(
                "  clusters: sklearn={} rust={} | ARI={:.4}",
                result.sklearn_clusters, result.rust_clusters, result.ari,
            );
        }
    };
}

perf_test!(perf_n500, "perf_blobs_n500_c5_mcs10");
perf_test!(perf_n1000, "perf_blobs_n1000_c5_mcs10");
perf_test!(perf_n2000, "perf_blobs_n2000_c5_mcs10");
perf_test!(perf_n5000, "perf_blobs_n5000_c5_mcs10");
perf_test!(perf_n10000, "perf_blobs_n10000_c5_mcs10");

// ===== Summary test — runs everything and prints a table =====

#[test]
fn summary_report() {
    let fixtures = [
        "moons_n200_mcs5_eom",
        "moons_n200_mcs10_eom",
        "blobs_n300_c4_mcs5_eom",
        "blobs_n300_c4_mcs10_eom",
        "blobs_n300_c4_mcs10_leaf",
        "circles_n300_mcs5_eom",
        "circles_n300_mcs10_eom",
        "duplicates_n80_mcs5_eom",
        "varying_density_n200_mcs10_eom",
        "blobs_n150_c3_manhattan_mcs5",
        "precomputed_blobs_n100_c3_mcs5",
        "perf_blobs_n500_c5_mcs10",
        "perf_blobs_n1000_c5_mcs10",
        "perf_blobs_n2000_c5_mcs10",
        "perf_blobs_n5000_c5_mcs10",
        "perf_blobs_n10000_c5_mcs10",
    ];

    println!();
    println!("{:<45} {:>5} {:>4}/{:<4} {:>4}/{:<4} {:>5} {:>7} {:>9} {:>9} {:>7}",
        "fixture", "n", "sk_c", "rs_c", "sk_n", "rs_n", "exact", "ARI", "sk(ms)", "rs(ms)", "speedup");
    println!("{}", "-".repeat(120));

    let mut total_ari = 0.0;
    let mut count = 0;

    for name in &fixtures {
        let r = run_comparison(name);
        println!(
            "{:<45} {:>5} {:>4}/{:<4} {:>4}/{:<4} {:>5} {:>7.4} {:>9.1} {:>9.1} {:>6.1}x",
            r.fixture_name, r.n_points,
            r.sklearn_clusters, r.rust_clusters,
            r.sklearn_noise, r.rust_noise,
            if r.labels_match_exactly { "YES" } else { "no" },
            r.ari,
            r.sklearn_ms, r.rust_ms, r.speedup,
        );
        total_ari += r.ari;
        count += 1;
    }

    println!("{}", "-".repeat(120));
    println!("Average ARI: {:.4}", total_ari / count as f64);
    println!();
}
