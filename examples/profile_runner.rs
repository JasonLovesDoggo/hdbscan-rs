use hdbscan_rs::params::{HdbscanParams, Metric};
use ndarray::Array2;
use std::env;
use std::fs;
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: profile_runner <data.csv> <min_cluster_size>");
        std::process::exit(1);
    }

    let data_path = &args[1];
    let mcs: usize = args[2].parse().unwrap();

    // Read CSV
    let content = fs::read_to_string(data_path).unwrap();
    let rows: Vec<Vec<f64>> = content
        .lines()
        .filter(|l| !l.is_empty())
        .map(|line| {
            line.split(',')
                .map(|s| s.trim().parse::<f64>().unwrap())
                .collect()
        })
        .collect();

    let n = rows.len();
    let d = rows[0].len();
    let mut data = Array2::zeros((n, d));
    for (i, row) in rows.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            data[[i, j]] = val;
        }
    }

    let params = HdbscanParams {
        min_cluster_size: mcs,
        ..Default::default()
    };
    let min_samples = params.effective_min_samples();

    println!("n={}, d={}, mcs={}, min_samples={}", n, d, mcs, min_samples);

    // Step 1: Core distances
    let t0 = Instant::now();
    let core_distances = hdbscan_rs::core_distance::compute_core_distances(
        &data.view(),
        &Metric::Euclidean,
        min_samples,
    );
    let t1 = Instant::now();
    println!(
        "  core_distances: {:.2}ms",
        (t1 - t0).as_secs_f64() * 1000.0
    );

    // Step 2: MST (auto-selects Boruvka for large Euclidean datasets)
    let mst_edges = hdbscan_rs::mst::auto_mst(
        &data.view(),
        &core_distances.view(),
        &Metric::Euclidean,
        1.0,
    );
    let t2 = Instant::now();
    println!("  prim_mst: {:.2}ms", (t2 - t1).as_secs_f64() * 1000.0);

    // Step 3: Single linkage
    let single_linkage = hdbscan_rs::linkage::mst_to_single_linkage(&mst_edges, n);
    let t3 = Instant::now();
    println!(
        "  single_linkage: {:.2}ms",
        (t3 - t2).as_secs_f64() * 1000.0
    );

    // Step 4: Condensed tree
    let condensed = hdbscan_rs::condensed_tree::build_condensed_tree(&single_linkage, n, mcs);
    let t4 = Instant::now();
    println!(
        "  condensed_tree: {:.2}ms",
        (t4 - t3).as_secs_f64() * 1000.0
    );

    // Step 5: Cluster selection
    let selection = hdbscan_rs::cluster_selection::select_clusters(
        &condensed,
        n,
        hdbscan_rs::ClusterSelectionMethod::Eom,
        0.0,
        false,
    );
    let t5 = Instant::now();
    println!(
        "  cluster_selection: {:.2}ms",
        (t5 - t4).as_secs_f64() * 1000.0
    );

    // Step 6: Labels
    let labels =
        hdbscan_rs::labels::assign_labels(&condensed, &selection.selected_clusters, n, false, 0.0);
    let t6 = Instant::now();
    println!("  labels: {:.2}ms", (t6 - t5).as_secs_f64() * 1000.0);

    // Step 7: Probabilities
    let _probs = hdbscan_rs::membership::compute_probabilities(
        &condensed,
        &selection.selected_clusters,
        &labels,
        n,
    );
    let t7 = Instant::now();
    println!("  probabilities: {:.2}ms", (t7 - t6).as_secs_f64() * 1000.0);

    // Step 8: Outlier scores
    let _outlier = hdbscan_rs::outlier::compute_outlier_scores(&condensed, n);
    let t8 = Instant::now();
    println!(
        "  outlier_scores: {:.2}ms",
        (t8 - t7).as_secs_f64() * 1000.0
    );

    println!("  TOTAL: {:.2}ms", (t8 - t0).as_secs_f64() * 1000.0);
}
