/// Native benchmark matching the WASM benchmark configs for comparison.
fn main() {
    let configs: Vec<(&str, usize, usize, usize, usize)> = vec![
        ("5Kx10D", 5000, 10, 5, 10),
        ("5Kx50D", 5000, 50, 5, 10),
        ("50Kx2D", 50000, 2, 5, 10),
        ("1Kx256D", 1000, 256, 3, 5),
        ("10Kx10D", 10000, 10, 5, 10),
    ];

    println!("=== HDBSCAN-RS Native Benchmark ===\n");

    for (label, n, dim, n_centers, mcs) in configs {
        let data = hdbscan_rs::bench_utils::make_blobs(n, dim, n_centers, 42);

        // Warmup
        for _ in 0..2 {
            let mut h = hdbscan_rs::Hdbscan::new(hdbscan_rs::HdbscanParams {
                min_cluster_size: mcs,
                ..Default::default()
            });
            let _ = h.fit_predict(&data.view());
        }

        // Timed
        let mut times = Vec::new();
        let mut last_labels = vec![];
        for _ in 0..5 {
            let mut h = hdbscan_rs::Hdbscan::new(hdbscan_rs::HdbscanParams {
                min_cluster_size: mcs,
                ..Default::default()
            });
            let t0 = std::time::Instant::now();
            let labels = h.fit_predict(&data.view()).unwrap();
            times.push(t0.elapsed().as_secs_f64() * 1000.0);
            last_labels = labels;
        }

        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = times[times.len() / 2];
        let n_clusters = last_labels
            .iter()
            .filter(|&&l| l >= 0)
            .collect::<std::collections::HashSet<_>>()
            .len();
        let noise = last_labels.iter().filter(|&&l| l == -1).count();
        println!("  {label}: {median:.1}ms (median of 5), {n_clusters} clusters, {noise} noise");
    }
}
