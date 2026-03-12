use std::time::Instant;

fn bench_config(n: usize, dim: usize, mcs: usize, runs: usize) {
    let data = hdbscan_rs::bench_utils::make_blobs(n, dim, 5, 42);
    // warmup
    {
        let mut h = hdbscan_rs::Hdbscan::new(hdbscan_rs::HdbscanParams {
            min_cluster_size: mcs,
            ..Default::default()
        });
        let _ = h.fit_predict(&data.view());
    }
    let mut times = Vec::new();
    for _ in 0..runs {
        let mut h = hdbscan_rs::Hdbscan::new(hdbscan_rs::HdbscanParams {
            min_cluster_size: mcs,
            ..Default::default()
        });
        let t = Instant::now();
        let labels = h.fit_predict(&data.view()).unwrap();
        times.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times[times.len() / 2];
    let best = times[0];
    println!(
        "  {}x{}D mcs={}: median={:.1}ms best={:.1}ms",
        n, dim, mcs, median, best
    );
}

fn main() {
    println!("Benchmark results (best of 7):");
    bench_config(1000, 2, 10, 7);
    bench_config(2000, 2, 10, 7);
    bench_config(5000, 2, 10, 7);
    bench_config(10000, 2, 10, 7);
    bench_config(50000, 2, 10, 5);
    bench_config(5000, 10, 10, 7);
    bench_config(5000, 50, 10, 5);
    bench_config(1000, 256, 10, 7);
    bench_config(500, 1536, 10, 7);
}
