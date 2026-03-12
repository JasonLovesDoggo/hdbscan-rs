use hdbscan_rs::{Hdbscan, HdbscanParams};
use ndarray::Array2;
use std::time::Instant;

fn make_gaussian_blobs(n: usize, dim: usize, n_clusters: usize, seed: u64) -> Array2<f64> {
    let mut data = Array2::zeros((n, dim));
    let mut rng = seed;
    let per_cluster = n / n_clusters;
    for c in 0..n_clusters {
        let start = c * per_cluster;
        let end = if c == n_clusters - 1 { n } else { start + per_cluster };
        let center = (c as f64) * 10.0;
        for i in start..end {
            for d in 0..dim {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let u1 = (rng >> 33) as f64 / (1u64 << 31) as f64;
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let u2 = (rng >> 33) as f64 / (1u64 << 31) as f64;
                let z = (-2.0 * u1.max(1e-300).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                data[[i, d]] = center + z * 2.0;
            }
        }
    }
    data
}

fn bench_config(n: usize, dim: usize, mcs: usize, iters: usize) -> f64 {
    let data = make_gaussian_blobs(n, dim, 5, 42);
    let params = HdbscanParams { min_cluster_size: mcs, ..Default::default() };
    // Warmup
    let mut h = Hdbscan::new(params.clone());
    let _ = h.fit_predict(&data.view());
    let mut best = f64::INFINITY;
    for _ in 0..iters {
        let mut h = Hdbscan::new(params.clone());
        let t0 = Instant::now();
        let _ = h.fit_predict(&data.view());
        let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
        if elapsed < best { best = elapsed; }
    }
    best
}

fn main() {
    let configs = vec![
        (500, 2, 5), (1000, 2, 5), (2000, 2, 5), (5000, 2, 5), (10000, 2, 5), (50000, 2, 5),
        (5000, 10, 10), (5000, 50, 10),
        (2000, 256, 10), (1000, 256, 10), (500, 1536, 10),
    ];
    println!("=== Full Pipeline Benchmark (gaussian blobs) ===");
    for (n, dim, mcs) in configs {
        let iters = if n <= 2000 { 8 } else if n <= 10000 { 5 } else { 3 };
        let ms = bench_config(n, dim, mcs, iters);
        println!("  {:>5}x{:<4}D (mcs={}): {:.1}ms", n, dim, mcs, ms);
    }
}
