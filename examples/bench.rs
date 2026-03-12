use hdbscan_rs::{Hdbscan, HdbscanParams};
use ndarray::Array2;
use std::time::Instant;

fn make_blobs(n: usize, centers: &[[f64; 2]], seed: u64) -> Array2<f64> {
    let mut data = Array2::zeros((n, 2));
    let mut rng = seed;
    let per_center = n / centers.len();
    for (c_idx, center) in centers.iter().enumerate() {
        let start = c_idx * per_center;
        let end = if c_idx == centers.len() - 1 { n } else { start + per_center };
        for i in start..end {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u1 = (rng >> 33) as f64 / (1u64 << 31) as f64;
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u2 = (rng >> 33) as f64 / (1u64 << 31) as f64;
            let z0 = (-2.0 * u1.max(1e-300).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u3 = (rng >> 33) as f64 / (1u64 << 31) as f64;
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u4 = (rng >> 33) as f64 / (1u64 << 31) as f64;
            let z1 = (-2.0 * u3.max(1e-300).ln()).sqrt() * (2.0 * std::f64::consts::PI * u4).cos();
            data[[i, 0]] = center[0] + z0 * 0.5;
            data[[i, 1]] = center[1] + z1 * 0.5;
        }
    }
    data
}

fn bench(n: usize, iters: usize) -> f64 {
    let centers = &[[0.0, 0.0], [10.0, 10.0], [20.0, 0.0], [-10.0, 5.0], [15.0, -5.0]];
    let data = make_blobs(n, centers, 42);
    let params = HdbscanParams {
        min_cluster_size: 10,
        ..Default::default()
    };

    // Warmup
    let mut h = Hdbscan::new(params.clone());
    let _ = h.fit_predict(&data.view());

    let mut best = f64::INFINITY;
    for _ in 0..iters {
        let mut h = Hdbscan::new(params.clone());
        let t0 = Instant::now();
        let _ = h.fit_predict(&data.view());
        let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
        if elapsed < best {
            best = elapsed;
        }
    }
    best
}

fn main() {
    println!("=== HDBSCAN Benchmark ===");
    for &n in &[500, 1000, 2000, 5000, 10000, 20000, 50000] {
        let iters = if n <= 2000 { 10 } else if n <= 10000 { 5 } else { 3 };
        let ms = bench(n, iters);
        println!("  n={:>6}: {:.2}ms", n, ms);
    }
}
