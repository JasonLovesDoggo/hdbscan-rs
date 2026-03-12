//! Large-scale tests that are opt-in (behind feature flags or env vars).
//! Run with: HDBSCAN_LARGE_TESTS=1 cargo test large_scale -- --ignored
//!
//! These test correctness and performance at 100K and 1M points.

use hdbscan_rs::{Hdbscan, HdbscanParams};
use ndarray::Array2;
use std::time::Instant;

/// Simple LCG pseudo-random number generator.
fn lcg_next(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 33) as f64 / (1u64 << 31) as f64
}

/// Generate blobs dataset with given centers.
fn generate_blobs(n: usize, centers: &[[f64; 2]], std_dev: f64, seed: u64) -> Array2<f64> {
    let mut data = Array2::zeros((n, 2));
    let mut state = seed;
    let per_center = n / centers.len();

    for (c_idx, center) in centers.iter().enumerate() {
        let start = c_idx * per_center;
        let end = if c_idx == centers.len() - 1 {
            n
        } else {
            start + per_center
        };
        for i in start..end {
            // Box-Muller transform for Gaussian noise
            let u1 = lcg_next(&mut state).max(1e-300);
            let u2 = lcg_next(&mut state);
            let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            let u3 = lcg_next(&mut state).max(1e-300);
            let u4 = lcg_next(&mut state);
            let z1 = (-2.0 * u3.ln()).sqrt() * (2.0 * std::f64::consts::PI * u4).cos();
            data[[i, 0]] = center[0] + z0 * std_dev;
            data[[i, 1]] = center[1] + z1 * std_dev;
        }
    }
    data
}

#[test]
#[ignore] // Run with: cargo test large_scale_100k -- --ignored
fn large_scale_100k() {
    let n = 100_000;
    let centers = [
        [0.0, 0.0],
        [20.0, 0.0],
        [0.0, 20.0],
        [20.0, 20.0],
        [10.0, 10.0],
    ];
    let data = generate_blobs(n, &centers, 2.0, 42);

    let params = HdbscanParams {
        min_cluster_size: 50,
        ..Default::default()
    };
    let mut hdbscan = Hdbscan::new(params);

    let start = Instant::now();
    let labels = hdbscan.fit_predict(&data.view()).unwrap();
    let elapsed = start.elapsed();

    let n_clusters = *labels.iter().max().unwrap() + 1;
    let n_noise = labels.iter().filter(|&&l| l == -1).count();

    println!(
        "\n=== LARGE SCALE: n={} ===\n  clusters: {}\n  noise: {} ({:.1}%)\n  time: {:.1}ms\n",
        n,
        n_clusters,
        n_noise,
        n_noise as f64 / n as f64 * 100.0,
        elapsed.as_secs_f64() * 1000.0,
    );

    // Sanity checks
    assert_eq!(labels.len(), n);
    assert!(
        n_clusters >= 3,
        "Should find at least 3 clusters, got {}",
        n_clusters
    );
    assert!(
        n_clusters <= 10,
        "Should find at most 10 clusters, got {}",
        n_clusters
    );
    assert!(
        n_noise < n / 5,
        "Too many noise points: {} ({:.1}%)",
        n_noise,
        n_noise as f64 / n as f64 * 100.0,
    );

    // Performance check
    let ms = elapsed.as_secs_f64() * 1000.0;
    println!(
        "  Performance: {:.0}ms ({:.1} points/ms)",
        ms,
        n as f64 / ms
    );
    assert!(
        ms < 10_000.0,
        "n=100K should complete in under 10 seconds, took {:.0}ms",
        ms,
    );
}

#[test]
#[ignore] // Run with: cargo test large_scale_1m -- --ignored
fn large_scale_1m() {
    let n = 1_000_000;
    let centers = [
        [0.0, 0.0],
        [30.0, 0.0],
        [0.0, 30.0],
        [30.0, 30.0],
        [15.0, 15.0],
        [15.0, 0.0],
        [0.0, 15.0],
        [30.0, 15.0],
    ];
    let data = generate_blobs(n, &centers, 2.0, 42);

    let params = HdbscanParams {
        min_cluster_size: 100,
        ..Default::default()
    };
    let mut hdbscan = Hdbscan::new(params);

    let start = Instant::now();
    let labels = hdbscan.fit_predict(&data.view()).unwrap();
    let elapsed = start.elapsed();

    let n_clusters = *labels.iter().max().unwrap() + 1;
    let n_noise = labels.iter().filter(|&&l| l == -1).count();

    println!(
        "\n=== LARGE SCALE: n={} ===\n  clusters: {}\n  noise: {} ({:.1}%)\n  time: {:.1}ms\n",
        n,
        n_clusters,
        n_noise,
        n_noise as f64 / n as f64 * 100.0,
        elapsed.as_secs_f64() * 1000.0,
    );

    assert_eq!(labels.len(), n);
    assert!(
        n_clusters >= 4,
        "Should find at least 4 clusters, got {}",
        n_clusters
    );
    assert!(
        n_clusters <= 20,
        "Should find at most 20 clusters, got {}",
        n_clusters
    );

    let ms = elapsed.as_secs_f64() * 1000.0;
    println!(
        "  Performance: {:.0}ms ({:.1} points/ms)",
        ms,
        n as f64 / ms
    );
    assert!(
        ms < 120_000.0,
        "n=1M should complete in under 2 minutes, took {:.0}ms",
        ms,
    );
}
