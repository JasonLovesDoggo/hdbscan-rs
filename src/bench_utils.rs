//! Shared utilities for examples and benchmarks.

use ndarray::Array2;

/// Generate clustered Gaussian blob data for benchmarking.
///
/// Creates `n` points in `dim` dimensions, split evenly across `n_centers` clusters.
/// Each cluster is centered at `(center_idx * 20.0, ...)` with std dev 0.5.
/// Uses a deterministic LCG-based Box-Muller RNG seeded by `seed`.
pub fn make_blobs(n: usize, dim: usize, n_centers: usize, seed: u64) -> Array2<f64> {
    let mut data = Array2::zeros((n, dim));
    let mut rng = seed;
    let per_center = n / n_centers;
    for c in 0..n_centers {
        let start = c * per_center;
        let end = if c == n_centers - 1 { n } else { start + per_center };
        for i in start..end {
            for d in 0..dim {
                rng = rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u1 = (rng >> 33) as f64 / (1u64 << 31) as f64;
                rng = rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u2 = (rng >> 33) as f64 / (1u64 << 31) as f64;
                let z = (-2.0 * u1.max(1e-300).ln()).sqrt()
                    * (2.0 * std::f64::consts::PI * u2).cos();
                data[[i, d]] = (c as f64) * 20.0 + z * 0.5;
            }
        }
    }
    data
}
