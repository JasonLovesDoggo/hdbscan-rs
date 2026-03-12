//! SIMD-friendly squared Euclidean distance.
//!
//! For dimensions >= 8, uses manual 4-wide unrolling so LLVM can reliably
//! auto-vectorize the hot loop. For low dimensions (< 8), uses a simple loop
//! which LLVM can optimize better without unrolling overhead.

/// Dimension threshold below which the simple loop is faster.
const SIMD_THRESHOLD: usize = 8;

/// Simple squared Euclidean distance -- best for low dimensions.
#[inline(always)]
fn squared_euclidean_simple(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mut sum = 0.0f64;
    for i in 0..a.len() {
        let d = unsafe { *a.get_unchecked(i) - *b.get_unchecked(i) };
        sum += d * d;
    }
    sum
}

/// 4-wide unrolled squared Euclidean distance -- best for medium dimensions.
#[inline(always)]
fn squared_euclidean_unrolled(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mut sum0 = 0.0f64;
    let mut sum1 = 0.0f64;
    let mut sum2 = 0.0f64;
    let mut sum3 = 0.0f64;

    let mut i = 0;
    for _ in 0..chunks {
        let d0 = unsafe { *a.get_unchecked(i) - *b.get_unchecked(i) };
        let d1 = unsafe { *a.get_unchecked(i + 1) - *b.get_unchecked(i + 1) };
        let d2 = unsafe { *a.get_unchecked(i + 2) - *b.get_unchecked(i + 2) };
        let d3 = unsafe { *a.get_unchecked(i + 3) - *b.get_unchecked(i + 3) };
        sum0 += d0 * d0;
        sum1 += d1 * d1;
        sum2 += d2 * d2;
        sum3 += d3 * d3;
        i += 4;
    }

    for _ in 0..remainder {
        let d = unsafe { *a.get_unchecked(i) - *b.get_unchecked(i) };
        sum0 += d * d;
        i += 1;
    }

    (sum0 + sum1) + (sum2 + sum3)
}

/// Squared Euclidean distance between two slices of the same length.
///
/// Adaptively selects between a simple loop (low dimensions) and
/// 4-wide unrolled accumulation (high dimensions >= 8) for best performance.
#[inline(always)]
pub fn squared_euclidean_simd(a: &[f64], b: &[f64]) -> f64 {
    if a.len() < SIMD_THRESHOLD {
        squared_euclidean_simple(a, b)
    } else {
        squared_euclidean_unrolled(a, b)
    }
}

/// Squared Euclidean distance between two points in a flat data array.
#[inline(always)]
pub fn squared_euclidean_flat(data: &[f64], i: usize, j: usize, dim: usize) -> f64 {
    let off_i = i * dim;
    let off_j = j * dim;
    let a = unsafe { data.get_unchecked(off_i..off_i + dim) };
    let b = unsafe { data.get_unchecked(off_j..off_j + dim) };
    squared_euclidean_simd(a, b)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        let a = [0.0, 0.0, 0.0];
        let b = [3.0, 4.0, 0.0];
        assert!((squared_euclidean_simd(&a, &b) - 25.0).abs() < 1e-12);
    }

    #[test]
    fn test_high_dim() {
        let dim = 50;
        let a: Vec<f64> = (0..dim).map(|i| i as f64).collect();
        let b: Vec<f64> = (0..dim).map(|i| i as f64 + 1.0).collect();
        // Each dimension contributes 1.0^2 = 1.0, total = 50.0
        assert!((squared_euclidean_simd(&a, &b) - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_matches_naive() {
        let a = [1.5, 2.3, 4.7, 0.1, 9.9, 3.3, 7.7];
        let b = [0.5, 3.3, 2.7, 1.1, 8.9, 4.3, 6.7];
        let naive: f64 = a.iter().zip(&b).map(|(x, y)| (x - y) * (x - y)).sum();
        assert!((squared_euclidean_simd(&a, &b) - naive).abs() < 1e-12);
    }

    #[test]
    fn test_flat() {
        let data = [0.0, 0.0, 3.0, 4.0];
        assert!((squared_euclidean_flat(&data, 0, 1, 2) - 25.0).abs() < 1e-12);
    }

    #[test]
    fn test_threshold_boundary() {
        // Test at exactly the threshold (8 dims) -- should use unrolled path
        let a: Vec<f64> = (0..8).map(|i| i as f64).collect();
        let b: Vec<f64> = (0..8).map(|i| i as f64 + 1.0).collect();
        assert!((squared_euclidean_simd(&a, &b) - 8.0).abs() < 1e-12);
    }

    #[test]
    fn test_below_threshold() {
        // Test below threshold (7 dims) -- should use simple path
        let a: Vec<f64> = (0..7).map(|i| i as f64).collect();
        let b: Vec<f64> = (0..7).map(|i| i as f64 + 1.0).collect();
        assert!((squared_euclidean_simd(&a, &b) - 7.0).abs() < 1e-12);
    }

    #[test]
    fn test_256d_embedding() {
        let dim = 256;
        let a: Vec<f64> = (0..dim).map(|i| (i as f64 * 0.01).sin()).collect();
        let b: Vec<f64> = (0..dim).map(|i| (i as f64 * 0.01).cos()).collect();
        let naive: f64 = a.iter().zip(&b).map(|(x, y)| (x - y) * (x - y)).sum();
        assert!((squared_euclidean_simd(&a, &b) - naive).abs() < 1e-10);
    }

    #[test]
    fn test_1536d_embedding() {
        let dim = 1536;
        let a: Vec<f64> = (0..dim).map(|i| (i as f64 * 0.001).sin()).collect();
        let b: Vec<f64> = (0..dim).map(|i| (i as f64 * 0.001).cos()).collect();
        let naive: f64 = a.iter().zip(&b).map(|(x, y)| (x - y) * (x - y)).sum();
        assert!((squared_euclidean_simd(&a, &b) - naive).abs() < 1e-8);
    }
}
