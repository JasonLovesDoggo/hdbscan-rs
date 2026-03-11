use ndarray::ArrayView1;

/// Chebyshev (L∞) distance: the maximum absolute difference across dimensions.
#[inline]
pub fn chebyshev_distance(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_chebyshev() {
        let a = arr1(&[0.0, 0.0]);
        let b = arr1(&[3.0, 4.0]);
        assert!((chebyshev_distance(&a.view(), &b.view()) - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_chebyshev_same_point() {
        let a = arr1(&[1.0, 2.0, 3.0]);
        assert!(chebyshev_distance(&a.view(), &a.view()).abs() < 1e-12);
    }

    #[test]
    fn test_chebyshev_negative() {
        let a = arr1(&[-1.0, 5.0]);
        let b = arr1(&[2.0, -3.0]);
        assert!((chebyshev_distance(&a.view(), &b.view()) - 8.0).abs() < 1e-12);
    }
}
