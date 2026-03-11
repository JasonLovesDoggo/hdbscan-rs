use ndarray::ArrayView1;

use super::chebyshev::chebyshev_distance;
use super::euclidean::euclidean_distance;
use super::manhattan::manhattan_distance;

#[inline]
pub fn minkowski_distance(a: &ArrayView1<f64>, b: &ArrayView1<f64>, p: f64) -> f64 {
    if p == 1.0 {
        return manhattan_distance(a, b);
    }
    if p == 2.0 {
        return euclidean_distance(a, b);
    }
    if p.is_infinite() {
        return chebyshev_distance(a, b);
    }
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs().powf(p))
        .sum::<f64>()
        .powf(1.0 / p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_minkowski_p2_is_euclidean() {
        let a = arr1(&[0.0, 0.0]);
        let b = arr1(&[3.0, 4.0]);
        assert!((minkowski_distance(&a.view(), &b.view(), 2.0) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_minkowski_p1_is_manhattan() {
        let a = arr1(&[0.0, 0.0]);
        let b = arr1(&[3.0, 4.0]);
        assert!((minkowski_distance(&a.view(), &b.view(), 1.0) - 7.0).abs() < 1e-12);
    }

    #[test]
    fn test_minkowski_pinf_is_chebyshev() {
        let a = arr1(&[0.0, 0.0]);
        let b = arr1(&[3.0, 4.0]);
        assert!((minkowski_distance(&a.view(), &b.view(), f64::INFINITY) - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_minkowski_p3() {
        let a = arr1(&[0.0, 0.0]);
        let b = arr1(&[3.0, 4.0]);
        let expected = (3.0_f64.powi(3) + 4.0_f64.powi(3)).powf(1.0 / 3.0);
        assert!((minkowski_distance(&a.view(), &b.view(), 3.0) - expected).abs() < 1e-12);
    }
}
