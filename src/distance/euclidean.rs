use ndarray::ArrayView1;

#[inline]
pub fn euclidean_distance(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_euclidean() {
        let a = arr1(&[0.0, 0.0]);
        let b = arr1(&[3.0, 4.0]);
        assert!((euclidean_distance(&a.view(), &b.view()) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_same_point() {
        let a = arr1(&[1.0, 2.0, 3.0]);
        assert!(euclidean_distance(&a.view(), &a.view()).abs() < 1e-12);
    }
}
