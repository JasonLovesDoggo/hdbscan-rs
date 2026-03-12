use ndarray::ArrayView1;

#[inline]
pub fn manhattan_distance(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).abs()).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_manhattan() {
        let a = arr1(&[0.0, 0.0]);
        let b = arr1(&[3.0, 4.0]);
        assert!((manhattan_distance(&a.view(), &b.view()) - 7.0).abs() < 1e-12);
    }
}
