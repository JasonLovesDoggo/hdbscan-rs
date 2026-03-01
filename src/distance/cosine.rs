use ndarray::ArrayView1;

#[inline]
pub fn cosine_distance(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        return 0.0;
    }
    // Clamp to handle floating point errors
    let sim = (dot / denom).clamp(-1.0, 1.0);
    1.0 - sim
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_cosine_identical() {
        let a = arr1(&[1.0, 2.0, 3.0]);
        assert!(cosine_distance(&a.view(), &a.view()).abs() < 1e-12);
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = arr1(&[1.0, 0.0]);
        let b = arr1(&[0.0, 1.0]);
        assert!((cosine_distance(&a.view(), &b.view()) - 1.0).abs() < 1e-12);
    }
}
