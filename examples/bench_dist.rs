use std::time::Instant;

fn main() {
    let dim = 10;
    let n = 5000;
    let mut data = vec![0.0f64; n * dim];
    let mut rng = 42u64;
    for v in data.iter_mut() {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        *v = (rng >> 33) as f64 / (1u64 << 31) as f64;
    }

    // Benchmark: compute n*(n-1)/2 pairwise distances
    let t = Instant::now();
    let mut sum = 0.0f64;
    for i in 0..n {
        for j in (i+1)..n {
            let d = hdbscan_rs::simd_distance::squared_euclidean_flat(&data, i, j, dim);
            sum += d;
        }
    }
    let elapsed = t.elapsed();
    println!("{}x{}D pairwise: {:.1}ms (sum={})", n, dim, elapsed.as_secs_f64() * 1000.0, sum);

    // Time per distance
    let n_pairs = (n * (n - 1)) / 2;
    println!("  {:.1}ns/dist, {} M pairs", elapsed.as_secs_f64() * 1e9 / n_pairs as f64, n_pairs / 1_000_000);
}
