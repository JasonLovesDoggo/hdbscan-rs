use ndarray::Array2;
use std::time::Instant;

fn make_gaussian_blobs(n: usize, dim: usize, n_clusters: usize, seed: u64) -> Array2<f64> {
    let mut data = Array2::zeros((n, dim));
    let mut rng = seed;
    let per_cluster = n / n_clusters;
    for c in 0..n_clusters {
        let start = c * per_cluster;
        let end = if c == n_clusters - 1 {
            n
        } else {
            start + per_cluster
        };
        let center = (c as f64) * 10.0;
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
                let z =
                    (-2.0 * u1.max(1e-300).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                data[[i, d]] = center + z * 2.0;
            }
        }
    }
    data
}

fn main() {
    for &(n, dim) in &[(5000usize, 10usize), (5000, 50)] {
        let data = make_gaussian_blobs(n, dim, 5, 42);
        let k = 10;

        let t0 = Instant::now();
        let tree = hdbscan_rs::ball_tree::BallTree::build(&data.view());
        let build_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t1 = Instant::now();
        for i in 0..n {
            let query = data.row(i);
            let _ = tree.query_knn(query.as_slice().unwrap(), k);
        }
        let knn_ms = t1.elapsed().as_secs_f64() * 1000.0;

        println!(
            "{}x{}D: build={:.1}ms, kNN={:.1}ms, total={:.1}ms, nodes={}",
            n,
            dim,
            build_ms,
            knn_ms,
            build_ms + knn_ms,
            tree.nodes.len()
        );
    }
}
