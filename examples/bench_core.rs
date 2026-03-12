use std::time::Instant;

fn main() {
    for n in &[5000usize, 10000] {
        let data = hdbscan_rs::bench_utils::make_blobs(*n, 10, 5, 42);
        let k = 10;

        // Warmup
        {
            let tree = hdbscan_rs::kdtree_bounded::BoundedKdTree::build(&data.view());
            let _ = hdbscan_rs::core_distance::compute_core_distances_with_bounded_kdtree(&tree, &data.view(), k);
        }

        // kd-tree approach
        let t = Instant::now();
        let tree = hdbscan_rs::kdtree_bounded::BoundedKdTree::build(&data.view());
        let (cd_kd, nn_kd) = hdbscan_rs::core_distance::compute_core_distances_with_bounded_kdtree(&tree, &data.view(), k);
        let kd_ms = t.elapsed().as_secs_f64() * 1000.0;

        // Brute-force upper triangle
        let t = Instant::now();
        let (cd_bf, nn_bf) = hdbscan_rs::core_distance::compute_core_distances_brute_upper_triangle(&data.view(), k);
        let bf_ms = t.elapsed().as_secs_f64() * 1000.0;

        // Verify they match
        let max_diff = cd_kd.iter().zip(cd_bf.iter()).map(|(a,b)| (a-b).abs()).fold(0.0f64, f64::max);

        eprintln!("{}x10D: kd-tree={:.1}ms brute={:.1}ms diff={:.2e}", n, kd_ms, bf_ms, max_diff);
    }
}
