use std::time::Instant;

fn main() {
    for &(n, dim) in &[(5000, 10), (50000, 10)] {
        let data = hdbscan_rs::bench_utils::make_blobs(n, dim, 5, 42);
        let tree = hdbscan_rs::kdtree_bounded::BoundedKdTree::build(&data.view());
        let (core_distances, nn_indices) =
            hdbscan_rs::core_distance::compute_core_distances_with_bounded_kdtree(&tree, &data.view(), 10);

        // Time just Boruvka multiple times
        let mut times = Vec::new();
        for _ in 0..5 {
            let t = Instant::now();
            let _edges = hdbscan_rs::mst::dual_tree_boruvka::dual_tree_boruvka_mst(
                &tree, &core_distances.view(), 1.0, Some(&nn_indices),
            );
            times.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("{}x{}D Boruvka: best={:.1}ms median={:.1}ms", n, dim, times[0], times[2]);
    }
}
