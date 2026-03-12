use std::time::Instant;

fn main() {
    let data = hdbscan_rs::bench_utils::make_blobs(5000, 10, 5, 42);

    // warmup
    {
        let tree = hdbscan_rs::kdtree_bounded::BoundedKdTree::build(&data.view());
        let (cd, nn) = hdbscan_rs::core_distance::compute_core_distances_with_bounded_kdtree(&tree, &data.view(), 10);
        let _ = hdbscan_rs::mst::dual_tree_boruvka::dual_tree_boruvka_mst(&tree, &cd.view(), 1.0, Some(&nn));
    }

    // Measure
    let tree = hdbscan_rs::kdtree_bounded::BoundedKdTree::build(&data.view());
    let (cd, nn) = hdbscan_rs::core_distance::compute_core_distances_with_bounded_kdtree(&tree, &data.view(), 10);

    // Run Boruvka 3 times
    for run in 0..3 {
        let t = Instant::now();
        let edges = hdbscan_rs::mst::dual_tree_boruvka::dual_tree_boruvka_mst(&tree, &cd.view(), 1.0, Some(&nn));
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        eprintln!("Run {}: Boruvka {:.1}ms ({} edges)", run, ms, edges.len());
    }
}
