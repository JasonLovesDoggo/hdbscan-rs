use std::time::Instant;

fn main() {
    for &(n, dim) in &[(5000, 10), (5000, 50), (50000, 10)] {
        let data = hdbscan_rs::bench_utils::make_blobs(n, dim, 5, 42);
        let min_samples = 10usize;
        let k = min_samples.min(n);

        // Warmup
        {
            let mut h = hdbscan_rs::Hdbscan::new(hdbscan_rs::HdbscanParams {
                min_cluster_size: 10,
                ..Default::default()
            });
            let _ = h.fit_predict(&data.view());
        }

        // Phase 1: Core distances
        let t0 = Instant::now();
        let tree = hdbscan_rs::kdtree_bounded::BoundedKdTree::build(&data.view());
        let t_tree = t0.elapsed();

        let t1 = Instant::now();
        let (core_distances, nn_indices) =
            hdbscan_rs::core_distance::compute_core_distances_with_bounded_kdtree(
                &tree,
                &data.view(),
                min_samples,
            );
        let t_core = t1.elapsed();

        // Phase 2: MST
        let t2 = Instant::now();
        let mst_edges = hdbscan_rs::mst::dual_tree_boruvka::dual_tree_boruvka_mst(
            &tree,
            &core_distances.view(),
            1.0,
            Some(&nn_indices),
        );
        let t_mst = t2.elapsed();

        // Phase 3: Everything else
        let t3 = Instant::now();
        let single_linkage = hdbscan_rs::linkage::mst_to_single_linkage(&mst_edges, n);
        let condensed = hdbscan_rs::condensed_tree::build_condensed_tree(&single_linkage, n, 10);
        let selection = hdbscan_rs::cluster_selection::select_clusters(
            &condensed,
            n,
            hdbscan_rs::ClusterSelectionMethod::Eom,
            0.0,
            false,
        );
        let labels = hdbscan_rs::labels::assign_labels(
            &condensed,
            &selection.selected_clusters,
            n,
            false,
            0.0,
        );
        let probs = hdbscan_rs::membership::compute_probabilities(
            &condensed,
            &selection.selected_clusters,
            &labels,
            n,
        );
        let outliers = hdbscan_rs::outlier::compute_outlier_scores(&condensed, n);
        let t_rest = t3.elapsed();

        println!("{}x{}D breakdown:", n, dim);
        println!("  Tree build:  {:>7.1}ms", t_tree.as_secs_f64() * 1000.0);
        println!("  Core dists:  {:>7.1}ms", t_core.as_secs_f64() * 1000.0);
        println!("  Boruvka MST: {:>7.1}ms", t_mst.as_secs_f64() * 1000.0);
        println!("  Post-proc:   {:>7.1}ms", t_rest.as_secs_f64() * 1000.0);
        println!(
            "  Total:       {:>7.1}ms",
            (t_tree + t_core + t_mst + t_rest).as_secs_f64() * 1000.0
        );
        println!();
    }
}
