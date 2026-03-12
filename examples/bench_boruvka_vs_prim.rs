use std::time::Instant;
use ndarray::Array2;

fn bench_pipeline(n: usize, dim: usize) {
    let mut data = Array2::zeros((n, dim));
    for i in 0..n {
        let cluster = i / (n / 5);
        for d in 0..dim {
            let center = (cluster as f64) * 20.0;
            let seed = ((i * dim + d) as u64)
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1);
            let offset = ((seed >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 2.0;
            data[[i, d]] = center + offset;
        }
    }

    // Core distances: ball tree kNN (current path for dim > 8)
    let t = Instant::now();
    let cd = hdbscan_rs::core_distance::compute_core_distances(
        &data.view(),
        &hdbscan_rs::params::Metric::Euclidean,
        5,
    );
    let core_ms = t.elapsed().as_millis();

    // MST: kd-tree Boruvka
    let t = Instant::now();
    let kd_tree = hdbscan_rs::kdtree_bounded::BoundedKdTree::build(&data.view());
    let kd_edges = hdbscan_rs::mst::dual_tree_boruvka::dual_tree_boruvka_mst(
        &kd_tree, &cd.view(), 1.0, None,
    );
    let kd_mst_ms = t.elapsed().as_millis();

    // MST: Prim's
    let t = Instant::now();
    let prim_edges = hdbscan_rs::mst::prim::prim_mst(
        &data.view(), &cd.view(), &hdbscan_rs::params::Metric::Euclidean, 1.0,
    );
    let prim_mst_ms = t.elapsed().as_millis();

    // Rest of pipeline (linkage, condensed tree, etc) - use kd edges
    let t = Instant::now();
    let sl = hdbscan_rs::linkage::mst_to_single_linkage(&kd_edges, n);
    let condensed = hdbscan_rs::condensed_tree::build_condensed_tree(&sl, n, 10);
    let _selection = hdbscan_rs::cluster_selection::select_clusters(
        &condensed, n,
        hdbscan_rs::params::ClusterSelectionMethod::Eom,
        0.0, false,
    );
    let rest_ms = t.elapsed().as_millis();

    // Full end-to-end with auto_mst
    let t = Instant::now();
    let params = hdbscan_rs::HdbscanParams {
        min_cluster_size: 10,
        ..Default::default()
    };
    let mut h = hdbscan_rs::Hdbscan::new(params);
    let _ = h.fit_predict(&data.view());
    let total_ms = t.elapsed().as_millis();

    let _ = (prim_edges, kd_edges);

    eprintln!(
        "  {:>5}x{:>4}D: core={}ms  kd_mst={}ms  prim_mst={}ms  rest={}ms  total={}ms",
        n, dim, core_ms, kd_mst_ms, prim_mst_ms, rest_ms, total_ms
    );
}

fn main() {
    eprintln!("=== Pipeline Breakdown ===");
    bench_pipeline(5000, 10);
    bench_pipeline(5000, 50);
    bench_pipeline(2000, 256);
}
