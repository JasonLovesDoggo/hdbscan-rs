use hdbscan_rs::{Hdbscan, HdbscanParams, Metric};
use ndarray::array;

fn main() {
    // Precomputed distance matrix for 6 points in two groups
    // Group A: points 0,1,2 (close to each other)
    // Group B: points 3,4,5 (close to each other)
    let dist_matrix = array![
        [0.0, 0.1, 0.2, 5.0, 5.1, 5.2],
        [0.1, 0.0, 0.1, 5.1, 5.0, 5.1],
        [0.2, 0.1, 0.0, 5.2, 5.1, 5.0],
        [5.0, 5.1, 5.2, 0.0, 0.1, 0.2],
        [5.1, 5.0, 5.1, 0.1, 0.0, 0.1],
        [5.2, 5.1, 5.0, 0.2, 0.1, 0.0],
    ];

    let params = HdbscanParams {
        min_cluster_size: 2,
        metric: Metric::Precomputed,
        ..Default::default()
    };

    let mut hdbscan = Hdbscan::new(params);
    let labels = hdbscan.fit_predict(&dist_matrix.view()).unwrap();

    println!("Labels: {:?}", labels);
    println!("Probabilities: {:?}", hdbscan.probabilities().unwrap());
}
