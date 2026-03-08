use hdbscan_rs::{Hdbscan, HdbscanParams, StoreCenters};
use ndarray::array;

fn main() {
    let data = array![
        [0.0, 0.0],
        [0.1, 0.0],
        [0.0, 0.1],
        [0.1, 0.1],
        [0.05, 0.05],
        [10.0, 10.0],
        [10.1, 10.0],
        [10.0, 10.1],
        [10.1, 10.1],
        [10.05, 10.05],
        [5.0, 5.0], // outlier
    ];

    let params = HdbscanParams {
        min_cluster_size: 3,
        store_centers: Some(StoreCenters::Centroid),
        ..Default::default()
    };

    let mut hdbscan = Hdbscan::new(params);
    let labels = hdbscan.fit_predict(&data.view()).unwrap();

    println!("Labels: {:?}", labels);
    println!("Probabilities: {:?}", hdbscan.probabilities().unwrap());
    println!("Outlier scores: {:?}", hdbscan.outlier_scores().unwrap());

    if let Some(centroids) = hdbscan.centroids() {
        println!("Centroids:\n{:?}", centroids);
    }
}
