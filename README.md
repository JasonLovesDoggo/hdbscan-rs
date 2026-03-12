# hdbscan-rs

A Rust implementation of [HDBSCAN](https://hdbscan.readthedocs.io/en/latest/) - Hierarchical Density-Based Spatial Clustering of Applications with Noise. Produces results compatible with scikit-learn's HDBSCAN, but runs significantly faster on large datasets thanks to a dual-tree Boruvka MST and tight pruning in native code.

## Quick start

Add it to your project:

```sh
cargo add hdbscan-rs
```

Cluster some data:

```rust
use hdbscan_rs::{Hdbscan, HdbscanParams};
use ndarray::array;

let data = array![
    [0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1], [0.05, 0.05],
    [10.0, 10.0], [10.1, 10.0], [10.0, 10.1], [10.1, 10.1], [10.05, 10.05],
];

let params = HdbscanParams { min_cluster_size: 3, ..Default::default() };
let mut hdbscan = Hdbscan::new(params);
let labels = hdbscan.fit_predict(&data.view()).unwrap();
// labels: [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
```

## Features

- **sklearn-compatible output** -labels, probabilities, outlier scores, and condensed tree all match the reference Python implementation (ARI > 0.99 across fixture suite)
- **Fast** -dual-tree Boruvka MST with per-node component caching, lazy sqrt, and closer-child-first traversal. Falls back to Prim's for non-Euclidean metrics or small datasets
- **Approximate prediction** -classify new points against a fitted model without re-clustering
- **Cluster centers** -optional centroid and/or medoid computation
- **Five distance metrics** -Euclidean, Manhattan, Cosine, Minkowski(p), or bring your own precomputed distance matrix

## Performance

Measured on 2D blobs (5 clusters, `min_cluster_size=10`), single thread, best-of-N:

| n       | Time      |
|---------|-----------|
| 500     | 1.8 ms    |
| 1,000   | 5.0 ms    |
| 2,000   | 7.9 ms    |
| 5,000   | 24.1 ms   |
| 10,000  | 54.5 ms   |
| 20,000  | 114.4 ms  |
| 50,000  | 290.2 ms  |

The MST algorithm is selected automatically: dual-tree Boruvka for Euclidean data with n >= 128, Prim's otherwise. See [BENCHMARKS.md](BENCHMARKS.md) for methodology and detailed results.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_cluster_size` | 5 | Smallest group that counts as a cluster |
| `min_samples` | `None` (= min_cluster_size) | Controls density estimate; higher = more conservative |
| `metric` | Euclidean | Distance metric |
| `alpha` | 1.0 | Mutual reachability scaling factor |
| `cluster_selection_epsilon` | 0.0 | Merge clusters below this distance threshold |
| `cluster_selection_method` | Eom | `Eom` (Excess of Mass) or `Leaf` |
| `allow_single_cluster` | false | Permit the entire dataset to form one cluster |
| `store_centers` | `None` | Compute `Centroid`, `Medoid`, or `Both` |

## Richer output

After calling `fit` or `fit_predict`, you can access:

```rust
hdbscan.labels()         // Option<&[i32]>      -cluster labels (-1 = noise)
hdbscan.probabilities()  // Option<&[f64]>      -membership strength [0, 1]
hdbscan.outlier_scores() // Option<&[f64]>      -GLOSH outlier scores [0, 1]
hdbscan.condensed_tree() // Option<&[CondensedTreeEdge]>
hdbscan.centroids()      // Option<&Array2<f64>> -if store_centers was set
hdbscan.medoids()        // Option<&Array2<f64>> -if store_centers was set
```

## Precomputed distances

If you already have a distance matrix:

```rust
use hdbscan_rs::{Hdbscan, HdbscanParams, Metric};

let params = HdbscanParams {
    min_cluster_size: 3,
    metric: Metric::Precomputed,
    ..Default::default()
};
let mut hdbscan = Hdbscan::new(params);
let labels = hdbscan.fit_predict(&dist_matrix.view()).unwrap();
```

## Testing

The test suite validates against scikit-learn fixtures (blobs, moons, circles, varying density, duplicates, precomputed matrices) and includes property-based invariant tests.

```sh
cargo test
# 71 tests, plus 2 optional large-scale tests (100K and 1M points):
cargo test -- --ignored
```

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT License](LICENSE-MIT), at your option.
