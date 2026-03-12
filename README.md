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

Best-of-3 wall time on a 4-core AMD EPYC (GitHub Codespace). Compared against sklearn HDBSCAN (Cython) and C-hdbscan (Cython + ball tree Boruvka). Data is `make_blobs`, `min_cluster_size=10`.

| Config | sklearn | C-hdbscan | hdbscan-rs | vs sklearn | vs C-hdbscan |
|--------|--------:|----------:|-----------:|-----------:|-------------:|
| 1Kx2D | 8.9 ms | 12.7 ms | **2.6 ms** | 3.4x | 4.9x |
| 5Kx2D | 128 ms | 80.2 ms | **10.6 ms** | 12.1x | 7.6x |
| 10Kx2D | 455 ms | 189 ms | **18.4 ms** | 24.7x | 10.3x |
| 50Kx2D | 12,812 ms | 1,024 ms | **124 ms** | 103x | 8.2x |
| 5Kx10D | 241 ms | 136 ms | **62 ms** | 3.9x | 2.2x |
| 5Kx50D | 924 ms | 379 ms | **272 ms** | 3.4x | 1.4x |
| 1Kx256D | 246 ms | 230 ms | **19 ms** | 12.6x | 11.8x |
| 500x1536D | 424 ms | 444 ms | **28 ms** | 14.9x | 15.7x |

Memory usage is **3-60x lower** than Python implementations (no interpreter/NumPy overhead), except for medium-dim fused GEMM path which trades memory for speed.

See [BENCHMARKS.md](BENCHMARKS.md) for full results, machine specs, methodology, and analysis.

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
