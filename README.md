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

Single-thread, best-of-3 wall time on a 4-core AMD EPYC (GitHub Codespace). Compared against sklearn HDBSCAN (Cython) and C-hdbscan (Cython + ball tree Boruvka). Data is `make_blobs`, `min_cluster_size=10`.

| Config | sklearn | C-hdbscan | hdbscan-rs | vs sklearn | vs C-hdbscan |
|--------|--------:|----------:|-----------:|-----------:|-------------:|
| 1Kx2D | 9.1 ms | 13.9 ms | **4.2 ms** | 2.2x | 3.3x |
| 5Kx2D | 127 ms | 80.7 ms | **16.2 ms** | 7.8x | 5.0x |
| 10Kx2D | 462 ms | 179 ms | **34.6 ms** | 13.4x | 5.2x |
| 50Kx2D | 12,828 ms | 1,029 ms | **183 ms** | 70.0x | 5.6x |
| 5Kx10D | 242 ms | 154 ms | **121 ms** | 2.0x | 1.3x |
| 5Kx50D | 934 ms | 390 ms | 495 ms | 1.9x | 0.8x |
| 1Kx256D | 236 ms | 228 ms | **76 ms** | 3.1x | 3.0x |
| 500x1536D | 418 ms | 442 ms | **153 ms** | 2.7x | 2.9x |

Memory usage is **5-60x lower** than Python implementations (no interpreter/NumPy overhead).

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
