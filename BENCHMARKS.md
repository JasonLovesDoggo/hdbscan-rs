# Benchmarks

All benchmarks run on a single thread. Times are best-of-N to reduce noise (N=10 for small datasets, N=3 for large).

## Setup

Data is generated as 2D Gaussian blobs with 5 cluster centers, `min_cluster_size=10`, default parameters. The benchmark includes the full pipeline: core distance computation, MST construction, condensed tree building, cluster extraction, and label assignment.

```sh
cargo run --release --example bench
```

## Results

Measured in a GitHub Codespace (4-core AMD EPYC, 16 GB RAM):

| n       | Time      | MST algorithm         |
|---------|-----------|-----------------------|
| 500     | 1.8 ms    | Dual-tree Boruvka     |
| 1,000   | 5.0 ms    | Dual-tree Boruvka     |
| 2,000   | 7.9 ms    | Dual-tree Boruvka     |
| 5,000   | 24.1 ms   | Dual-tree Boruvka     |
| 10,000  | 54.5 ms   | Dual-tree Boruvka     |
| 20,000  | 114.4 ms  | Dual-tree Boruvka     |
| 50,000  | 290.2 ms  | Dual-tree Boruvka     |

## MST algorithm selection

The crate picks the MST strategy automatically based on the metric and dataset size:

| Condition                        | Algorithm              | Complexity      |
|----------------------------------|------------------------|-----------------|
| Euclidean, n >= 128              | Dual-tree Boruvka      | O(n log^2 n)    |
| Euclidean, n < 128               | Prim's                 | O(n^2)          |
| Non-Euclidean (any n)            | Prim's                 | O(n^2)          |

Prim's avoids the KD-tree construction overhead on small inputs and is the only option when bounding-box pruning doesn't apply (Manhattan, Cosine, Minkowski, Precomputed).

## What's being timed

The benchmark measures wall-clock time for `Hdbscan::fit_predict`, which includes:

1. Input validation
2. Core distance computation (k-nearest neighbor search via KD-tree)
3. Minimum spanning tree on the mutual reachability graph
4. Single-linkage dendrogram construction
5. Condensed tree extraction
6. Cluster selection (Excess of Mass)
7. Label assignment, probabilities, and outlier scores

## Key optimizations in the dual-tree Boruvka path

- **Bounded KD-tree** with per-node bounding boxes for tight minimum-distance lower bounds between node pairs
- **Per-node component caching** - bottom-up labeling lets us skip entire subtree pairs in O(1) when both belong to the same union-find component
- **Lazy sqrt** - when both core distances exceed the Euclidean distance, the mutual reachability distance equals `max(core_a, core_b)` and we skip the square root entirely
- **Closer-child-first traversal** - visiting the nearer child of the reference node first produces tighter bounds earlier, improving pruning on the second child
- **Core distance shortcut** - points whose core distance already exceeds their component's current best edge are skipped without computing any pairwise distances
- **Cached point components** - component IDs are snapshot once per Boruvka round into a flat array, avoiding repeated path-compressed union-find lookups in the inner loop

## Reproducing

```sh
# Quick benchmark
cargo run --release --example bench

# Criterion micro-benchmarks (if criterion is set up)
cargo bench
```
