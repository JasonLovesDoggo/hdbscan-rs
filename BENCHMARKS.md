# Benchmarks

Head-to-head comparison against the two major HDBSCAN implementations:

- **sklearn** (`sklearn.cluster.HDBSCAN`) -- scikit-learn 1.6.1, Cython + NumPy
- **C-hdbscan** (`hdbscan` package) -- scikit-learn-contrib, Cython + ball tree Boruvka

All benchmarks are **single-threaded**, best-of-3 wall time. Data is `make_blobs` with 5 Gaussian centers, `min_cluster_size=10`, default parameters. Each run includes the full pipeline (core distances, MST, condensed tree, cluster selection, labels + probabilities + outlier scores).

## Machine

```
CPU:     AMD EPYC 7763 64-Core Processor (4 vCPUs)
RAM:     16 GB
L1d:     64 KiB (2 instances)
L2:      1 MiB (2 instances)
L3:      32 MiB (1 instance)
OS:      Ubuntu 24.04.3 LTS (Linux 6.8.0-1044-azure x86_64)
Rust:    rustc 1.94.0 (4a4ef493e 2026-03-02)
Python:  3.12.1
```

GitHub Codespace, Standard (4-core). Reproducible via `python3 tests/perf_comparison.py`.

## Results

### Low-dimensional (2D blobs)

| Config | sklearn | C-hdbscan | hdbscan-rs | vs sklearn | vs C-hdbscan | ARI |
|--------|--------:|----------:|-----------:|-----------:|-------------:|----:|
| 500x2D | 4.3 ms | 6.8 ms | **1.1 ms** | 3.9x | 6.2x | 1.00 |
| 1Kx2D | 9.6 ms | 13.2 ms | **3.5 ms** | 2.7x | 3.8x | 1.00 |
| 2Kx2D | 27.2 ms | 28.7 ms | **6.3 ms** | 4.3x | 4.6x | 1.00 |
| 5Kx2D | 171 ms | 133 ms | **22 ms** | 7.8x | 6.0x | 1.00 |
| 10Kx2D | 469 ms | 179 ms | **48 ms** | 9.8x | 3.7x | 1.00 |
| 50Kx2D | 13,099 ms | 1,092 ms | **239 ms** | 54.8x | 4.6x | 1.00 |

### Medium-dimensional

| Config | sklearn | C-hdbscan | hdbscan-rs | vs sklearn | vs C-hdbscan | ARI |
|--------|--------:|----------:|-----------:|-----------:|-------------:|----:|
| 5Kx10D | 264 ms | 144 ms | **134 ms** | 2.0x | 1.1x | 1.00 |
| 5Kx50D | 941 ms | 405 ms | **379 ms** | 2.5x | 1.1x | 1.00 |

### High-dimensional (LLM embeddings)

| Config | sklearn | C-hdbscan | hdbscan-rs | vs sklearn | vs C-hdbscan | ARI |
|--------|--------:|----------:|-----------:|-----------:|-------------:|----:|
| 2Kx256D | 943 ms | 863 ms | **245 ms** | 3.8x | 3.5x | 1.00 |
| 1Kx256D | 241 ms | 232 ms | **66 ms** | 3.7x | 3.5x | 1.00 |
| 500x1536D | 421 ms | 448 ms | **147 ms** | 2.9x | 3.0x | 1.00 |

### Peak memory (RSS)

| Config | sklearn | C-hdbscan | hdbscan-rs |
|--------|--------:|----------:|-----------:|
| 500x2D | 129 MB | 129 MB | **2 MB** |
| 10Kx2D | 138 MB | 138 MB | **6 MB** |
| 50Kx2D | 161 MB | 178 MB | **21 MB** |
| 5Kx50D | 178 MB | 178 MB | **16 MB** |
| 2Kx256D | 178 MB | 178 MB | **28 MB** |
| 500x1536D | 178 MB | 178 MB | **42 MB** |

Python-based implementations carry ~128 MB baseline from the interpreter + NumPy + sklearn. Rust runs as a standalone binary with no runtime overhead.

## MST algorithm selection

The crate picks the MST strategy automatically based on the metric, dataset size, and dimensionality:

| Condition | Algorithm | Complexity |
|-----------|-----------|------------|
| Euclidean, dim <= 16, n > 6,000 | Dual-tree Boruvka (kd-tree) | O(n log^2 n) |
| Euclidean, dim > 16, n > threshold | Dual-tree Boruvka (ball tree) | O(n log^2 n) |
| Small n or non-Euclidean | Prim's | O(n^2) |

Core distances use bounded kd-tree kNN for dim <= 10, ball tree kNN for dim 11-512, brute force SIMD for dim > 512, and precomputed/generic for non-Euclidean metrics.

## Key optimizations

- **SIMD auto-vectorization** -- 4-wide unrolled accumulators for distance computation (adaptive: simple loop for dim < 8, unrolled for dim >= 8)
- **Shared KnnHeap** -- deduplicated binary max-heap (O(log k) push) used by all tree-based kNN, with fast `query_core_dist` path that avoids sort/sqrt overhead
- **Ball tree kNN** -- sqrt-free pruning with cached `sqrt(max_dist)`, pre-computed child centroid distances passed to avoid redundant SIMD ops
- **Bounded KD-tree** with per-node bounding boxes for tight minimum-distance lower bounds
- **Shared tree construction** -- single ball/kd-tree shared between core distances and MST when both use tree-based algorithms
- **kNN-seeded Boruvka** -- nearest neighbor indices from core distance computation seed initial component bounds
- **Fully squared-distance Prim's** -- all comparisons in squared MR space, sqrt only at edge creation; precomputed core² for fast pruning
- **Cache-friendly active set** -- periodic sorting of active node indices for sequential memory access
- **Per-node component caching** -- bottom-up labeling for O(1) same-component subtree pruning
- **Closer-child-first traversal** -- tighter bounds earlier for better pruning on the second child
- **Core distance shortcut** -- skip points whose core distance exceeds their component's best edge
- **Vec-based post-processing** -- outlier scores, probabilities, and labels use indexed arrays instead of hash maps

## Reproducing

```sh
# Install Python dependencies
pip install scikit-learn hdbscan

# Run the full comparison
python3 tests/perf_comparison.py

# Quick Rust-only benchmark
cargo run --release --example bench
```
