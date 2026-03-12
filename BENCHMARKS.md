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
| 500x2D | 6.4 ms | 10.0 ms | **1.6 ms** | 4.1x | 6.3x | 1.00 |
| 1Kx2D | 9.9 ms | 13.3 ms | **1.9 ms** | 5.1x | 6.8x | 1.00 |
| 2Kx2D | 25.5 ms | 30.1 ms | **4.3 ms** | 5.9x | 6.9x | 1.00 |
| 5Kx2D | 125 ms | 82.3 ms | **12.0 ms** | 10.4x | 6.9x | 1.00 |
| 10Kx2D | 468 ms | 178 ms | **26.3 ms** | 17.8x | 6.8x | 1.00 |
| 50Kx2D | 12,906 ms | 1,029 ms | **167 ms** | 77.4x | 6.2x | 1.00 |

### Medium-dimensional

| Config | sklearn | C-hdbscan | hdbscan-rs | vs sklearn | vs C-hdbscan | ARI |
|--------|--------:|----------:|-----------:|-----------:|-------------:|----:|
| 5Kx10D | 245 ms | 136 ms | **99 ms** | 2.5x | 1.4x | 1.00 |
| 5Kx50D | 922 ms | 383 ms | **326 ms** | 2.8x | 1.2x | 1.00 |

### High-dimensional (LLM embeddings)

| Config | sklearn | C-hdbscan | hdbscan-rs | vs sklearn | vs C-hdbscan | ARI |
|--------|--------:|----------:|-----------:|-----------:|-------------:|----:|
| 2Kx256D | 921 ms | 854 ms | **78 ms** | 11.8x | 11.0x | 1.00 |
| 1Kx256D | 239 ms | 226 ms | **22 ms** | 11.0x | 10.4x | 1.00 |
| 500x1536D | 417 ms | 441 ms | **24 ms** | 17.3x | 18.4x | 1.00 |

### Peak memory (RSS)

| Config | sklearn | C-hdbscan | hdbscan-rs |
|--------|--------:|----------:|-----------:|
| 500x2D | 129 MB | 129 MB | **3 MB** |
| 10Kx2D | 136 MB | 137 MB | **6 MB** |
| 50Kx2D | 161 MB | 178 MB | **21 MB** |
| 5Kx50D | 178 MB | 178 MB | 207 MB |
| 2Kx256D | 178 MB | 178 MB | 60 MB |
| 500x1536D | 178 MB | 178 MB | 44 MB |

Python-based implementations carry ~128 MB baseline from the interpreter + NumPy + sklearn. Rust runs as a standalone binary with no runtime overhead.

Note: the 5Kx50D, 2Kx256D, and 500x1536D configs use a fused GEMM+Prim's approach that caches the Gram matrix (X@X.T) in memory. This trades memory for speed by computing all pairwise dot products via cache-blocked matrix multiply, then deriving distances as needed. The fused path is automatically selected for Euclidean metric with dim > 16.

## MST algorithm selection

The crate picks the MST strategy automatically based on the metric, dataset size, and dimensionality:

| Condition | Algorithm | Complexity |
|-----------|-----------|------------|
| Euclidean, dim <= 4, n > 500 | Dual-tree Boruvka (kd-tree) | O(n log^2 n) |
| Euclidean, dim 5-16, n > 4,000 | Dual-tree Boruvka (kd-tree) | O(n log^2 n) |
| Euclidean, dim > 16, n > threshold | Dual-tree Boruvka (ball tree) | O(n log^2 n) |
| Euclidean, dim > 16, small n | Fused GEMM+Prim's (cached Gram matrix) | O(n^2) |
| Small n or non-Euclidean | Prim's | O(n^2) |

Core distances use bounded kd-tree kNN for dim <= 10, ball tree kNN for dim 11-512, brute force SIMD for dim > 512, and precomputed/generic for non-Euclidean metrics.

## Key optimizations

- **SIMD auto-vectorization** -- 4-wide unrolled accumulators for distance computation (adaptive: simple loop for dim < 8, unrolled for dim >= 8)
- **Shared KnnHeap** -- deduplicated binary max-heap (O(log k) push) used by all tree-based kNN and fused core extraction
- **Ball tree kNN** -- sqrt-free pruning with cached `sqrt(max_dist)`, pre-computed child centroid distances passed to avoid redundant SIMD ops
- **Bounded KD-tree** with per-node bounding boxes for tight minimum-distance lower bounds, branchless AABB distance (compiles to maxpd SIMD)
- **Shared tree construction** -- single ball/kd-tree shared between core distances and MST when both use tree-based algorithms
- **kNN-seeded Boruvka** -- nearest neighbor indices from core distance computation seed initial component bounds
- **Fused GEMM+Prim's** -- for dim > 16: compute Gram matrix X@X.T via cache-blocked matmul, extract core distances via kNN heaps, run Prim's with O(1) distance lookups derived from cached dot products
- **Fully squared-distance Prim's** -- all comparisons in squared MR space, sqrt only at edge creation; precomputed core² for fast pruning
- **Unsafe-indexed distance** -- bounds-check-free flat array access in hot distance loops
- **Copy-derive structs** -- MstEdge, SingleLinkageMerge, CondensedTreeEdge are Copy for zero-overhead value semantics
- **Iterative condensed tree** -- stack-based fallout subtree emission avoids recursion overhead
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
