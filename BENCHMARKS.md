# Benchmarks

Head-to-head comparison against the three major HDBSCAN implementations:

- **sklearn** (`sklearn.cluster.HDBSCAN`) -- scikit-learn 1.6.1, Cython + NumPy
- **C-hdbscan** (`hdbscan` package) -- scikit-learn-contrib, Cython + ball tree Boruvka
- **fast-hdbscan** (`fast_hdbscan`) -- Numba JIT-compiled, optimized for speed

All benchmarks are best-of-3 wall time. Data is `make_blobs` with 5 Gaussian centers, `min_cluster_size=10`, default parameters. Each run includes the full pipeline (core distances, MST, condensed tree, cluster selection, labels + probabilities + outlier scores).

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

| Config |   sklearn | C-hdbscan | fast-hdbscan |  hdbscan-rs | vs sklearn |  vs C | vs fast |  ARI |
| ------ | --------: | --------: | -----------: | ----------: | ---------: | ----: | ------: | ---: |
| 500x2D |    4.1 ms |   10.9 ms |       2.0 ms |  **1.2 ms** |       3.3x |  8.9x |    1.6x | 1.00 |
| 1Kx2D  |   12.0 ms |   12.7 ms |       3.8 ms |  **2.2 ms** |       5.4x |  5.8x |    1.7x | 1.00 |
| 2Kx2D  |   32.1 ms |   27.3 ms |       6.6 ms |  **5.6 ms** |       5.7x |  4.9x |    1.2x | 1.00 |
| 5Kx2D  |    121 ms |   76.0 ms |      20.6 ms |  **9.2 ms** |      13.1x |  8.2x |    2.2x | 1.00 |
| 10Kx2D |    445 ms |    181 ms |      45.1 ms | **17.8 ms** |      25.0x | 10.1x |    2.5x | 1.00 |
| 50Kx2D | 12,757 ms |  1,011 ms |       302 ms |  **101 ms** |       126x | 10.0x |    3.0x | 1.00 |

### Medium-dimensional

| Config | sklearn | C-hdbscan | fast-hdbscan | hdbscan-rs | vs sklearn | vs C | vs fast |  ARI |
| ------ | ------: | --------: | -----------: | ---------: | ---------: | ---: | ------: | ---: |
| 5Kx10D |  240 ms |    133 ms |      70.5 ms |  **49 ms** |       4.9x | 2.7x |    1.4x | 1.00 |
| 5Kx50D |  911 ms |    369 ms |       324 ms | **268 ms** |       3.4x | 1.4x |    1.2x | 1.00 |

### High-dimensional (LLM embeddings)

| Config    | sklearn | C-hdbscan | fast-hdbscan | hdbscan-rs | vs sklearn |  vs C | vs fast |  ARI |
| --------- | ------: | --------: | -----------: | ---------: | ---------: | ----: | ------: | ---: |
| 2Kx256D   |  914 ms |    846 ms |       185 ms |  **74 ms** |      12.4x | 11.5x |    2.5x | 1.00 |
| 1Kx256D   |  235 ms |    230 ms |      65.4 ms |  **19 ms** |      12.1x | 11.9x |    3.4x | 1.00 |
| 500x1536D |  412 ms |    439 ms |      80.7 ms |  **27 ms** |      15.1x | 16.0x |    3.0x | 1.00 |

### Peak memory (RSS)

Each implementation runs in its own subprocess for independent measurement.

| Config    | sklearn | C-hdbscan | fast-hdbscan | hdbscan-rs |
| --------- | ------: | --------: | -----------: | ---------: |
| 500x2D    |  120 MB |    121 MB |       457 MB |   **3 MB** |
| 10Kx2D    |  126 MB |    129 MB |       459 MB |   **8 MB** |
| 50Kx2D    |  150 MB |    169 MB |       470 MB |  **29 MB** |
| 5Kx50D    |  125 MB |    128 MB |       460 MB |     207 MB |
| 2Kx256D   |  126 MB |    126 MB |       463 MB |  **56 MB** |
| 500x1536D |  130 MB |    130 MB |       466 MB |  **41 MB** |

sklearn and C-hdbscan carry ~120 MB baseline from the Python interpreter + NumPy + sklearn. fast-hdbscan adds ~340 MB from Numba JIT compilation. Rust runs as a standalone binary with no runtime overhead.

Note: the 5Kx50D, 2Kx256D, and 500x1536D configs use a fused GEMM+Prim's approach that caches the Gram matrix (X@X.T) in memory. This trades memory for speed by computing all pairwise dot products via cache-blocked matrix multiply, then deriving distances as needed. The fused path is automatically selected for Euclidean metric with dim > 16.

### WebAssembly (Node.js v8)

WASM build: `--profile wasm-release` (`opt-level = "s"`, `panic = "abort"`, `strip`, LTO) + `wasm-opt -Os`. Binary size: **462 KB**.

Single-threaded (WASM has no `std::thread`), same machine.

| Config  |   Native |     WASM | Slowdown |
| ------- | -------: | -------: | -------: |
| 5Kx10D  |  92.8 ms | 207.6 ms |     2.2x |
| 5Kx50D  | 260.6 ms | 572.6 ms |     2.2x |
| 50Kx2D  | 122.2 ms | 212.3 ms |     1.7x |
| 1Kx256D |  19.1 ms |  78.0 ms |     4.1x |
| 10Kx10D | 192.7 ms | 765.1 ms |     4.0x |

The 1.7–2.2x slowdown is from missing SIMD auto-vectorization. The 4x configs (10Kx10D, 1Kx256D) lose the multi-threaded kNN/Boruvka parallelism that native gets.

Reproducible via `tests/bench_wasm_vs_native.sh`.

## MST algorithm selection

The crate picks the MST strategy automatically based on the metric, dataset size, and dimensionality:

| Condition                          | Algorithm                              | Complexity   |
| ---------------------------------- | -------------------------------------- | ------------ |
| Euclidean, dim <= 4, n > 500       | Dual-tree Boruvka (kd-tree)            | O(n log^2 n) |
| Euclidean, dim 5-16, n > 4,000     | Dual-tree Boruvka (kd-tree)            | O(n log^2 n) |
| Euclidean, dim > 16, n > threshold | Dual-tree Boruvka (ball tree)          | O(n log^2 n) |
| Euclidean, dim > 16, small n       | Fused GEMM+Prim's (cached Gram matrix) | O(n^2)       |
| Small n or non-Euclidean           | Prim's                                 | O(n^2)       |

Core distances use bounded kd-tree kNN for dim <= 10, ball tree kNN for dim 11-512, brute force SIMD for dim > 512, and precomputed/generic for non-Euclidean metrics.

## Key optimizations

- **SIMD auto-vectorization** -- 4-wide unrolled accumulators for distance computation (adaptive: simple loop for dim < 8, unrolled for dim >= 8)
- **Fat LTO + target-cpu=native** -- cross-crate optimization of ndarray/matrixmultiply calls, AVX2 SIMD for distance computation
- **Shared KnnHeap** -- deduplicated binary max-heap (O(log k) push) used by all tree-based kNN and fused core extraction
- **Ball tree kNN** -- sqrt-free pruning with cached `sqrt(max_dist)`, pre-computed child centroid distances passed to avoid redundant SIMD ops
- **SoA tree layout** -- bounding boxes (kd-tree) and centroids (ball tree) stored in contiguous Structure-of-Arrays format, eliminating per-node Vec allocations and improving cache locality in hot distance loops
- **Bounded KD-tree** with per-node bounding boxes for tight minimum-distance lower bounds, branchless AABB distance (compiles to maxpd SIMD)
- **Reusable KnnHeap** -- single heap allocation reused across all n kNN queries via `clear()`, avoiding n separate alloc/dealloc cycles
- **Shared tree construction** -- single ball/kd-tree shared between core distances and MST when both use tree-based algorithms
- **kNN pre-merging** -- all k-1 nearest neighbor edges from core distance computation are used to merge components before the first Boruvka round, reducing initial component count by 30-60%
- **Adaptive parallel Boruvka** -- query subtree count scales with component density (many small → more subtrees, few large → fewer)
- **Cross-round edge preservation** -- valid cross-component edges from previous Boruvka round are preserved instead of reset, giving tighter initial bounds for dual-tree pruning
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
- **Parallel kNN core distances** -- kNN queries split across CPU cores via `std::thread::scope`, each thread with its own KnnHeap writing to disjoint output slices
- **Parallel dual-tree Boruvka** -- query tree split into subtrees processed in parallel, per-thread component_best arrays merged after each round

## Reproducing

```sh
# Run the full comparison (uv auto-installs dependencies)
uv run tests/perf_comparison.py

# Quick Rust-only benchmark
cargo run --release --example bench
```
