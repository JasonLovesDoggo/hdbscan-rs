# hdbscan-rs

High-performance HDBSCAN clustering for Python, powered by a Rust core. Drop-in compatible with scikit-learn's API, but significantly faster -- especially on small and large datasets.

## Installation

```sh
pip install hdbscan-rs
```

Requires Python >= 3.12 and NumPy >= 1.20. Pre-built wheels available for Linux, macOS, and Windows.

## Quick start

```python
import numpy as np
from hdbscan_rs import HDBSCAN

data = np.random.randn(10000, 2)

clusterer = HDBSCAN(min_cluster_size=15)
labels = clusterer.fit_predict(data)

print(f"Found {labels.max() + 1} clusters, {(labels == -1).sum()} noise points")
```

## API

```python
HDBSCAN(
    min_cluster_size=5,       # Smallest group that counts as a cluster
    min_samples=None,         # Controls density estimate (default: min_cluster_size)
    metric="euclidean",       # "euclidean", "manhattan", "cosine", "minkowski", "precomputed"
    p=None,                   # Minkowski p parameter
    alpha=1.0,                # Mutual reachability scaling factor
    cluster_selection_epsilon=0.0,  # Merge clusters below this distance
    cluster_selection_method="eom", # "eom" (Excess of Mass) or "leaf"
    allow_single_cluster=False,
)
```

### Methods

- **`fit_predict(X)`** -- Fit and return cluster labels (numpy array, -1 = noise)
- **`fit(X)`** -- Fit the model without returning labels
- **`approximate_predict(X)`** -- Predict labels for new points (returns labels, probabilities)

### Properties (after fitting)

- **`labels_`** -- Cluster labels (-1 = noise)
- **`probabilities_`** -- Membership strength [0, 1]
- **`outlier_scores_`** -- GLOSH outlier scores [0, 1]

## Performance

Single-thread, best-of-3 wall time on a 4-core AMD EPYC. Data is `make_blobs` with 5 centers, `min_cluster_size=10`.

| Config | sklearn HDBSCAN | hdbscan (C) | hdbscan-rs | vs sklearn | vs C |
|--------|----------------:|------------:|-----------:|-----------:|-----:|
| 1Kx2D | 9.6 ms | 13.2 ms | **5.3 ms** | 1.8x | 2.4x |
| 5Kx2D | 171 ms | 133 ms | **25 ms** | 6.8x | 5.3x |
| 10Kx2D | 469 ms | 179 ms | **52 ms** | 9.0x | 3.4x |
| 50Kx2D | 13,099 ms | 1,092 ms | **302 ms** | 43.4x | 3.6x |
| 5Kx10D | 264 ms | 144 ms | **146 ms** | 1.8x | ~1.0x |
| 1Kx256D | 241 ms | 232 ms | **98 ms** | 2.5x | 2.4x |
| 500x1536D | 421 ms | 448 ms | **175 ms** | 2.4x | 2.6x |

Memory usage is 5-60x lower than Python-based implementations.

## Migrating from sklearn

```python
# Before
from sklearn.cluster import HDBSCAN
clusterer = HDBSCAN(min_cluster_size=15)

# After
from hdbscan_rs import HDBSCAN
clusterer = HDBSCAN(min_cluster_size=15)
```

The API matches sklearn's interface. Input should be a 2D NumPy array of float64. Results are sklearn-compatible (ARI > 0.99 across the test suite).

## Precomputed distances

```python
from hdbscan_rs import HDBSCAN
import numpy as np

# Compute your own distance matrix
dist_matrix = np.array([[0, 1, 5], [1, 0, 3], [5, 3, 0]], dtype=np.float64)

clusterer = HDBSCAN(min_cluster_size=2, metric="precomputed")
labels = clusterer.fit_predict(dist_matrix)
```

## License

Licensed under either of Apache License, Version 2.0 or MIT License, at your option.
