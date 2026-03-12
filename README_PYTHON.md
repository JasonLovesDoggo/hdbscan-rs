# hdbscan-rs

High-performance HDBSCAN clustering for Python, powered by a Rust core. Drop-in compatible with scikit-learn's API, but significantly faster -- especially on small and large datasets.

## Installation

```sh
pip install hdbscan-rs
```

Requires Python >= 3.11 and NumPy >= 1.20. Pre-built wheels available for Linux, macOS, and Windows.

## Quick start

```python
import numpy as np
from hdbscan_rs import HDBSCAN

data = np.random.randn(10000, 2)

clusterer = HDBSCAN(min_cluster_size=15)
labels = clusterer.fit_predict(data)

print(f"Found {labels.max() + 1} clusters, {(labels == -1).sum()} noise points")
```

## Migrating from other HDBSCAN packages

### From `hdbscan` (standalone)

```diff
- import hdbscan
- clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=2, metric='euclidean')
+ from hdbscan_rs import HDBSCAN
+ clusterer = HDBSCAN(min_cluster_size=15, min_samples=2, metric='euclidean')
  labels = clusterer.fit_predict(data)
```

### From `sklearn.cluster.HDBSCAN`

```diff
- from sklearn.cluster import HDBSCAN
+ from hdbscan_rs import HDBSCAN
  clusterer = HDBSCAN(min_cluster_size=15)
  labels = clusterer.fit_predict(data)
```

### From BERTopic

```python
from hdbscan_rs import HDBSCAN
from bertopic import BERTopic

topic_model = BERTopic(hdbscan_model=HDBSCAN(min_cluster_size=15))
```

No other code changes needed. Labels, probabilities, and outlier scores are all compatible.

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

Best-of-3 wall time on a 4-core AMD EPYC. Data is `make_blobs` with 5 centers, `min_cluster_size=10`. Numbers are from the native Rust core; the Python binding adds <5ms overhead for data conversion.

| Config    |   sklearn | C-hdbscan | fast-hdbscan |  hdbscan-rs | vs sklearn | vs fast |
| --------- | --------: | --------: | -----------: | ----------: | ---------: | ------: |
| 1Kx2D     |   12.0 ms |   12.7 ms |       3.8 ms |  **2.2 ms** |       5.4x |    1.7x |
| 5Kx2D     |    121 ms |   76.0 ms |      20.6 ms |  **9.2 ms** |      13.1x |    2.2x |
| 10Kx2D    |    445 ms |    181 ms |      45.1 ms | **17.8 ms** |      25.0x |    2.5x |
| 50Kx2D    | 12,757 ms |  1,011 ms |       302 ms |  **101 ms** |       126x |    3.0x |
| 5Kx10D    |    240 ms |    133 ms |      70.5 ms |   **49 ms** |       4.9x |    1.4x |
| 1Kx256D   |    235 ms |    230 ms |      65.4 ms |   **19 ms** |      12.1x |    3.4x |
| 500x1536D |    412 ms |    439 ms |      80.7 ms |   **27 ms** |      15.1x |    3.0x |

Memory: 3-41 MB (Rust) vs 120-150 MB (sklearn) vs 121-169 MB (C-hdbscan) vs 457-470 MB (fast-hdbscan + Numba JIT).

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
