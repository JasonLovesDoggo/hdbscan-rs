"""Example: using hdbscan-rs from Python.

Install:
    pip install maturin numpy
    maturin develop --release

Or from PyPI (once published):
    pip install hdbscan-rs
"""

import numpy as np
from hdbscan_rs import HDBSCAN

# Generate some clustered data
rng = np.random.RandomState(42)
cluster_1 = rng.randn(50, 2) * 0.5 + [0, 0]
cluster_2 = rng.randn(50, 2) * 0.5 + [5, 5]
cluster_3 = rng.randn(50, 2) * 0.5 + [10, 0]
noise = rng.uniform(-2, 12, size=(10, 2))
data = np.vstack([cluster_1, cluster_2, cluster_3, noise])

# Cluster
hdb = HDBSCAN(min_cluster_size=10)
labels = hdb.fit_predict(data)

n_clusters = len(set(l for l in labels if l >= 0))
n_noise = sum(1 for l in labels if l == -1)
print(f"Found {n_clusters} clusters, {n_noise} noise points")

# Access probabilities and outlier scores
print(f"Mean probability: {hdb.probabilities_.mean():.3f}")
print(f"Mean outlier score: {hdb.outlier_scores_.mean():.3f}")

# Predict new points
new_points = np.array([[0.0, 0.0], [5.0, 5.0], [50.0, 50.0]])
pred_labels, pred_probs = hdb.approximate_predict(new_points)
for i, (pt, lbl, prob) in enumerate(zip(new_points, pred_labels, pred_probs)):
    print(f"  Point {pt} -> cluster {lbl} (confidence {prob:.3f})")
