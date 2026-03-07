#!/usr/bin/env python3
"""Generate test fixtures from scikit-learn HDBSCAN for cross-implementation validation."""

import json
import os
import time
import warnings

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.metrics import pairwise_distances

warnings.filterwarnings("ignore")

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
os.makedirs(FIXTURE_DIR, exist_ok=True)


def save_fixture(name, data, params, labels, probabilities, elapsed_ms):
    fixture = {
        "name": name,
        "params": params,
        "data": data.tolist(),
        "n_points": int(data.shape[0]),
        "n_features": int(data.shape[1]),
        "expected_labels": [int(l) for l in labels],
        "expected_probabilities": [round(float(p), 12) for p in probabilities],
        "n_clusters": int(max(labels) + 1) if max(labels) >= 0 else 0,
        "n_noise": int(sum(1 for l in labels if l == -1)),
        "sklearn_version": __import__("sklearn").__version__,
        "elapsed_ms": round(elapsed_ms, 2),
    }
    path = os.path.join(FIXTURE_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(fixture, f)
    print(f"  {name}: {fixture['n_clusters']} clusters, {fixture['n_noise']} noise, {elapsed_ms:.1f}ms")


def run_hdbscan(data, **kwargs):
    hdb = HDBSCAN(copy=True, **kwargs)
    t0 = time.perf_counter()
    hdb.fit(data)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return hdb.labels_, hdb.probabilities_, elapsed_ms


def gen_two_moons():
    print("two_moons:")
    np.random.seed(42)
    X, _ = make_moons(n_samples=200, noise=0.05, random_state=42)
    for mcs in [5, 10, 15]:
        for method in ["eom", "leaf"]:
            name = f"moons_n200_mcs{mcs}_{method}"
            params = {"min_cluster_size": mcs, "cluster_selection_method": method}
            labels, probs, elapsed = run_hdbscan(X, **params)
            save_fixture(name, X, params, labels, probs, elapsed)

    name = "moons_n200_mcs10_ms5_eom"
    params = {"min_cluster_size": 10, "min_samples": 5, "cluster_selection_method": "eom"}
    labels, probs, elapsed = run_hdbscan(X, **params)
    save_fixture(name, X, params, labels, probs, elapsed)


def gen_blobs():
    print("blobs:")
    np.random.seed(42)
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.5, random_state=42)
    for mcs in [5, 10, 20]:
        name = f"blobs_n300_c4_mcs{mcs}_eom"
        params = {"min_cluster_size": mcs, "cluster_selection_method": "eom"}
        labels, probs, elapsed = run_hdbscan(X, **params)
        save_fixture(name, X, params, labels, probs, elapsed)

    name = "blobs_n300_c4_mcs10_leaf"
    params = {"min_cluster_size": 10, "cluster_selection_method": "leaf"}
    labels, probs, elapsed = run_hdbscan(X, **params)
    save_fixture(name, X, params, labels, probs, elapsed)

    name = "blobs_n300_c4_mcs10_eps1"
    params = {"min_cluster_size": 10, "cluster_selection_epsilon": 1.0}
    labels, probs, elapsed = run_hdbscan(X, **params)
    save_fixture(name, X, params, labels, probs, elapsed)


def gen_circles():
    print("circles:")
    np.random.seed(42)
    X, _ = make_circles(n_samples=300, factor=0.5, noise=0.05, random_state=42)
    for mcs in [5, 10]:
        for method in ["eom", "leaf"]:
            name = f"circles_n300_mcs{mcs}_{method}"
            params = {"min_cluster_size": mcs, "cluster_selection_method": method}
            labels, probs, elapsed = run_hdbscan(X, **params)
            save_fixture(name, X, params, labels, probs, elapsed)


def gen_duplicates():
    print("duplicates:")
    cluster_a = np.array([[0.0, 0.0]] * 20 + [[0.1, 0.0]] * 10 + [[0.0, 0.1]] * 10)
    cluster_b = np.array([[5.0, 5.0]] * 20 + [[5.1, 5.0]] * 10 + [[5.0, 5.1]] * 10)
    X = np.vstack([cluster_a, cluster_b])
    name = "duplicates_n80_mcs5_eom"
    params = {"min_cluster_size": 5, "cluster_selection_method": "eom"}
    labels, probs, elapsed = run_hdbscan(X, **params)
    save_fixture(name, X, params, labels, probs, elapsed)


def gen_all_identical():
    print("all_identical:")
    X = np.ones((30, 2))
    name = "all_identical_n30_mcs5"
    params = {"min_cluster_size": 5}
    labels, probs, elapsed = run_hdbscan(X, **params)
    save_fixture(name, X, params, labels, probs, elapsed)


def gen_single_cluster():
    print("single_cluster:")
    np.random.seed(42)
    X, _ = make_blobs(n_samples=100, centers=1, cluster_std=0.5, random_state=42)
    name = "single_blob_n100_mcs5_no_allow"
    params = {"min_cluster_size": 5, "allow_single_cluster": False}
    labels, probs, elapsed = run_hdbscan(X, **params)
    save_fixture(name, X, params, labels, probs, elapsed)

    name = "single_blob_n100_mcs5_allow"
    params = {"min_cluster_size": 5, "allow_single_cluster": True}
    labels, probs, elapsed = run_hdbscan(X, **params)
    save_fixture(name, X, params, labels, probs, elapsed)


def gen_varying_density():
    print("varying_density:")
    np.random.seed(42)
    tight = np.random.randn(100, 2) * 0.1
    spread = np.random.randn(100, 2) * 2.0 + np.array([10, 10])
    X = np.vstack([tight, spread])
    for method in ["eom", "leaf"]:
        name = f"varying_density_n200_mcs10_{method}"
        params = {"min_cluster_size": 10, "cluster_selection_method": method}
        labels, probs, elapsed = run_hdbscan(X, **params)
        save_fixture(name, X, params, labels, probs, elapsed)


def gen_manhattan():
    print("manhattan:")
    np.random.seed(42)
    X, _ = make_blobs(n_samples=150, centers=3, cluster_std=0.5, random_state=42)
    name = "blobs_n150_c3_manhattan_mcs5"
    params = {"min_cluster_size": 5, "metric": "manhattan"}
    labels, probs, elapsed = run_hdbscan(X, **params)
    save_fixture(name, X, params, labels, probs, elapsed)


def gen_precomputed():
    print("precomputed:")
    np.random.seed(42)
    X, _ = make_blobs(n_samples=100, centers=3, cluster_std=0.5, random_state=42)
    D = pairwise_distances(X, metric="euclidean")
    name = "precomputed_blobs_n100_c3_mcs5"
    params = {"min_cluster_size": 5, "metric": "precomputed"}
    labels, probs, elapsed = run_hdbscan(D, **params)
    save_fixture(name, D, params, labels, probs, elapsed)


def gen_perf():
    print("performance:")
    for n in [500, 1000, 2000, 5000]:
        np.random.seed(42)
        X, _ = make_blobs(n_samples=n, centers=5, cluster_std=1.0, random_state=42)
        name = f"perf_blobs_n{n}_c5_mcs10"
        params = {"min_cluster_size": 10}
        labels, probs, elapsed = run_hdbscan(X, **params)
        save_fixture(name, X, params, labels, probs, elapsed)


if __name__ == "__main__":
    print("Generating HDBSCAN fixtures from scikit-learn " + __import__("sklearn").__version__)
    print("-" * 60)
    gen_two_moons()
    gen_blobs()
    gen_circles()
    gen_duplicates()
    gen_all_identical()
    gen_single_cluster()
    gen_varying_density()
    gen_manhattan()
    gen_precomputed()
    gen_perf()
    n = len([f for f in os.listdir(FIXTURE_DIR) if f.endswith(".json")])
    print(f"-" * 60)
    print(f"Done: {n} fixtures in {FIXTURE_DIR}")
