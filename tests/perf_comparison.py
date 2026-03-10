#!/usr/bin/env python3
"""Run side-by-side performance comparison: sklearn vs hdbscan (C) vs hdbscan-rs (Rust).

Usage: python3 tests/perf_comparison.py
"""

import json
import os
import subprocess
import sys
import time
import warnings

import numpy as np
from sklearn.cluster import HDBSCAN as SklearnHDBSCAN
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

warnings.filterwarnings("ignore")

# Try to import the standalone C-based hdbscan library
try:
    import hdbscan as hdbscan_c
    HAS_C_HDBSCAN = True
except ImportError:
    HAS_C_HDBSCAN = False
    print("WARNING: standalone hdbscan package not installed (pip install hdbscan)")

SIZES = [100, 200, 500, 1000, 2000, 5000, 10000, 50000]
DIMS = 2
N_CENTERS = 5
MCS = 10
N_RUNS = 3


def run_sklearn(X, min_cluster_size):
    times = []
    labels = None
    for _ in range(N_RUNS):
        hdb = SklearnHDBSCAN(min_cluster_size=min_cluster_size, copy=True)
        t0 = time.perf_counter()
        hdb.fit(X)
        times.append(time.perf_counter() - t0)
        labels = hdb.labels_
    return min(times), labels


def run_c_hdbscan(X, min_cluster_size):
    """Run the standalone C-based hdbscan library."""
    if not HAS_C_HDBSCAN:
        return None, None
    times = []
    labels = None
    for _ in range(N_RUNS):
        clusterer = hdbscan_c.HDBSCAN(min_cluster_size=min_cluster_size)
        t0 = time.perf_counter()
        labels = clusterer.fit_predict(X)
        times.append(time.perf_counter() - t0)
    return min(times), labels


def run_rust(data_path, min_cluster_size):
    """Run the Rust benchmark binary."""
    result = subprocess.run(
        [
            os.path.join(os.path.dirname(__file__), "..", "target", "release", "examples", "bench_runner"),
            data_path,
            str(min_cluster_size),
            str(N_RUNS),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Rust binary failed: {result.stderr}")
        return None, None
    lines = result.stdout.strip().split("\n")
    elapsed = float(lines[0].split(":")[1])
    labels = [int(x) for x in lines[1].split(":")[1].split(",")]
    return elapsed / 1000.0, labels


def main():
    # Build the Rust benchmark runner first
    print("Building hdbscan-rs in release mode...")
    cargo = os.path.expanduser("~/.cargo/bin/cargo")
    ret = subprocess.run(
        [cargo, "build", "--release", "--example", "bench_runner"],
        cwd=os.path.join(os.path.dirname(__file__), ".."),
        capture_output=True,
        text=True,
    )
    if ret.returncode != 0:
        print(f"Build failed:\n{ret.stderr}")
        sys.exit(1)
    print("Build complete.\n")

    tmp_dir = "/tmp/hdbscan_bench"
    os.makedirs(tmp_dir, exist_ok=True)

    if HAS_C_HDBSCAN:
        header = f"{'n':>6} {'sklearn(ms)':>12} {'C-hdbscan(ms)':>14} {'rust(ms)':>12} {'rs/sk':>8} {'rs/C':>8} {'ARI(sk)':>8} {'ARI(C)':>8} {'sk_c':>5} {'C_c':>5} {'rs_c':>5}"
        sep = "-" * len(header)
    else:
        header = f"{'n':>6} {'sklearn(ms)':>12} {'rust(ms)':>12} {'speedup':>8} {'ARI':>6} {'sk_c':>5} {'rs_c':>5}"
        sep = "-" * 65

    print(header)
    print(sep)

    for n in SIZES:
        np.random.seed(42)
        X, _ = make_blobs(n_samples=n, centers=N_CENTERS, cluster_std=1.0, random_state=42)

        # Save data for Rust
        data_path = os.path.join(tmp_dir, f"data_{n}.csv")
        np.savetxt(data_path, X, delimiter=",")

        sk_time, sk_labels = run_sklearn(X, MCS)
        c_time, c_labels = run_c_hdbscan(X, MCS)
        rs_time, rs_labels = run_rust(data_path, MCS)

        if rs_time is not None:
            sk_c_count = len(set(sk_labels) - {-1})
            rs_c_count = len(set(rs_labels) - {-1})
            ari_sk = adjusted_rand_score(sk_labels, rs_labels)

            if HAS_C_HDBSCAN and c_time is not None:
                c_c_count = len(set(c_labels) - {-1})
                ari_c = adjusted_rand_score(c_labels, rs_labels)
                speedup_sk = sk_time / rs_time if rs_time > 0 else float("inf")
                speedup_c = c_time / rs_time if rs_time > 0 else float("inf")
                print(
                    f"{n:>6} {sk_time*1000:>12.2f} {c_time*1000:>14.2f} {rs_time*1000:>12.2f} "
                    f"{speedup_sk:>7.2f}x {speedup_c:>7.2f}x {ari_sk:>8.4f} {ari_c:>8.4f} "
                    f"{sk_c_count:>5} {c_c_count:>5} {rs_c_count:>5}"
                )
            else:
                speedup = sk_time / rs_time if rs_time > 0 else float("inf")
                print(
                    f"{n:>6} {sk_time*1000:>12.2f} {rs_time*1000:>12.2f} {speedup:>7.2f}x {ari_sk:>6.4f} {sk_c_count:>5} {rs_c_count:>5}"
                )
        else:
            print(f"{n:>6} {sk_time*1000:>12.2f} {'FAILED':>12}")


if __name__ == "__main__":
    main()
