#!/usr/bin/env python3
"""Run side-by-side performance comparison: sklearn vs hdbscan (C) vs fast-hdbscan vs hdbscan-rs (Rust).

Compares wall time, peak memory (RSS), and clustering quality (ARI).

Usage: python3 tests/perf_comparison.py
"""

import os
import resource
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

# Try to import fast-hdbscan
try:
    from fast_hdbscan import HDBSCAN as FastHDBSCAN

    HAS_FAST_HDBSCAN = True
except ImportError:
    HAS_FAST_HDBSCAN = False
    print("WARNING: fast-hdbscan package not installed (pip install fast-hdbscan)")

# --- Benchmark configurations ---
# Each entry: (n_samples, n_dims, n_centers, min_cluster_size, label)
BENCHMARKS = [
    # Low-dimensional (classic spatial clustering)
    (500, 2, 5, 10, "500x2D"),
    (1000, 2, 5, 10, "1Kx2D"),
    (2000, 2, 5, 10, "2Kx2D"),
    (5000, 2, 5, 10, "5Kx2D"),
    (10000, 2, 5, 10, "10Kx2D"),
    (50000, 2, 5, 10, "50Kx2D"),
    # Medium-dimensional
    (5000, 10, 5, 10, "5Kx10D"),
    (5000, 50, 5, 10, "5Kx50D"),
    # High-dimensional (LLM embeddings)
    (2000, 256, 5, 10, "2Kx256D"),
    (1000, 256, 5, 10, "1Kx256D"),
    (500, 1536, 5, 10, "500x1536D"),
]

N_RUNS = 3


def get_peak_rss_mb():
    """Get current process peak RSS in MB (includes children)."""
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_maxrss / 1024  # Linux reports in KB


def run_sklearn(X, min_cluster_size):
    times = []
    labels = None
    for _ in range(N_RUNS):
        hdb = SklearnHDBSCAN(min_cluster_size=min_cluster_size, copy=True)
        t0 = time.perf_counter()
        hdb.fit(X)
        times.append(time.perf_counter() - t0)
        labels = hdb.labels_
    peak_mb = get_peak_rss_mb()
    return min(times), labels, peak_mb


def run_c_hdbscan(X, min_cluster_size):
    """Run the standalone C-based hdbscan library."""
    if not HAS_C_HDBSCAN:
        return None, None, None
    times = []
    labels = None
    for _ in range(N_RUNS):
        clusterer = hdbscan_c.HDBSCAN(min_cluster_size=min_cluster_size)
        t0 = time.perf_counter()
        labels = clusterer.fit_predict(X)
        times.append(time.perf_counter() - t0)
    peak_mb = get_peak_rss_mb()
    return min(times), labels, peak_mb


def run_fast_hdbscan(X, min_cluster_size):
    """Run fast-hdbscan."""
    if not HAS_FAST_HDBSCAN:
        return None, None, None
    times = []
    labels = None
    for _ in range(N_RUNS):
        clusterer = FastHDBSCAN(min_cluster_size=min_cluster_size)
        t0 = time.perf_counter()
        labels = clusterer.fit_predict(X)
        times.append(time.perf_counter() - t0)
    peak_mb = get_peak_rss_mb()
    return min(times), labels, peak_mb


def run_rust(data_path, min_cluster_size):
    """Run the Rust benchmark binary and parse time + memory."""
    bin_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "target",
        "release",
        "examples",
        "bench_runner",
    )
    result = subprocess.run(
        [bin_path, data_path, str(min_cluster_size), str(N_RUNS)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  Rust binary failed: {result.stderr.strip()}")
        return None, None, None
    lines = result.stdout.strip().split("\n")
    elapsed = float(lines[0].split(":")[1])
    peak_kb = 0
    label_line = 1
    for ln in lines:
        if ln.startswith("peak_rss_kb:"):
            peak_kb = int(ln.split(":")[1])
        if ln.startswith("labels:"):
            label_line = lines.index(ln)
    labels = [int(x) for x in lines[label_line].split(":")[1].split(",")]
    peak_mb = peak_kb / 1024.0
    return elapsed / 1000.0, labels, peak_mb


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

    # Header
    print(
        f"{'config':>12} {'sklearn':>10} {'C-hdb':>10} {'fast-hdb':>10} {'rust':>10} "
        f"{'rs/sk':>7} {'rs/C':>7} {'rs/fast':>7} {'ARI(sk)':>8} "
        f"{'sk_mem':>8} {'C_mem':>8} {'fast_mem':>8} {'rs_mem':>8} "
        f"{'clust':>5}"
    )
    print("-" * 145)

    for n, dims, centers, mcs, label in BENCHMARKS:
        np.random.seed(42)
        X, _ = make_blobs(
            n_samples=n,
            n_features=dims,
            centers=centers,
            cluster_std=1.0,
            random_state=42,
        )

        # Save data for Rust
        data_path = os.path.join(tmp_dir, f"data_{label}.csv")
        np.savetxt(data_path, X, delimiter=",")

        sk_time, sk_labels, sk_mem = run_sklearn(X, mcs)
        c_time, c_labels, c_mem = run_c_hdbscan(X, mcs)
        fast_time, fast_labels, fast_mem = run_fast_hdbscan(X, mcs)
        rs_time, rs_labels, rs_mem = run_rust(data_path, mcs)

        if rs_time is None:
            print(f"{label:>12} {sk_time*1000:>10.1f} {'':>10} {'':>10} {'FAILED':>10}")
            continue

        rs_c_count = len(set(rs_labels) - {-1})
        ari_sk = adjusted_rand_score(sk_labels, rs_labels)

        speedup_sk = sk_time / rs_time if rs_time > 0 else float("inf")
        speedup_c = ""
        c_time_str = ""
        c_mem_str = ""

        if HAS_C_HDBSCAN and c_time is not None:
            speedup_c_val = c_time / rs_time if rs_time > 0 else float("inf")
            speedup_c = f"{speedup_c_val:.1f}x"
            c_time_str = f"{c_time*1000:.1f}"
            c_mem_str = f"{c_mem:.0f}MB"
        else:
            speedup_c = "N/A"
            c_time_str = "N/A"
            c_mem_str = "N/A"

        speedup_fast = ""
        fast_time_str = ""
        fast_mem_str = ""

        if HAS_FAST_HDBSCAN and fast_time is not None:
            speedup_fast_val = fast_time / rs_time if rs_time > 0 else float("inf")
            speedup_fast = f"{speedup_fast_val:.1f}x"
            fast_time_str = f"{fast_time*1000:.1f}"
            fast_mem_str = f"{fast_mem:.0f}MB"
        else:
            speedup_fast = "N/A"
            fast_time_str = "N/A"
            fast_mem_str = "N/A"

        print(
            f"{label:>12} {sk_time*1000:>9.1f}ms {c_time_str:>10} {fast_time_str:>10} {rs_time*1000:>9.1f}ms "
            f"{speedup_sk:>6.1f}x {speedup_c:>7} {speedup_fast:>7} {ari_sk:>8.4f} "
            f"{sk_mem:>7.0f}MB {c_mem_str:>8} {fast_mem_str:>8} {rs_mem:>7.0f}MB "
            f"{rs_c_count:>5}"
        )

    print()
    print(f"Runs per benchmark: {N_RUNS} (best of {N_RUNS} wall time reported)")
    print(f"Memory: peak RSS. sklearn/C-hdb/fast-hdb share Python process; Rust runs as subprocess.")


if __name__ == "__main__":
    main()
