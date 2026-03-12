#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "scikit-learn",
#     "hdbscan",
#     "fast-hdbscan",
#     "numpy",
# ]
# ///
"""Run side-by-side performance comparison: sklearn vs hdbscan (C) vs fast-hdbscan vs hdbscan-rs (Rust).

Compares wall time, peak memory (RSS), and clustering quality (ARI).
Each Python implementation runs in its own subprocess for independent memory measurement.

Usage: uv run tests/perf_comparison.py
"""

import json
import os
import resource
import subprocess
import sys
import time
import warnings

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

warnings.filterwarnings("ignore")

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


def _run_python_impl_subprocess(impl_name, data_path, min_cluster_size, n_runs):
    """Run a Python HDBSCAN implementation in a separate subprocess for clean memory measurement.

    Returns (best_time_sec, labels_list, peak_rss_mb) or (None, None, None) on failure.
    """
    script = f"""
import json, resource, sys, time, warnings, numpy as np
warnings.filterwarnings("ignore")
X = np.loadtxt("{data_path}", delimiter=",")
mcs = {min_cluster_size}
n_runs = {n_runs}
impl_name = "{impl_name}"

times = []
labels = None

if impl_name == "sklearn":
    from sklearn.cluster import HDBSCAN
    for _ in range(n_runs):
        h = HDBSCAN(min_cluster_size=mcs, copy=True)
        t0 = time.perf_counter()
        h.fit(X)
        times.append(time.perf_counter() - t0)
        labels = h.labels_.tolist()
elif impl_name == "c_hdbscan":
    import hdbscan
    for _ in range(n_runs):
        h = hdbscan.HDBSCAN(min_cluster_size=mcs)
        t0 = time.perf_counter()
        labels = h.fit_predict(X).tolist()
        times.append(time.perf_counter() - t0)
elif impl_name == "fast_hdbscan":
    from fast_hdbscan import HDBSCAN
    for _ in range(n_runs):
        h = HDBSCAN(min_cluster_size=mcs)
        t0 = time.perf_counter()
        labels = h.fit_predict(X).tolist()
        times.append(time.perf_counter() - t0)

peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
result = {{"time": min(times), "peak_rss_mb": peak_kb / 1024, "labels": labels}}
print(json.dumps(result))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        return None, None, None
    try:
        data = json.loads(result.stdout.strip())
        return data["time"], data["labels"], data["peak_rss_mb"]
    except (json.JSONDecodeError, KeyError):
        return None, None, None


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

        # Save data for all subprocesses
        data_path = os.path.join(tmp_dir, f"data_{label}.csv")
        np.savetxt(data_path, X, delimiter=",")

        # Run each implementation in its own subprocess
        sk_time, sk_labels, sk_mem = _run_python_impl_subprocess("sklearn", data_path, mcs, N_RUNS)
        c_time, c_labels, c_mem = _run_python_impl_subprocess("c_hdbscan", data_path, mcs, N_RUNS)
        fast_time, fast_labels, fast_mem = _run_python_impl_subprocess("fast_hdbscan", data_path, mcs, N_RUNS)
        rs_time, rs_labels, rs_mem = run_rust(data_path, mcs)

        if rs_time is None:
            print(f"{label:>12} {sk_time*1000:>10.1f} {'':>10} {'':>10} {'FAILED':>10}")
            continue

        rs_c_count = len(set(rs_labels) - {-1})
        ari_sk = adjusted_rand_score(sk_labels, rs_labels) if sk_labels else 0.0

        speedup_sk = sk_time / rs_time if sk_time and rs_time > 0 else float("inf")

        if c_time is not None:
            speedup_c = f"{c_time / rs_time:.1f}x"
            c_time_str = f"{c_time*1000:.1f}"
            c_mem_str = f"{c_mem:.0f}MB"
        else:
            speedup_c = "N/A"
            c_time_str = "N/A"
            c_mem_str = "N/A"

        if fast_time is not None:
            speedup_fast = f"{fast_time / rs_time:.1f}x"
            fast_time_str = f"{fast_time*1000:.1f}"
            fast_mem_str = f"{fast_mem:.0f}MB"
        else:
            speedup_fast = "N/A"
            fast_time_str = "N/A"
            fast_mem_str = "N/A"

        sk_time_str = f"{sk_time*1000:.1f}" if sk_time else "N/A"
        sk_mem_str = f"{sk_mem:.0f}MB" if sk_mem else "N/A"

        print(
            f"{label:>12} {sk_time_str + 'ms':>10} {c_time_str:>10} {fast_time_str:>10} {rs_time*1000:>9.1f}ms "
            f"{speedup_sk:>6.1f}x {speedup_c:>7} {speedup_fast:>7} {ari_sk:>8.4f} "
            f"{sk_mem_str:>8} {c_mem_str:>8} {fast_mem_str:>8} {rs_mem:>7.0f}MB "
            f"{rs_c_count:>5}"
        )

    print()
    print(f"Runs per benchmark: {N_RUNS} (best of {N_RUNS} wall time reported)")
    print(f"Memory: peak RSS per subprocess (each impl runs in its own process).")


if __name__ == "__main__":
    main()
