"""Tests for the hdbscan_rs Python bindings."""

import numpy as np
from hdbscan_rs import HDBSCAN


def test_two_clusters():
    data = np.array([
        [0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1], [0.05, 0.05],
        [10.0, 10.0], [10.1, 10.0], [10.0, 10.1], [10.1, 10.1], [10.05, 10.05],
    ])
    h = HDBSCAN(min_cluster_size=3)
    labels = h.fit_predict(data)

    assert len(labels) == 10
    assert labels[0] == labels[1] == labels[2] == labels[3] == labels[4]
    assert labels[5] == labels[6] == labels[7] == labels[8] == labels[9]
    assert labels[0] != labels[5]
    print("  two_clusters: OK")


def test_probabilities_and_outlier_scores():
    data = np.array([
        [0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1], [0.05, 0.05],
        [10.0, 10.0], [10.1, 10.0], [10.0, 10.1], [10.1, 10.1], [10.05, 10.05],
    ])
    h = HDBSCAN(min_cluster_size=3)
    h.fit(data)

    probs = h.probabilities_
    assert len(probs) == 10
    assert all(0.0 <= p <= 1.0 for p in probs)

    scores = h.outlier_scores_
    assert len(scores) == 10
    assert all(0.0 <= s <= 1.0 for s in scores)
    print("  probabilities_and_outlier_scores: OK")


def test_approximate_predict():
    data = np.array([
        [0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1], [0.05, 0.05],
        [10.0, 10.0], [10.1, 10.0], [10.0, 10.1], [10.1, 10.1], [10.05, 10.05],
    ])
    h = HDBSCAN(min_cluster_size=3)
    h.fit(data)

    new_points = np.array([[0.05, 0.0], [10.05, 10.0]])
    pred_labels, pred_probs = h.approximate_predict(new_points)

    assert len(pred_labels) == 2
    assert len(pred_probs) == 2
    # First point should be in same cluster as [0,0] group
    assert pred_labels[0] == h.labels_[0]
    # Second point should be in same cluster as [10,10] group
    assert pred_labels[1] == h.labels_[5]
    print("  approximate_predict: OK")


def test_all_noise():
    data = np.array([[0.0, 0.0], [100.0, 100.0]])
    h = HDBSCAN(min_cluster_size=5, min_samples=2)
    labels = h.fit_predict(data)
    assert all(l == -1 for l in labels)
    print("  all_noise: OK")


def test_metrics():
    data = np.random.RandomState(42).randn(50, 3)
    for metric in ["euclidean", "manhattan", "cosine"]:
        h = HDBSCAN(min_cluster_size=5, metric=metric)
        labels = h.fit_predict(data)
        assert len(labels) == 50
    print("  metrics: OK")


def test_precomputed():
    np.random.seed(42)
    pts = np.vstack([
        np.random.randn(15, 2),
        np.random.randn(15, 2) + 10,
    ])
    dist = np.sqrt(((pts[:, None] - pts[None, :]) ** 2).sum(axis=2))

    h = HDBSCAN(min_cluster_size=5, metric="precomputed")
    labels = h.fit_predict(dist)
    assert len(labels) == 30
    n_clusters = len(set(l for l in labels if l >= 0))
    assert n_clusters >= 2
    print("  precomputed: OK")


def test_repr():
    h = HDBSCAN(min_cluster_size=10, metric="manhattan")
    r = repr(h)
    assert "min_cluster_size=10" in r
    assert "manhattan" in r
    print("  repr: OK")


def test_not_fitted_error():
    h = HDBSCAN()
    try:
        _ = h.labels_
        assert False, "Should have raised"
    except ValueError:
        pass
    print("  not_fitted_error: OK")


def test_invalid_metric():
    try:
        HDBSCAN(metric="bogus")
        assert False, "Should have raised"
    except ValueError:
        pass
    print("  invalid_metric: OK")


def test_leaf_selection():
    data = np.array([
        [0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1], [0.05, 0.05],
        [10.0, 10.0], [10.1, 10.0], [10.0, 10.1], [10.1, 10.1], [10.05, 10.05],
    ])
    h = HDBSCAN(min_cluster_size=3, cluster_selection_method="leaf")
    labels = h.fit_predict(data)
    assert len(labels) == 10
    print("  leaf_selection: OK")


def test_float32_input():
    data = np.array([
        [0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1], [0.05, 0.05],
        [10.0, 10.0], [10.1, 10.0], [10.0, 10.1], [10.1, 10.1], [10.05, 10.05],
    ], dtype=np.float32)
    assert data.dtype == np.float32
    h = HDBSCAN(min_cluster_size=3)
    labels = h.fit_predict(data)
    assert len(labels) == 10
    assert labels[0] == labels[1] == labels[2] == labels[3] == labels[4]
    assert labels[5] == labels[6] == labels[7] == labels[8] == labels[9]
    assert labels[0] != labels[5]
    print("  float32_input: OK")


def test_sklearn_dropin():
    try:
        from sklearn.cluster import HDBSCAN as SklearnHDBSCAN
        from sklearn.datasets import make_blobs
        from sklearn.metrics import adjusted_rand_score
    except ImportError:
        print("  sklearn_dropin: SKIPPED (sklearn not installed)")
        return

    X, y_true = make_blobs(n_samples=300, centers=5, random_state=42)

    sk = SklearnHDBSCAN(min_cluster_size=15)
    sk_labels = sk.fit_predict(X)

    rs = HDBSCAN(min_cluster_size=15)
    rs_labels = rs.fit_predict(X)

    ari = adjusted_rand_score(sk_labels, rs_labels)
    assert ari > 0.95, f"ARI between sklearn and hdbscan_rs was {ari}, expected > 0.95"
    print(f"  sklearn_dropin: OK (ARI={ari:.4f})")


def test_standalone_hdbscan_dropin():
    try:
        import hdbscan as hdbscan_pkg
        from sklearn.datasets import make_blobs
        from sklearn.metrics import adjusted_rand_score
    except ImportError:
        print("  standalone_hdbscan_dropin: SKIPPED (hdbscan or sklearn not installed)")
        return

    X, y_true = make_blobs(n_samples=300, centers=5, random_state=42)

    ref = hdbscan_pkg.HDBSCAN(min_cluster_size=15)
    ref_labels = ref.fit_predict(X)

    rs = HDBSCAN(min_cluster_size=15)
    rs_labels = rs.fit_predict(X)

    ari = adjusted_rand_score(ref_labels, rs_labels)
    assert ari > 0.95, f"ARI between hdbscan and hdbscan_rs was {ari}, expected > 0.95"
    print(f"  standalone_hdbscan_dropin: OK (ARI={ari:.4f})")


def test_bertopic_compatible_interface():
    h = HDBSCAN(min_cluster_size=5)

    # BERTopic expects fit_predict as a callable method
    assert callable(getattr(h, "fit_predict", None)), "Missing fit_predict method"

    # Fit on some data so we can check post-fit attributes
    data = np.array([
        [0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1], [0.05, 0.05],
        [10.0, 10.0], [10.1, 10.0], [10.0, 10.1], [10.1, 10.1], [10.05, 10.05],
    ])
    h.fit_predict(data)

    # BERTopic reads labels_ and probabilities_ after fitting
    labels = h.labels_
    assert labels is not None
    assert len(labels) == 10

    probs = h.probabilities_
    assert probs is not None
    assert len(probs) == 10
    assert all(0.0 <= p <= 1.0 for p in probs)

    print("  bertopic_compatible_interface: OK")


def test_large_dataset():
    rng = np.random.RandomState(42)
    data = rng.randn(5000, 10)
    # Plant some structure: shift 3 groups apart
    data[:1500] += 20
    data[1500:3000] -= 20

    h = HDBSCAN(min_cluster_size=50)
    labels = h.fit_predict(data)

    assert len(labels) == 5000
    n_clusters = len(set(l for l in labels if l >= 0))
    assert n_clusters >= 2, f"Expected at least 2 clusters, got {n_clusters}"
    n_noise = sum(1 for l in labels if l == -1)
    assert n_noise < 5000, "All points classified as noise"
    print(f"  large_dataset: OK ({n_clusters} clusters, {n_noise} noise)")


if __name__ == "__main__":
    print("Running Python binding tests...")
    test_two_clusters()
    test_probabilities_and_outlier_scores()
    test_approximate_predict()
    test_all_noise()
    test_metrics()
    test_precomputed()
    test_repr()
    test_not_fitted_error()
    test_invalid_metric()
    test_leaf_selection()
    test_float32_input()
    test_sklearn_dropin()
    test_standalone_hdbscan_dropin()
    test_bertopic_compatible_interface()
    test_large_dataset()
    print("All Python tests passed!")
