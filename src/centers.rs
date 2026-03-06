use crate::params::StoreCenters;
use ndarray::{ArrayView2, Array1, Array2};

/// Compute cluster centers (centroids and/or medoids).
pub fn compute_centers(
    data: &ArrayView2<f64>,
    labels: &[i32],
    n_clusters: usize,
    mode: StoreCenters,
) -> (Option<Array2<f64>>, Option<Array2<f64>>) {
    let n_features = data.ncols();

    let centroids = if matches!(mode, StoreCenters::Centroid | StoreCenters::Both) {
        Some(compute_centroids(data, labels, n_clusters, n_features))
    } else {
        None
    };

    let medoids = if matches!(mode, StoreCenters::Medoid | StoreCenters::Both) {
        Some(compute_medoids(data, labels, n_clusters, n_features))
    } else {
        None
    };

    (centroids, medoids)
}

fn compute_centroids(
    data: &ArrayView2<f64>,
    labels: &[i32],
    n_clusters: usize,
    n_features: usize,
) -> Array2<f64> {
    let mut centroids = Array2::zeros((n_clusters, n_features));
    let mut counts = vec![0usize; n_clusters];

    for (i, &label) in labels.iter().enumerate() {
        if label >= 0 {
            let c = label as usize;
            counts[c] += 1;
            for f in 0..n_features {
                centroids[[c, f]] += data[[i, f]];
            }
        }
    }

    for c in 0..n_clusters {
        if counts[c] > 0 {
            for f in 0..n_features {
                centroids[[c, f]] /= counts[c] as f64;
            }
        }
    }

    centroids
}

fn compute_medoids(
    data: &ArrayView2<f64>,
    labels: &[i32],
    n_clusters: usize,
    n_features: usize,
) -> Array2<f64> {
    let mut medoids = Array2::zeros((n_clusters, n_features));

    for c in 0..n_clusters {
        let points: Vec<usize> = labels
            .iter()
            .enumerate()
            .filter(|(_, &l)| l == c as i32)
            .map(|(i, _)| i)
            .collect();

        if points.is_empty() {
            continue;
        }

        // Centroid for distance computation
        let centroid = {
            let mut c = Array1::zeros(n_features);
            for &p in &points {
                for f in 0..n_features {
                    c[f] += data[[p, f]];
                }
            }
            let n = points.len() as f64;
            c.mapv_inplace(|v: f64| v / n);
            c
        };

        // Find point closest to centroid
        let mut best_idx = points[0];
        let mut best_dist = f64::INFINITY;
        for &p in &points {
            let dist: f64 = (0..n_features)
                .map(|f| {
                    let d = data[[p, f]] - centroid[f];
                    d * d
                })
                .sum::<f64>()
                .sqrt();
            if dist < best_dist {
                best_dist = dist;
                best_idx = p;
            }
        }

        for f in 0..n_features {
            medoids[[c, f]] = data[[best_idx, f]];
        }
    }

    medoids
}
