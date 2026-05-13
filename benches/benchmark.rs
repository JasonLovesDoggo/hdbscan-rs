use hdbscan_rs::types::CondensedTreeEdge;
use hdbscan_rs::{
    cluster_selection, condensed_tree, core_distance, labels, linkage, membership, mst, outlier,
    ClusterSelectionMethod, Hdbscan, HdbscanParams,
};
use ndarray::{Array2, ArrayView2};
use std::collections::HashSet;
use std::hint::black_box;
use std::time::{Duration, Instant};

const SAMPLES: usize = 7;
const OUTPUT_TOLERANCE: f64 = 1e-5;

fn main() {
    for case in [
        BenchCase {
            name: "blobs_1000_eom_mcs10",
            n_points: 1_000,
            n_features: 8,
            n_clusters: 5,
            min_cluster_size: 10,
            method: ClusterSelectionMethod::Eom,
            iterations: 2_000,
            full_fit_iterations: 100,
        },
        BenchCase {
            name: "blobs_5000_eom_mcs25",
            n_points: 5_000,
            n_features: 8,
            n_clusters: 8,
            min_cluster_size: 25,
            method: ClusterSelectionMethod::Eom,
            iterations: 700,
            full_fit_iterations: 30,
        },
        BenchCase {
            name: "blobs_10000_eom_mcs50",
            n_points: 10_000,
            n_features: 8,
            n_clusters: 10,
            min_cluster_size: 50,
            method: ClusterSelectionMethod::Eom,
            iterations: 350,
            full_fit_iterations: 15,
        },
        BenchCase {
            name: "blobs_10000_leaf_mcs50",
            n_points: 10_000,
            n_features: 8,
            n_clusters: 10,
            min_cluster_size: 50,
            method: ClusterSelectionMethod::Leaf,
            iterations: 350,
            full_fit_iterations: 15,
        },
    ] {
        run_case(case);
    }
}

#[derive(Clone, Copy)]
struct BenchCase {
    name: &'static str,
    n_points: usize,
    n_features: usize,
    n_clusters: usize,
    min_cluster_size: usize,
    method: ClusterSelectionMethod,
    iterations: usize,
    full_fit_iterations: usize,
}

fn run_case(case: BenchCase) {
    let data = make_blobs(case.n_points, case.n_features, case.n_clusters);
    let params = HdbscanParams {
        min_cluster_size: case.min_cluster_size,
        cluster_selection_method: case.method,
        ..Default::default()
    };
    let mut hdbscan = Hdbscan::new(params.clone());
    let fit_start = Instant::now();
    hdbscan
        .fit(&data.view())
        .expect("benchmark fixture should fit");
    let fit_ms = fit_start.elapsed().as_secs_f64() * 1000.0;
    let condensed = hdbscan
        .condensed_tree()
        .expect("fit should produce condensed tree");

    let deltas = assert_same_cluster_assignment(condensed, case.n_points, &params);

    warm_pipeline(condensed, case.n_points, &params);
    let legacy = measure_pipeline(
        || legacy::pipeline(condensed, case.n_points, &params),
        case.iterations,
    );
    let current = measure_pipeline(
        || current_pipeline(condensed, case.n_points, &params),
        case.iterations,
    );

    println!(
        "hdbscan postprocess {} n={} d={} method={:?} condensed_edges={} fit_once={:.3}ms iterations={}: legacy_median={:.6}ms current_median={:.6}ms speedup={:.2}x legacy_p95={:.6}ms current_p95={:.6}ms max_prob_delta={:.6} max_outlier_delta={:.6}",
        case.name,
        case.n_points,
        case.n_features,
        case.method,
        condensed.len(),
        fit_ms,
        case.iterations,
        legacy.median_ms,
        current.median_ms,
        legacy.median_ms / current.median_ms,
        legacy.p95_ms,
        current.p95_ms,
        deltas.max_probability_delta,
        deltas.max_outlier_delta,
    );

    let legacy_full = measure_full_fit(
        || legacy::full_fit(&data.view(), &params),
        case.full_fit_iterations,
    );
    let current_full = measure_full_fit(
        || current_full_fit(&data.view(), &params),
        case.full_fit_iterations,
    );
    assert_eq!(
        legacy_full.last_output.labels,
        current_full.last_output.labels
    );
    assert_eq!(
        legacy_full.last_output.condensed_edges,
        current_full.last_output.condensed_edges
    );
    assert_close_full_outputs(&legacy_full.last_output, &current_full.last_output);
    println!(
        "hdbscan full-fit {} n={} d={} method={:?} iterations={}: legacy_median={:.3}ms current_median={:.3}ms speedup={:.3}x legacy_p95={:.3}ms current_p95={:.3}ms max_prob_delta={:.6} max_outlier_delta={:.6}",
        case.name,
        case.n_points,
        case.n_features,
        case.method,
        case.full_fit_iterations,
        legacy_full.timing.median_ms,
        current_full.timing.median_ms,
        legacy_full.timing.median_ms / current_full.timing.median_ms,
        legacy_full.timing.p95_ms,
        current_full.timing.p95_ms,
        max_abs_delta(
            &legacy_full.last_output.probabilities,
            &current_full.last_output.probabilities,
        ),
        max_abs_delta(
            &legacy_full.last_output.outlier_scores,
            &current_full.last_output.outlier_scores,
        ),
    );
}

fn warm_pipeline(condensed: &[CondensedTreeEdge], n_points: usize, params: &HdbscanParams) {
    black_box(legacy::pipeline(condensed, n_points, params));
    black_box(current_pipeline(condensed, n_points, params));
}

fn assert_same_cluster_assignment(
    condensed: &[CondensedTreeEdge],
    n_points: usize,
    params: &HdbscanParams,
) -> OutputDeltas {
    let legacy = legacy::pipeline(condensed, n_points, params);
    let current = current_pipeline(condensed, n_points, params);
    assert_eq!(legacy.selected, current.selected);
    assert_eq!(legacy.labels, current.labels);
    OutputDeltas {
        max_probability_delta: max_abs_delta(&legacy.probabilities, &current.probabilities),
        max_outlier_delta: max_abs_delta(&legacy.outlier_scores, &current.outlier_scores),
    }
}

#[derive(Clone, Copy)]
struct OutputDeltas {
    max_probability_delta: f64,
    max_outlier_delta: f64,
}

fn max_abs_delta(left: &[f64], right: &[f64]) -> f64 {
    assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right.iter())
        .map(|(&left, &right)| (left - right).abs())
        .fold(0.0, f64::max)
}

fn current_pipeline(
    condensed: &[CondensedTreeEdge],
    n_points: usize,
    params: &HdbscanParams,
) -> PipelineOutput {
    let selection = cluster_selection::select_clusters(
        condensed,
        n_points,
        params.cluster_selection_method,
        params.cluster_selection_epsilon,
        params.allow_single_cluster,
    );
    let labels = labels::assign_labels(
        condensed,
        &selection.selected_clusters,
        n_points,
        params.allow_single_cluster,
        params.cluster_selection_epsilon,
    );
    let probabilities = membership::compute_probabilities(
        condensed,
        &selection.selected_clusters,
        &labels,
        n_points,
    );
    let outlier_scores = outlier::compute_outlier_scores(condensed, n_points);
    PipelineOutput {
        selected: selection.selected_clusters,
        labels,
        probabilities,
        outlier_scores,
    }
}

fn current_full_fit(data: &ArrayView2<f64>, params: &HdbscanParams) -> FullFitOutput {
    let min_samples = params.effective_min_samples();
    let core_distances = core_distance::compute_core_distances(data, &params.metric, min_samples);
    let mst_edges = mst::auto_mst(
        data,
        &core_distances.view(),
        &params.metric,
        params.alpha,
        None,
    );
    let single_linkage = linkage::mst_to_single_linkage(&mst_edges, data.nrows());
    let condensed = condensed_tree::build_condensed_tree(
        &single_linkage,
        data.nrows(),
        params.min_cluster_size,
    );
    let output = current_pipeline(&condensed, data.nrows(), params);

    let retained_training = data.to_owned();
    let retained_core_distances = core_distances.to_owned();
    black_box((retained_training, retained_core_distances));

    FullFitOutput {
        labels: output.labels,
        probabilities: output.probabilities,
        outlier_scores: output.outlier_scores,
        condensed_edges: condensed.len(),
    }
}

#[derive(Debug)]
struct PipelineOutput {
    selected: HashSet<usize>,
    labels: Vec<i32>,
    probabilities: Vec<f64>,
    outlier_scores: Vec<f64>,
}

#[derive(Debug)]
struct FullFitOutput {
    labels: Vec<i32>,
    probabilities: Vec<f64>,
    outlier_scores: Vec<f64>,
    condensed_edges: usize,
}

#[derive(Clone, Copy)]
struct Timing {
    median_ms: f64,
    p95_ms: f64,
}

struct FullFitTiming {
    timing: Timing,
    last_output: FullFitOutput,
}

fn measure_pipeline<F>(mut pipeline: F, iterations: usize) -> Timing
where
    F: FnMut() -> PipelineOutput,
{
    let mut samples = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let start = Instant::now();
        for _ in 0..iterations {
            black_box(pipeline());
        }
        samples.push(start.elapsed());
    }
    timing(samples, iterations)
}

fn measure_full_fit<F>(mut fit: F, iterations: usize) -> FullFitTiming
where
    F: FnMut() -> FullFitOutput,
{
    let mut samples = Vec::with_capacity(SAMPLES);
    let mut last_output = None;
    for _ in 0..SAMPLES {
        let start = Instant::now();
        for _ in 0..iterations {
            last_output = Some(black_box(fit()));
        }
        samples.push(start.elapsed());
    }
    FullFitTiming {
        timing: timing(samples, iterations),
        last_output: last_output.expect("at least one fit should run"),
    }
}

fn timing(mut samples: Vec<Duration>, iterations: usize) -> Timing {
    samples.sort_unstable();
    let median = samples[samples.len() / 2];
    let p95_index = ((samples.len() as f64) * 0.95).ceil() as usize - 1;
    let p95 = samples[p95_index.min(samples.len() - 1)];
    Timing {
        median_ms: median.as_secs_f64() * 1000.0 / iterations as f64,
        p95_ms: p95.as_secs_f64() * 1000.0 / iterations as f64,
    }
}

fn assert_close_full_outputs(left: &FullFitOutput, right: &FullFitOutput) {
    assert_close_values(&left.probabilities, &right.probabilities);
    assert_close_values(&left.outlier_scores, &right.outlier_scores);
}

fn assert_close_values(left: &[f64], right: &[f64]) {
    let delta = max_abs_delta(left, right);
    assert!(
        delta <= OUTPUT_TOLERANCE,
        "max output delta {delta} exceeds tolerance {OUTPUT_TOLERANCE}"
    );
}

fn make_blobs(n_points: usize, n_features: usize, n_clusters: usize) -> Array2<f64> {
    let mut data = Array2::zeros((n_points, n_features));
    let mut state = 0x9e37_79b9_7f4a_7c15u64 ^ n_points as u64;
    for row in 0..n_points {
        let cluster = row % n_clusters;
        for feature in 0..n_features {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let noise = ((state >> 33) as f64 / (1u64 << 31) as f64) - 0.5;
            let center = cluster as f64 * 12.0 + feature as f64 * 0.17;
            data[[row, feature]] = center + noise;
        }
    }
    data
}

mod legacy {
    use hdbscan_rs::types::CondensedTreeEdge;
    use hdbscan_rs::{ClusterSelectionMethod, HdbscanParams};
    use ndarray::ArrayView2;
    use std::collections::{HashMap, HashSet};

    use crate::{condensed_tree, core_distance, linkage, mst, FullFitOutput, PipelineOutput};

    pub fn pipeline(
        condensed: &[CondensedTreeEdge],
        n_points: usize,
        params: &HdbscanParams,
    ) -> PipelineOutput {
        let selected = select_clusters(
            condensed,
            n_points,
            params.cluster_selection_method,
            params.cluster_selection_epsilon,
            params.allow_single_cluster,
        );
        let labels = assign_labels(
            condensed,
            &selected,
            n_points,
            params.allow_single_cluster,
            params.cluster_selection_epsilon,
        );
        let probabilities = compute_probabilities(condensed, &selected, &labels, n_points);
        let outlier_scores = compute_outlier_scores(condensed, n_points);
        PipelineOutput {
            selected,
            labels,
            probabilities,
            outlier_scores,
        }
    }

    pub fn full_fit(data: &ArrayView2<f64>, params: &HdbscanParams) -> FullFitOutput {
        let min_samples = params.effective_min_samples();
        let core_distances =
            core_distance::compute_core_distances(data, &params.metric, min_samples);
        let mst_edges = mst::auto_mst(
            data,
            &core_distances.view(),
            &params.metric,
            params.alpha,
            None,
        );
        let single_linkage = linkage::mst_to_single_linkage(&mst_edges, data.nrows());
        let condensed = condensed_tree::build_condensed_tree(
            &single_linkage,
            data.nrows(),
            params.min_cluster_size,
        );
        let output = pipeline(&condensed, data.nrows(), params);

        let retained_training = data.to_owned();
        let retained_core_distances = core_distances.to_owned();
        std::hint::black_box((retained_training, retained_core_distances));

        FullFitOutput {
            labels: output.labels,
            probabilities: output.probabilities,
            outlier_scores: output.outlier_scores,
            condensed_edges: condensed.len(),
        }
    }

    fn select_clusters(
        condensed_tree: &[CondensedTreeEdge],
        n_points: usize,
        method: ClusterSelectionMethod,
        cluster_selection_epsilon: f64,
        allow_single_cluster: bool,
    ) -> HashSet<usize> {
        if condensed_tree.is_empty() {
            return HashSet::new();
        }

        let mut all_clusters: HashSet<usize> = HashSet::new();
        let mut children_of: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut cluster_birth_lambda: HashMap<usize, f64> = HashMap::new();

        for edge in condensed_tree {
            all_clusters.insert(edge.parent);
            if edge.child >= n_points {
                all_clusters.insert(edge.child);
                children_of.entry(edge.parent).or_default().push(edge.child);
                cluster_birth_lambda
                    .entry(edge.child)
                    .and_modify(|v| {
                        if edge.lambda_val < *v {
                            *v = edge.lambda_val;
                        }
                    })
                    .or_insert(edge.lambda_val);
            }
        }

        let root = *all_clusters.iter().min().unwrap_or(&n_points);
        cluster_birth_lambda.entry(root).or_insert(0.0);

        let leaf_clusters: HashSet<usize> = all_clusters
            .iter()
            .filter(|c| !children_of.contains_key(c))
            .copied()
            .collect();

        let selected = match method {
            ClusterSelectionMethod::Eom => eom_selection(
                condensed_tree,
                n_points,
                &all_clusters,
                &children_of,
                allow_single_cluster,
            ),
            ClusterSelectionMethod::Leaf => leaf_clusters,
        };

        if cluster_selection_epsilon > 0.0 {
            apply_epsilon_merging(
                &selected,
                &children_of,
                &cluster_birth_lambda,
                cluster_selection_epsilon,
            )
        } else {
            selected
        }
    }

    fn eom_selection(
        condensed_tree: &[CondensedTreeEdge],
        n_points: usize,
        all_clusters: &HashSet<usize>,
        children_of: &HashMap<usize, Vec<usize>>,
        allow_single_cluster: bool,
    ) -> HashSet<usize> {
        let root = *all_clusters.iter().min().unwrap_or(&n_points);
        let mut births: HashMap<usize, f64> = HashMap::new();
        for edge in condensed_tree {
            births.insert(edge.child, edge.lambda_val);
        }
        births.insert(root, 0.0);

        let mut stability: HashMap<usize, f64> = HashMap::new();
        for &cluster in all_clusters {
            stability.insert(cluster, 0.0);
        }
        for edge in condensed_tree {
            let birth_lambda = *births.get(&edge.parent).unwrap_or(&0.0);
            let contribution = (edge.lambda_val - birth_lambda) * edge.child_size as f64;
            *stability.entry(edge.parent).or_insert(0.0) += contribution;
        }

        let mut node_list: Vec<usize> = if allow_single_cluster {
            all_clusters.iter().copied().collect()
        } else {
            all_clusters
                .iter()
                .copied()
                .filter(|&cluster| cluster != root)
                .collect()
        };
        node_list.sort_unstable_by(|a, b| b.cmp(a));

        let mut is_cluster: HashMap<usize, bool> = HashMap::new();
        for &cluster in &node_list {
            is_cluster.insert(cluster, true);
        }

        for &node in &node_list {
            if let Some(children) = children_of.get(&node) {
                let subtree_stability: f64 = children
                    .iter()
                    .map(|child| *stability.get(child).unwrap_or(&0.0))
                    .sum();
                let own_stability = *stability.get(&node).unwrap_or(&0.0);
                if subtree_stability > own_stability {
                    is_cluster.insert(node, false);
                    stability.insert(node, subtree_stability);
                } else {
                    for sub_node in bfs_descendants(node, children_of) {
                        is_cluster.insert(sub_node, false);
                    }
                }
            }
        }

        let selected: HashSet<usize> = is_cluster
            .iter()
            .filter(|(_, &selected)| selected)
            .map(|(&cluster, _)| cluster)
            .collect();
        if !allow_single_cluster && selected.len() == 1 && selected.contains(&root) {
            HashSet::new()
        } else {
            selected
        }
    }

    fn bfs_descendants(node: usize, children_of: &HashMap<usize, Vec<usize>>) -> Vec<usize> {
        let mut result = Vec::new();
        let mut queue = Vec::new();
        if let Some(children) = children_of.get(&node) {
            queue.extend(children.iter().copied());
        }
        while let Some(current) = queue.pop() {
            result.push(current);
            if let Some(children) = children_of.get(&current) {
                queue.extend(children.iter().copied());
            }
        }
        result
    }

    fn apply_epsilon_merging(
        selected: &HashSet<usize>,
        children_of: &HashMap<usize, Vec<usize>>,
        birth_lambda: &HashMap<usize, f64>,
        epsilon: f64,
    ) -> HashSet<usize> {
        let epsilon_lambda = if epsilon > 0.0 {
            1.0 / epsilon
        } else {
            f64::INFINITY
        };
        let mut result = selected.clone();
        let mut changed = true;
        while changed {
            changed = false;
            let current = result.clone();
            for &cluster in &current {
                if let Some(children) = children_of.get(&cluster) {
                    let all_children_selected_and_fine = children.iter().all(|child| {
                        result.contains(child)
                            && birth_lambda.get(child).copied().unwrap_or(0.0) > epsilon_lambda
                    });
                    if all_children_selected_and_fine && !children.is_empty() {
                        for &child in children {
                            result.remove(&child);
                        }
                        result.insert(cluster);
                        changed = true;
                    }
                }
            }
        }
        result
    }

    fn assign_labels(
        condensed_tree: &[CondensedTreeEdge],
        selected_clusters: &HashSet<usize>,
        n_points: usize,
        allow_single_cluster: bool,
        cluster_selection_epsilon: f64,
    ) -> Vec<i32> {
        if selected_clusters.is_empty() {
            return vec![-1; n_points];
        }

        let mut sorted_selected: Vec<usize> = selected_clusters.iter().copied().collect();
        sorted_selected.sort_unstable();
        let cluster_to_label: HashMap<usize, i32> = sorted_selected
            .iter()
            .enumerate()
            .map(|(i, &cluster)| (cluster, i as i32))
            .collect();

        let root_cluster = condensed_tree.iter().map(|edge| edge.parent).min().unwrap();
        let max_parent = condensed_tree
            .iter()
            .map(|edge| edge.parent)
            .max()
            .unwrap_or(0);
        let max_child = condensed_tree
            .iter()
            .map(|edge| edge.child)
            .max()
            .unwrap_or(0);
        let mut uf = UnionFind::new(max_parent.max(max_child) + 1);

        for edge in condensed_tree {
            if !selected_clusters.contains(&edge.child) {
                uf.union(edge.parent, edge.child);
            }
        }

        let mut point_lambda: HashMap<usize, f64> = HashMap::new();
        for edge in condensed_tree {
            if edge.child < n_points {
                point_lambda.insert(edge.child, edge.lambda_val);
            }
        }

        let single_cluster_threshold = if selected_clusters.len() == 1
            && allow_single_cluster
            && selected_clusters.contains(&root_cluster)
        {
            if cluster_selection_epsilon != 0.0 {
                Some(1.0 / cluster_selection_epsilon)
            } else {
                Some(
                    condensed_tree
                        .iter()
                        .filter(|edge| edge.parent == root_cluster)
                        .map(|edge| edge.lambda_val)
                        .fold(0.0f64, f64::max),
                )
            }
        } else {
            None
        };

        let mut labels = vec![-1i32; n_points];
        for (point, label_slot) in labels.iter_mut().enumerate() {
            let cluster = uf.find(point);
            if cluster != root_cluster {
                if let Some(&label) = cluster_to_label.get(&cluster) {
                    *label_slot = label;
                }
            } else if let Some(threshold) = single_cluster_threshold {
                let lambda = point_lambda.get(&point).copied().unwrap_or(0.0);
                if lambda >= threshold {
                    if let Some(&label) = cluster_to_label.get(&root_cluster) {
                        *label_slot = label;
                    }
                }
            }
        }
        labels
    }

    fn compute_probabilities(
        condensed_tree: &[CondensedTreeEdge],
        selected_clusters: &HashSet<usize>,
        labels: &[i32],
        n_points: usize,
    ) -> Vec<f64> {
        if selected_clusters.is_empty() {
            return vec![0.0; n_points];
        }

        let mut sorted_selected: Vec<usize> = selected_clusters.iter().copied().collect();
        sorted_selected.sort_unstable();
        let mut birth_lambda: HashMap<usize, f64> = HashMap::new();
        let mut max_lambda: HashMap<usize, f64> = HashMap::new();

        for edge in condensed_tree {
            if edge.child >= n_points && selected_clusters.contains(&edge.child) {
                birth_lambda
                    .entry(edge.child)
                    .and_modify(|value| {
                        if edge.lambda_val < *value {
                            *value = edge.lambda_val;
                        }
                    })
                    .or_insert(edge.lambda_val);
            }
        }
        if let Some(&root) = sorted_selected.first() {
            birth_lambda.entry(root).or_insert(0.0);
        }

        let mut cluster_parent: HashMap<usize, usize> = HashMap::new();
        for edge in condensed_tree {
            if edge.child >= n_points {
                cluster_parent.insert(edge.child, edge.parent);
            }
        }

        let mut effective_cluster: HashMap<usize, usize> = HashMap::new();
        for &cluster in selected_clusters {
            effective_cluster.insert(cluster, cluster);
        }

        let all_clusters: HashSet<usize> = condensed_tree
            .iter()
            .flat_map(|edge| {
                let mut clusters = vec![edge.parent];
                if edge.child >= n_points {
                    clusters.push(edge.child);
                }
                clusters
            })
            .collect();

        for &cluster in &all_clusters {
            if !effective_cluster.contains_key(&cluster) {
                let mut current = cluster;
                while let Some(&parent) = cluster_parent.get(&current) {
                    if let Some(&effective) = effective_cluster.get(&parent) {
                        effective_cluster.insert(cluster, effective);
                        break;
                    }
                    current = parent;
                }
            }
        }

        for edge in condensed_tree {
            if edge.child < n_points {
                if let Some(&effective) = effective_cluster.get(&edge.parent) {
                    let current_max = max_lambda.entry(effective).or_insert(0.0_f64);
                    if edge.lambda_val.is_finite() && edge.lambda_val > *current_max {
                        *current_max = edge.lambda_val;
                    }
                }
            }
        }

        let mut point_lambda: HashMap<usize, f64> = HashMap::new();
        let mut point_cluster_id: HashMap<usize, usize> = HashMap::new();
        for edge in condensed_tree {
            if edge.child < n_points {
                if let Some(&effective) = effective_cluster.get(&edge.parent) {
                    let current = point_lambda
                        .get(&edge.child)
                        .copied()
                        .unwrap_or(f64::NEG_INFINITY);
                    if edge.lambda_val >= current {
                        point_lambda.insert(edge.child, edge.lambda_val);
                        point_cluster_id.insert(edge.child, effective);
                    }
                }
            }
        }

        let mut probabilities = vec![0.0; n_points];
        for point in 0..n_points {
            if labels[point] < 0 {
                continue;
            }
            let cluster = sorted_selected[labels[point] as usize];
            let birth = *birth_lambda.get(&cluster).unwrap_or(&0.0);
            let max = *max_lambda.get(&cluster).unwrap_or(&0.0);
            let point_lambda = *point_lambda.get(&point).unwrap_or(&0.0);
            let range = max - birth;
            if range > 0.0 && range.is_finite() {
                probabilities[point] = ((point_lambda - birth) / range).clamp(0.0, 1.0);
            } else {
                probabilities[point] = 1.0;
            }
        }
        probabilities
    }

    fn compute_outlier_scores(condensed_tree: &[CondensedTreeEdge], n_points: usize) -> Vec<f64> {
        if condensed_tree.is_empty() {
            return vec![0.0; n_points];
        }

        let mut cluster_parent: HashMap<usize, usize> = HashMap::new();
        let mut max_lambda: HashMap<usize, f64> = HashMap::new();
        for edge in condensed_tree {
            if edge.child >= n_points {
                cluster_parent.insert(edge.child, edge.parent);
            }
        }

        let mut point_parent: HashMap<usize, usize> = HashMap::new();
        let mut point_lambda: HashMap<usize, f64> = HashMap::new();
        for edge in condensed_tree {
            if edge.child < n_points {
                let current = point_lambda
                    .get(&edge.child)
                    .copied()
                    .unwrap_or(f64::NEG_INFINITY);
                if edge.lambda_val >= current {
                    point_parent.insert(edge.child, edge.parent);
                    point_lambda.insert(edge.child, edge.lambda_val);
                }
            }
        }

        for edge in condensed_tree {
            if edge.child < n_points && edge.lambda_val.is_finite() {
                let current = max_lambda.entry(edge.parent).or_insert(0.0_f64);
                if edge.lambda_val > *current {
                    *current = edge.lambda_val;
                }
            }
        }

        let all_clusters: HashSet<usize> = condensed_tree
            .iter()
            .flat_map(|edge| {
                let mut clusters = vec![edge.parent];
                if edge.child >= n_points {
                    clusters.push(edge.child);
                }
                clusters
            })
            .collect();
        let mut sorted_clusters: Vec<usize> = all_clusters.iter().copied().collect();
        sorted_clusters.sort_unstable_by(|left, right| right.cmp(left));
        for &cluster in &sorted_clusters {
            if let Some(&parent) = cluster_parent.get(&cluster) {
                let child_max = *max_lambda.get(&cluster).unwrap_or(&0.0);
                let parent_max = max_lambda.entry(parent).or_insert(0.0_f64);
                if child_max > *parent_max {
                    *parent_max = child_max;
                }
            }
        }

        let mut scores = vec![0.0; n_points];
        for (point, score) in scores.iter_mut().enumerate() {
            if let Some(&parent) = point_parent.get(&point) {
                let point_lambda = *point_lambda.get(&point).unwrap_or(&0.0);
                let max = *max_lambda.get(&parent).unwrap_or(&0.0);
                if max > 0.0 && max.is_finite() {
                    *score = ((max - point_lambda) / max).clamp(0.0, 1.0);
                }
            }
        }
        scores
    }

    struct UnionFind {
        parent: Vec<usize>,
        size: Vec<usize>,
    }

    impl UnionFind {
        fn new(n: usize) -> Self {
            UnionFind {
                parent: (0..n).collect(),
                size: vec![1; n],
            }
        }

        fn find(&mut self, x: usize) -> usize {
            let mut root = x;
            while self.parent[root] != root {
                root = self.parent[root];
            }
            let mut current = x;
            while self.parent[current] != root {
                let next = self.parent[current];
                self.parent[current] = root;
                current = next;
            }
            root
        }

        fn union(&mut self, x: usize, y: usize) {
            let root_x = self.find(x);
            let root_y = self.find(y);
            if root_x == root_y {
                return;
            }
            if self.size[root_x] < self.size[root_y] {
                self.parent[root_x] = root_y;
                self.size[root_y] += self.size[root_x];
            } else {
                self.parent[root_y] = root_x;
                self.size[root_x] += self.size[root_y];
            }
        }
    }
}
