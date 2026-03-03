use ndarray::ArrayView2;

/// A simple KD-tree for nearest neighbor queries with runtime dimensionality.
///
/// This is optimized for the HDBSCAN use case: build once, query k-NN many times.
/// Uses the sliding midpoint rule for balanced construction.
pub struct KdTree {
    nodes: Vec<KdNode>,
    data: Vec<Vec<f64>>,
}

struct KdNode {
    /// Index into the original data array
    point_idx: usize,
    /// Split dimension
    split_dim: usize,
    /// Split value
    split_val: f64,
    /// Left child index (or usize::MAX if none)
    left: usize,
    /// Right child index (or usize::MAX if none)
    right: usize,
}

const NO_CHILD: usize = usize::MAX;

impl KdTree {
    /// Build a KD-tree from an n×d data matrix.
    pub fn build(data: &ArrayView2<f64>) -> Self {
        let n = data.nrows();
        let dim = data.ncols();

        let points: Vec<Vec<f64>> = (0..n).map(|i| data.row(i).to_vec()).collect();
        let mut indices: Vec<usize> = (0..n).collect();
        let mut nodes = Vec::with_capacity(n);

        if n > 0 {
            Self::build_recursive(&points, &mut indices, 0, n, 0, dim, &mut nodes);
        }

        KdTree {
            nodes,
            data: points,
        }
    }

    fn build_recursive(
        points: &[Vec<f64>],
        indices: &mut [usize],
        start: usize,
        end: usize,
        depth: usize,
        dim: usize,
        nodes: &mut Vec<KdNode>,
    ) -> usize {
        if start >= end {
            return NO_CHILD;
        }

        let split_dim = depth % dim;

        // Find the dimension with most spread for better balance
        let split_dim = {
            let mut best_dim = split_dim;
            let mut best_spread = f64::NEG_INFINITY;
            for d in 0..dim {
                let mut min_val = f64::INFINITY;
                let mut max_val = f64::NEG_INFINITY;
                for &idx in &indices[start..end] {
                    let v = points[idx][d];
                    if v < min_val {
                        min_val = v;
                    }
                    if v > max_val {
                        max_val = v;
                    }
                }
                let spread = max_val - min_val;
                if spread > best_spread {
                    best_spread = spread;
                    best_dim = d;
                }
            }
            best_dim
        };

        // Partition around median
        let mid = start + (end - start) / 2;
        indices[start..end].select_nth_unstable_by(mid - start, |&a, &b| {
            points[a][split_dim]
                .partial_cmp(&points[b][split_dim])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let point_idx = indices[mid];
        let split_val = points[point_idx][split_dim];

        let node_idx = nodes.len();
        nodes.push(KdNode {
            point_idx,
            split_dim,
            split_val,
            left: NO_CHILD,
            right: NO_CHILD,
        });

        let left = Self::build_recursive(points, indices, start, mid, depth + 1, dim, nodes);
        let right = Self::build_recursive(points, indices, mid + 1, end, depth + 1, dim, nodes);

        nodes[node_idx].left = left;
        nodes[node_idx].right = right;

        node_idx
    }

    /// Find the k nearest neighbors of a query point.
    /// Returns pairs of (squared_distance, point_index), sorted by distance.
    pub fn query_knn(&self, query: &[f64], k: usize) -> Vec<(f64, usize)> {
        if self.nodes.is_empty() || k == 0 {
            return vec![];
        }

        let mut heap = BoundedMaxHeap::new(k);
        self.knn_recursive(0, query, &mut heap);

        let mut result: Vec<(f64, usize)> = heap
            .items
            .into_iter()
            .map(|(dist_sq, idx)| (dist_sq.sqrt(), idx))
            .collect();
        result.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        result
    }

    fn knn_recursive(&self, node_idx: usize, query: &[f64], heap: &mut BoundedMaxHeap) {
        if node_idx == NO_CHILD {
            return;
        }

        let node = &self.nodes[node_idx];
        let point = &self.data[node.point_idx];

        // Compute squared distance to this point
        let dist_sq: f64 = query
            .iter()
            .zip(point.iter())
            .map(|(&a, &b)| {
                let d = a - b;
                d * d
            })
            .sum();

        heap.push(dist_sq, node.point_idx);

        // Decide which child to visit first
        let diff = query[node.split_dim] - node.split_val;
        let (first, second) = if diff <= 0.0 {
            (node.left, node.right)
        } else {
            (node.right, node.left)
        };

        self.knn_recursive(first, query, heap);

        // Only visit the other subtree if the splitting plane is closer than the worst neighbor
        let plane_dist_sq = diff * diff;
        if heap.len() < heap.capacity || plane_dist_sq < heap.max_dist() {
            self.knn_recursive(second, query, heap);
        }
    }

    /// Query k nearest neighbors for point at index `idx` in the original data,
    /// excluding the point itself.
    pub fn query_knn_exclude_self(&self, idx: usize, k: usize) -> Vec<(f64, usize)> {
        if self.nodes.is_empty() || k == 0 {
            return vec![];
        }

        let query = &self.data[idx];
        let mut heap = BoundedMaxHeap::new(k + 1);
        self.knn_recursive(0, query, &mut heap);

        let mut result: Vec<(f64, usize)> = heap
            .items
            .into_iter()
            .filter(|&(_, pidx)| pidx != idx)
            .map(|(dist_sq, pidx)| (dist_sq.sqrt(), pidx))
            .collect();
        result.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        result.truncate(k);
        result
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }
}

/// Max-heap bounded to k items, keeping the k smallest distances.
struct BoundedMaxHeap {
    items: Vec<(f64, usize)>, // (squared_distance, index)
    capacity: usize,
}

impl BoundedMaxHeap {
    fn new(capacity: usize) -> Self {
        BoundedMaxHeap {
            items: Vec::with_capacity(capacity + 1),
            capacity,
        }
    }

    fn push(&mut self, dist_sq: f64, idx: usize) {
        if self.items.len() < self.capacity {
            self.items.push((dist_sq, idx));
            if self.items.len() == self.capacity {
                // Heapify to find max quickly
                self.items
                    .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            }
        } else if dist_sq < self.items[0].0 {
            self.items[0] = (dist_sq, idx);
            // Re-sort to maintain max at front
            self.items
                .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        }
    }

    fn max_dist(&self) -> f64 {
        self.items.first().map_or(f64::INFINITY, |&(d, _)| d)
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_knn_basic() {
        let data = array![
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [10.0, 10.0],
        ];
        let tree = KdTree::build(&data.view());

        // Nearest neighbor of [0,0] should be [1,0]
        let nn = tree.query_knn(&[0.0, 0.0], 1);
        assert_eq!(nn.len(), 1);
        assert_eq!(nn[0].1, 0); // itself
        assert!((nn[0].0).abs() < 1e-12);

        // 2-NN of [0,0] = itself and [1,0]
        let nn = tree.query_knn(&[0.0, 0.0], 2);
        assert_eq!(nn.len(), 2);
        assert_eq!(nn[0].1, 0);
        assert_eq!(nn[1].1, 1);
    }

    #[test]
    fn test_knn_exclude_self() {
        let data = array![
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ];
        let tree = KdTree::build(&data.view());

        let nn = tree.query_knn_exclude_self(0, 1);
        assert_eq!(nn.len(), 1);
        assert_eq!(nn[0].1, 1); // [1,0] is nearest to [0,0]
        assert!((nn[0].0 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_knn_larger() {
        // 100 random-ish points
        let n = 100;
        let mut data_vec = Vec::new();
        for i in 0..n {
            data_vec.push(vec![(i as f64) * 0.1, ((i * 7) % 13) as f64]);
        }
        let mut arr = ndarray::Array2::zeros((n, 2));
        for (i, row) in data_vec.iter().enumerate() {
            arr[[i, 0]] = row[0];
            arr[[i, 1]] = row[1];
        }
        let tree = KdTree::build(&arr.view());

        // Verify KNN by brute force for point 0
        let query = arr.row(0).to_vec();
        let knn = tree.query_knn(&query, 5);

        // Brute force
        let mut brute: Vec<(f64, usize)> = (0..n)
            .map(|j| {
                let d: f64 = (0..2)
                    .map(|d| {
                        let diff = query[d] - arr[[j, d]];
                        diff * diff
                    })
                    .sum::<f64>()
                    .sqrt();
                (d, j)
            })
            .collect();
        brute.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        for i in 0..5 {
            assert_eq!(knn[i].1, brute[i].1, "Mismatch at position {}", i);
            assert!((knn[i].0 - brute[i].0).abs() < 1e-10);
        }
    }
}
