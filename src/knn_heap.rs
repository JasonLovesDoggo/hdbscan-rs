//! Shared bounded max-heap for k-nearest-neighbor queries.
//!
//! Used by ball_tree, kdtree, and kdtree_bounded to avoid duplicating
//! the heap implementation.

/// Max-heap bounded to k items, keeping the k smallest squared distances.
/// Uses a binary max-heap for O(log k) push instead of O(k log k) sort.
pub struct KnnHeap {
    items: Vec<(f64, usize)>,
    capacity: usize,
}

impl KnnHeap {
    pub fn new(capacity: usize) -> Self {
        KnnHeap {
            items: Vec::with_capacity(capacity),
            capacity,
        }
    }

    /// Reset for reuse without deallocating.
    #[inline]
    pub fn clear(&mut self) {
        self.items.clear();
    }

    #[inline]
    pub fn push(&mut self, dist_sq: f64, idx: usize) {
        if self.items.len() < self.capacity {
            self.items.push((dist_sq, idx));
            self.sift_up(self.items.len() - 1);
        } else if dist_sq < self.items[0].0 {
            self.items[0] = (dist_sq, idx);
            self.sift_down(0);
        }
    }

    #[inline]
    pub fn is_full(&self) -> bool {
        self.items.len() >= self.capacity
    }

    #[inline]
    pub fn max_dist_sq(&self) -> f64 {
        self.items.first().map_or(f64::INFINITY, |&(d, _)| d)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Return the index of the nearest neighbor that is not self_idx.
    #[inline]
    pub fn nearest_non_self(&self, self_idx: usize) -> usize {
        let mut best_dist = f64::INFINITY;
        let mut best_idx = 0;
        for &(d, idx) in &self.items {
            if idx != self_idx && d > 0.0 && d < best_dist {
                best_dist = d;
                best_idx = idx;
            }
        }
        if best_dist == f64::INFINITY && self.items.len() > 1 {
            // Fallback: return any non-self index
            for &(_, idx) in &self.items {
                if idx != self_idx {
                    return idx;
                }
            }
        }
        best_idx
    }

    /// Return items sorted by distance (squared), converting to actual distance.
    pub fn into_sorted_distances(mut self) -> Vec<(f64, usize)> {
        self.items.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        self.items
            .into_iter()
            .map(|(dist_sq, idx)| (dist_sq.sqrt(), idx))
            .collect()
    }

    /// Return items sorted by squared distance (no sqrt).
    pub fn into_sorted_sq(mut self) -> Vec<(f64, usize)> {
        self.items.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        self.items
    }

    /// Write all neighbor indices (excluding self_idx) into the output slice.
    /// Returns the number of neighbors written.
    #[inline]
    pub fn all_neighbors(&self, self_idx: usize, out: &mut [usize]) -> usize {
        let mut count = 0;
        for &(_, idx) in &self.items {
            if idx != self_idx && count < out.len() {
                out[count] = idx;
                count += 1;
            }
        }
        count
    }

    fn sift_up(&mut self, mut idx: usize) {
        while idx > 0 {
            let parent = (idx - 1) / 2;
            if self.items[idx].0 > self.items[parent].0 {
                self.items.swap(idx, parent);
                idx = parent;
            } else {
                break;
            }
        }
    }

    fn sift_down(&mut self, mut idx: usize) {
        let len = self.items.len();
        loop {
            let left = 2 * idx + 1;
            let right = 2 * idx + 2;
            let mut largest = idx;
            if left < len && self.items[left].0 > self.items[largest].0 {
                largest = left;
            }
            if right < len && self.items[right].0 > self.items[largest].0 {
                largest = right;
            }
            if largest != idx {
                self.items.swap(idx, largest);
                idx = largest;
            } else {
                break;
            }
        }
    }
}
