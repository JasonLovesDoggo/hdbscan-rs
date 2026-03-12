use std::time::Instant;

fn main() {
    let n = 5000;
    let dim = 50;
    let data = hdbscan_rs::bench_utils::make_blobs(n, dim, 5, 42);
    let k = 10usize; // min_samples

    // Warmup
    {
        use hdbscan_rs::{Hdbscan, HdbscanParams};
        let mut h = Hdbscan::new(HdbscanParams { min_cluster_size: 10, ..Default::default() });
        let _ = h.fit_predict(&data.view());
    }

    // Phase 1: GEMM
    let t0 = Instant::now();
    let gram = data.view().dot(&data.view().t());
    let t_gemm = t0.elapsed();

    let gram_slice = gram.as_slice().unwrap();
    let norms_sq: Vec<f64> = (0..n).map(|i| gram_slice[i * n + i]).collect();

    // Phase 2: Core extraction
    let t1 = Instant::now();
    let heap_k = k - 1;
    let mut core_dists_sq = vec![0.0f64; n];
    let mut heap = hdbscan_rs::knn_heap::KnnHeap::new(heap_k);
    for i in 0..n {
        heap.clear();
        let ni = norms_sq[i];
        let row_off = i * n;
        for j in 0..n {
            if i == j { continue; }
            let d_sq = unsafe {
                (ni + *norms_sq.get_unchecked(j) - 2.0 * *gram_slice.get_unchecked(row_off + j)).max(0.0)
            };
            heap.push(d_sq, j);
        }
        core_dists_sq[i] = heap.max_dist_sq();
    }
    let t_core = t1.elapsed();

    let gram_vec = gram.into_raw_vec_and_offset().0;

    // Phase 3: Prim's
    let t2 = Instant::now();
    let mut min_weight_sq = vec![f64::INFINITY; n];
    let mut nearest = vec![0usize; n];
    let mut edges = Vec::with_capacity(n - 1);
    let mut active: Vec<usize> = (1..n).collect();

    let core_0_sq = core_dists_sq[0];
    let n0 = norms_sq[0];
    for &j in &active {
        let d_sq = (n0 + norms_sq[j] - 2.0 * gram_vec[j]).max(0.0);
        let mr_sq = f64::max(f64::max(core_0_sq, core_dists_sq[j]), d_sq);
        min_weight_sq[j] = mr_sq;
    }

    for _ in 0..(n - 1) {
        if active.is_empty() { break; }
        let mut best_pos = 0;
        let mut best_sq = min_weight_sq[active[0]];
        let mut best_idx = active[0];
        for (pos, &j) in active.iter().enumerate().skip(1) {
            let w = min_weight_sq[j];
            if w < best_sq || (w == best_sq && j < best_idx) {
                best_sq = w;
                best_pos = pos;
                best_idx = j;
            }
        }
        let min_idx = best_idx;
        edges.push((nearest[min_idx], min_idx, best_sq.sqrt()));
        active.swap_remove(best_pos);

        if active.len() > 64 && active.len() % 128 == 0 {
            active.sort_unstable();
        }

        let core_i_sq = core_dists_sq[min_idx];
        let ni = norms_sq[min_idx];
        let row_offset = min_idx * n;
        for &j in &active {
            let mw = min_weight_sq[j];
            if core_i_sq >= mw { continue; }
            if core_dists_sq[j] >= mw { continue; }
            let d_sq = (ni + norms_sq[j] - 2.0 * gram_vec[row_offset + j]).max(0.0);
            if d_sq >= mw { continue; }
            let mr_sq = f64::max(f64::max(core_i_sq, core_dists_sq[j]), d_sq);
            if mr_sq < mw {
                min_weight_sq[j] = mr_sq;
                nearest[j] = min_idx;
            }
        }
    }
    let t_prim = t2.elapsed();

    println!("5000x50D phase breakdown:");
    println!("  GEMM:       {:>6.1}ms", t_gemm.as_secs_f64() * 1000.0);
    println!("  Core:       {:>6.1}ms", t_core.as_secs_f64() * 1000.0);
    println!("  Prim's:     {:>6.1}ms", t_prim.as_secs_f64() * 1000.0);
    println!("  Total:      {:>6.1}ms", (t_gemm + t_core + t_prim).as_secs_f64() * 1000.0);
}
