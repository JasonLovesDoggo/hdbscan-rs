use hdbscan_rs::{Hdbscan, HdbscanParams};
use ndarray::Array2;
use std::env;
use std::fs;
use std::time::Instant;

/// Read peak RSS from /proc/self/status (Linux).
fn peak_rss_kb() -> Option<u64> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if line.starts_with("VmHWM:") {
            let val = line.split_whitespace().nth(1)?;
            return val.parse().ok();
        }
    }
    None
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: bench_runner <data.csv> <min_cluster_size> [n_runs]");
        std::process::exit(1);
    }

    let data_path = &args[1];
    let mcs: usize = args[2].parse().unwrap();
    let n_runs: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(3);

    // Read CSV
    let content = fs::read_to_string(data_path).unwrap();
    let rows: Vec<Vec<f64>> = content
        .lines()
        .filter(|l| !l.is_empty())
        .map(|line| {
            line.split(',')
                .map(|s| s.trim().parse::<f64>().unwrap())
                .collect()
        })
        .collect();

    let n = rows.len();
    let d = rows[0].len();
    let mut data = Array2::zeros((n, d));
    for (i, row) in rows.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            data[[i, j]] = val;
        }
    }

    let mut best_ms = f64::INFINITY;
    let mut labels = vec![];

    for _ in 0..n_runs {
        let params = HdbscanParams {
            min_cluster_size: mcs,
            ..Default::default()
        };
        let mut hdbscan = Hdbscan::new(params);

        let start = Instant::now();
        labels = hdbscan.fit_predict(&data.view()).unwrap();
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        if elapsed < best_ms {
            best_ms = elapsed;
        }
    }

    println!("elapsed_ms:{:.2}", best_ms);
    println!("peak_rss_kb:{}", peak_rss_kb().unwrap_or(0));
    let label_str: Vec<String> = labels.iter().map(|l| l.to_string()).collect();
    println!("labels:{}", label_str.join(","));
}
