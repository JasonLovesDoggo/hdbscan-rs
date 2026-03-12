import { HDBSCAN } from "./hdbscan_rs.js";

// Simple seeded PRNG (mulberry32)
function mulberry32(seed) {
  return function () {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function makeBlobs(n, dim, nCenters, seed) {
  const rng = mulberry32(seed);
  const data = new Float64Array(n * dim);
  // Generate centers
  const centers = [];
  for (let c = 0; c < nCenters; c++) {
    const center = [];
    for (let d = 0; d < dim; d++) center.push((rng() - 0.5) * 100);
    centers.push(center);
  }
  // Generate points around centers
  for (let i = 0; i < n; i++) {
    const c = centers[i % nCenters];
    for (let d = 0; d < dim; d++) {
      // Box-Muller for gaussian noise
      const u1 = rng(),
        u2 = rng();
      const noise = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      data[i * dim + d] = c[d] + noise;
    }
  }
  return data;
}

function bench(label, n, dim, nCenters, mcs, warmup = 2, iters = 5) {
  const data = makeBlobs(n, dim, nCenters, 42);

  // Warmup
  for (let i = 0; i < warmup; i++) {
    const h = new HDBSCAN(mcs);
    h.fit_predict(data, n, dim);
    h.free();
  }

  // Timed runs
  const times = [];
  for (let i = 0; i < iters; i++) {
    const h = new HDBSCAN(mcs);
    const t0 = performance.now();
    const labels = h.fit_predict(data, n, dim);
    const t1 = performance.now();
    times.push(t1 - t0);

    // Count clusters on last iter
    if (i === iters - 1) {
      const unique = new Set(labels);
      const nClusters = [...unique].filter((l) => l >= 0).length;
      const noise = labels.filter((l) => l === -1).length;
      console.log(
        `  ${label}: ${median(times).toFixed(1)}ms (median of ${iters}), ${nClusters} clusters, ${noise} noise`,
      );
    }
    h.free();
  }
  return median(times);
}

function median(arr) {
  const sorted = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

console.log("=== HDBSCAN-RS WASM Benchmark ===\n");

const configs = [
  ["5Kx10D", 5000, 10, 5, 10],
  ["5Kx50D", 5000, 50, 5, 10],
  ["50Kx2D", 50000, 2, 5, 10],
  ["1Kx256D", 1000, 256, 3, 5],
  ["10Kx10D", 10000, 10, 5, 10],
];

for (const [label, n, dim, nc, mcs] of configs) {
  bench(label, n, dim, nc, mcs);
}
