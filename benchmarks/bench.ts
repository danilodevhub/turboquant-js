import {
  TurboQuantProd,
  VectorIndex,
  createPRNG,
} from '../src/index.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function randomUnitVector(d: number, rng: ReturnType<typeof createPRNG>): Float64Array {
  const v = new Float64Array(d);
  for (let i = 0; i < d; i++) v[i] = rng.nextGaussian();
  let norm = 0;
  for (let i = 0; i < d; i++) norm += v[i]! * v[i]!;
  norm = Math.sqrt(norm);
  if (norm > 0) for (let i = 0; i < d; i++) v[i] /= norm;
  return v;
}

function dot(a: Float64Array, b: Float64Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i]! * b[i]!;
  return s;
}

function pad(s: string, len: number): string {
  return s.length >= len ? s : s + ' '.repeat(len - s.length);
}

function padLeft(s: string, len: number): string {
  return s.length >= len ? s : ' '.repeat(len - s.length) + s;
}

// ---------------------------------------------------------------------------
// 1. Compression quality table
// ---------------------------------------------------------------------------

function compressionQuality(): void {
  console.log('\n## Compression Quality\n');
  console.log(
    '| Dim | Bits | Avg MSE      | Avg IP Bias  | Compression Ratio |',
  );
  console.log(
    '|-----|------|--------------|--------------|-------------------|',
  );

  const dims = [64, 128, 384];
  const bitsList = [2, 3, 4];
  const nVectors = 200;
  const nPairs = 100;

  for (const d of dims) {
    for (const bits of bitsList) {
      const tq = new TurboQuantProd({ dimension: d, bits, seed: 42 });
      const rng = createPRNG(123);

      // Generate vectors
      const vectors: Float64Array[] = [];
      for (let i = 0; i < nVectors; i++) {
        vectors.push(randomUnitVector(d, rng));
      }

      // Average MSE distortion
      let totalMSE = 0;
      for (const v of vectors) {
        const q = tq.quantize(v);
        const vHat = tq.dequantize(q);
        let mse = 0;
        for (let j = 0; j < d; j++) mse += (v[j]! - vHat[j]!) ** 2;
        totalMSE += mse / d;
      }
      const avgMSE = totalMSE / nVectors;

      // Average inner product bias
      let totalBias = 0;
      for (let i = 0; i < nPairs; i++) {
        const x = vectors[i]!;
        const y = vectors[i + nPairs]!;
        const q = tq.quantize(x);
        const estimated = tq.innerProduct(y, q);
        const exact = dot(y, x);
        totalBias += estimated - exact;
      }
      const avgBias = totalBias / nPairs;

      // Compression ratio
      const index = new VectorIndex({ dimension: d, bits, seed: 42 });
      index.add(0, vectors[0]!);
      const ratio = index.memoryUsage.compressionRatio;

      console.log(
        `| ${padLeft(String(d), 3)} | ${padLeft(String(bits), 4)} | ${padLeft(avgMSE.toExponential(4), 12)} | ${padLeft(avgBias.toFixed(6), 12)} | ${padLeft(ratio.toFixed(2), 17)} |`,
      );
    }
  }
}

// ---------------------------------------------------------------------------
// 2. Recall@10 benchmark
// ---------------------------------------------------------------------------

function recallBenchmark(): void {
  console.log('\n## Recall@10 (dim=384)\n');
  console.log('| Bits | Recall@10 |');
  console.log('|------|-----------|');

  const d = 384;
  const nVectors = 500;
  const nQueries = 20;
  const k = 10;
  const bitsList = [2, 3, 4];

  const rng = createPRNG(777);
  const vectors: Float64Array[] = [];
  for (let i = 0; i < nVectors; i++) {
    vectors.push(randomUnitVector(d, rng));
  }
  const queries: Float64Array[] = [];
  for (let i = 0; i < nQueries; i++) {
    queries.push(randomUnitVector(d, rng));
  }

  // Compute exact top-k for each query
  const exactTopK: number[][] = [];
  for (const q of queries) {
    const scores = vectors.map((v, i) => ({ i, s: dot(q, v) }));
    scores.sort((a, b) => b.s - a.s);
    exactTopK.push(scores.slice(0, k).map((x) => x.i));
  }

  for (const bits of bitsList) {
    const index = new VectorIndex({ dimension: d, bits, seed: 42 });
    for (let i = 0; i < nVectors; i++) {
      index.add(i, vectors[i]!);
    }

    let totalRecall = 0;
    for (let qi = 0; qi < nQueries; qi++) {
      const results = index.search(queries[qi]!, k);
      const resultIds = new Set(results.map((r) => r.id));
      let hits = 0;
      for (const id of exactTopK[qi]!) {
        if (resultIds.has(id)) hits++;
      }
      totalRecall += hits / k;
    }
    const recall = totalRecall / nQueries;
    console.log(`| ${padLeft(String(bits), 4)} | ${padLeft((recall * 100).toFixed(1) + '%', 9)} |`);
  }
}

// ---------------------------------------------------------------------------
// 3. Performance timing
// ---------------------------------------------------------------------------

function performanceTiming(): void {
  console.log('\n## Performance Timing (dim=384, bits=3)\n');

  const d = 384;
  const bits = 3;
  const nVectors = 1000;

  const tq = new TurboQuantProd({ dimension: d, bits, seed: 42 });
  const rng = createPRNG(999);

  const vectors: Float64Array[] = [];
  for (let i = 0; i < nVectors; i++) {
    vectors.push(randomUnitVector(d, rng));
  }

  // Quantize timing
  const t0 = performance.now();
  const quantized = vectors.map((v) => tq.quantize(v));
  const t1 = performance.now();
  const quantizeMs = t1 - t0;

  console.log(`| Operation          | Total (ms) | Per-vector (us) |`);
  console.log(`|--------------------|------------|-----------------|`);
  console.log(
    `| Quantize (${nVectors})    | ${padLeft(quantizeMs.toFixed(1), 10)} | ${padLeft(((quantizeMs / nVectors) * 1000).toFixed(1), 15)} |`,
  );

  // Search timing (brute-force scan over all quantized vectors)
  const index = new VectorIndex({ dimension: d, bits, seed: 42 });
  for (let i = 0; i < nVectors; i++) {
    index.add(i, vectors[i]!);
  }

  const nQueries = 100;
  const queryVecs: Float64Array[] = [];
  for (let i = 0; i < nQueries; i++) {
    queryVecs.push(randomUnitVector(d, rng));
  }

  const t2 = performance.now();
  for (const q of queryVecs) {
    index.search(q, 10);
  }
  const t3 = performance.now();
  const searchMs = t3 - t2;

  console.log(
    `| Search top-10 (${nQueries}) | ${padLeft(searchMs.toFixed(1), 10)} | ${padLeft(((searchMs / nQueries) * 1000).toFixed(1), 15)} |`,
  );
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

console.log('# TurboQuant-JS Benchmarks\n');

compressionQuality();
recallBenchmark();
performanceTiming();

console.log('\nDone.');
