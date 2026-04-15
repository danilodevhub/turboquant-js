<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./banner-light.svg" width="480" />
    <source media="(prefers-color-scheme: light)" srcset="./banner-dark.svg" width="480" />
    <img alt="turboquant-js" src="./banner-dark.svg" width="480" />
  </picture>
</p>

<p align="center">
  <a href="https://github.com/danilodevhub/turboquant-js/actions/workflows/ci.yml"><img src="https://github.com/danilodevhub/turboquant-js/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <a href="https://www.npmjs.com/package/turboquant-js"><img src="https://img.shields.io/npm/v/turboquant-js" alt="npm version" /></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT" /></a>
</p>
[![npm version](https://img.shields.io/npm/v/turboquant-js)](https://www.npmjs.com/package/turboquant-js)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

TypeScript implementation of Google's [TurboQuant](https://arxiv.org/abs/2504.19874) algorithm for near-optimal vector quantization.

> Based on the research paper: *TurboQuant: Online Vector Quantization* (ICLR 2026) by Zandieh, Daliri, Hadian, Mirrokni.

## Features

- **Near-optimal compression** — achieves distortion within ~2.7x of information-theoretic lower bounds
- **Zero training required** — data-oblivious quantization with no fine-tuning
- **Unbiased inner products** — two-stage quantization (MSE + QJL) guarantees unbiased dot product estimates
- **Universal runtime** — works in Node.js and browsers, zero runtime dependencies
- **Two application APIs** — `VectorIndex` for nearest-neighbor search, `KVCacheCompressor` for LLM KV cache compression

## Use Cases

### Client-Side Search & Privacy

- **Semantic search in browser extensions** — Index bookmarks, history, or notes with embeddings and search them locally. Data never leaves the device.
- **Offline-capable RAG** — Build retrieval-augmented generation apps that work entirely client-side with a local LLM (WebLLM, Transformers.js).
- **In-browser document search** — Upload and search PDFs or docs by meaning, not just keywords. See the [live demo](https://danilodevhub.github.io/turboquant-js-examples/).
- **E-commerce product discovery** — Embed product catalogs and let users search by natural language directly in the browser.

### Edge & Serverless

- **Cloudflare Workers / Vercel Edge Functions** — Pure JS runs where WASM can't. No binary dependencies, tiny bundle (~15 KB).
- **IoT / embedded Node.js** — Vector search on resource-constrained devices that can't afford full float64 storage.

### LLM Infrastructure

- **KV cache compression** — The `KVCacheCompressor` API compresses attention keys and values for in-browser LLM inference. Unbiased attention scores preserve model quality while cutting memory by 20x.
- **Agent memory** — AI agents that accumulate context over time can compress past embeddings to stay within memory limits.

### Developer Tools

- **Rapid prototyping** — Zero config, `npm install`, 3 lines of code. No Docker, no WASM compilation, no training data.
- **Testing & CI** — Deterministic vector search (seeded PRNG) for reproducible test suites without spinning up a vector database.

## Why turboquant-js?

| | turboquant-js | FAISS-WASM | Pinecone / Qdrant |
|---|---|---|---|
| Runtime | Pure TypeScript | WASM binary | Server-side |
| Setup | `npm install` | Complex build | API key + infra |
| Training required | None | PQ needs training data | N/A |
| Unbiased scores | Yes (provably) | No | N/A |
| Bundle size | ~15 KB | ~2 MB | N/A |
| Serialization | Built-in `toBuffer` / `fromBuffer` | Custom | Managed |
| Privacy | All data stays client-side | Client-side | Data sent to server |

## Installation

```bash
npm install turboquant-js
```

## Quick Start

### Vector Search Index

```ts
import { VectorIndex } from 'turboquant-js';

const index = new VectorIndex({ dimension: 384, bits: 3, metric: 'cosine' });

// Add vectors (accepts Float64Array or number[])
index.add('doc1', embedding1);
index.add('doc2', embedding2);

// Batch add
index.addBatch([
  { id: 'doc3', vector: embedding3 },
  { id: 'doc4', vector: embedding4 },
]);

// Search for top-k nearest neighbors
const results = index.search(queryEmbedding, 10);
// => [{ id: 'doc2', score: 0.93 }, { id: 'doc1', score: 0.87 }, ...]

// Check compression stats
console.log(index.memoryUsage);
// => { totalBits: 23200, bitsPerVector: 1184, compressionRatio: 20.8, actualBytes: ... }

// Serialize / deserialize
const buf = index.toBuffer();
const restored = VectorIndex.fromBuffer(buf, { dimension: 384 });
```

### KV Cache Compression

```ts
import { KVCacheCompressor } from 'turboquant-js';

const compressor = new KVCacheCompressor({
  keyDim: 128,
  valueDim: 128,
  keyBits: 3,   // unbiased attention scores
  valueBits: 2, // low-MSE value reconstruction
});

// Compress key-value pairs from transformer layers
compressor.append(keys, values);

// Compute attention scores against a query
const scores = compressor.attentionScores(queryVector);

// Retrieve (dequantize) values at specific positions
const vals = compressor.retrieveValues([0, 3, 7]);

// Serialize / deserialize
const buf = compressor.toBuffer();
const restored = KVCacheCompressor.fromBuffer(buf, { keyDim: 128, valueDim: 128 });
```

### Low-Level Quantizers

```ts
import { TurboQuantMSE, TurboQuantProd } from 'turboquant-js';

// Stage 1 only: MSE-optimal scalar quantization
const mse = new TurboQuantMSE({ dimension: 256, bits: 3 });
const quantized = mse.quantize(vector);
const reconstructed = mse.dequantize(quantized);

// Two-stage: MSE + QJL for unbiased inner products
const prod = new TurboQuantProd({ dimension: 256, bits: 3 });
const compressed = prod.quantize(vector);
const estimate = prod.innerProduct(queryVector, compressed);
// E[estimate] = <queryVector, vector>  (unbiased)
```

## API Reference

### `VectorIndex`

| Method / Property | Description |
|---|---|
| `add(id, vector)` | Add or replace a vector |
| `addBatch(items)` | Add multiple vectors at once |
| `search(query, k)` | Return top-k results as `{ id, score }[]` |
| `remove(id)` | Remove a vector by id |
| `size` | Number of stored vectors |
| `memoryUsage` | Compression statistics (includes `actualBytes`) |
| `toBuffer()` | Serialize the index to a compact binary `ArrayBuffer` |
| `VectorIndex.fromBuffer(buf, opts)` | Deserialize an index from a buffer |

**Options:** `dimension`, `bits` (default 3), `seed` (default 42), `metric` (`'cosine'` | `'ip'`)

**`memoryUsage` fields:** `totalBits`, `bitsPerVector`, `compressionRatio`, `actualBytes` (estimated in-memory byte footprint of stored entries).

### `KVCacheCompressor`

| Method / Property | Description |
|---|---|
| `append(keys, values)` | Compress and store KV pairs |
| `attentionScores(query)` | Unbiased attention score estimates |
| `retrieveValues(indices)` | Dequantize values at given positions |
| `length` | Number of cached pairs |
| `memoryUsage` | Compression statistics (includes `actualBytes`) |
| `clear()` | Remove all cached entries |
| `toBuffer()` | Serialize the cache to a compact binary `ArrayBuffer` |
| `KVCacheCompressor.fromBuffer(buf, opts)` | Deserialize a cache from a buffer |

**Options:** `keyDim`, `valueDim`, `keyBits` (default 3), `valueBits` (default 2), `seed` (default 42)

**`memoryUsage` fields:** `keyBitsPerVector`, `valueBitsPerVector`, `totalBytes`, `compressionRatio`, `actualBytes`.

### `TurboQuantMSE`

MSE-optimal scalar quantizer. Randomly rotates input then applies per-coordinate quantization using an optimal codebook derived from the coordinate distribution.

### `TurboQuantProd`

Two-stage quantizer (MSE + QJL) for unbiased inner product estimation. Uses `(b-1)` bits for MSE and 1 bit for QJL sign-based correction.

## How It Works

1. **Random rotation (Randomized Hadamard Transform)** — The input vector is rotated using 3 rounds of Walsh-Hadamard transforms combined with random sign flips (diagonal +/-1 matrices). The input is zero-padded to the next power of 2 before applying the transform, then truncated back to the original dimension. This runs in O(d log d) time and O(d) space, spreading vector energy uniformly across all coordinates so that per-coordinate scalar quantization is near-optimal.

2. **Lloyd-Max quantization** — Each rotated coordinate is scalar-quantized using an optimal codebook computed via the Lloyd-Max algorithm. The coordinate distribution after rotating a unit vector in R^d follows a Beta distribution; for d >= 64 the implementation switches to a Gaussian N(0, 1/d) approximation for efficiency. Centroids and decision boundaries are iteratively refined using adaptive Simpson quadrature for the conditional expectation integrals.

3. **QJL correction** (Prod quantizer only) — The quantization residual (x - x_hat) is projected through a dense d x d matrix with i.i.d. N(0,1) entries, then sign-bit quantized to 1 bit per coordinate. This Quantized Johnson-Lindenstrauss (QJL) sketch enables an unbiased correction term for inner product estimation. The correction formula is: sqrt(pi/2) / d * ||residual|| * signs^T * S * y. Note: the QJL projection matrix is O(d^2) in both storage and application time, which dominates the cost for large dimensions.

## Benchmarks

Results from `npm run bench` on random unit vectors:

### Compression quality (MSE and inner product bias)

| Dim | Bits | Avg MSE | IP Bias | Compression |
|-----|------|---------|---------|-------------|
| 64 | 2 | 5.5e-3 | ~0 | 25.6x |
| 64 | 3 | 1.8e-3 | ~0 | 18.3x |
| 64 | 4 | 5.2e-4 | ~0 | 14.2x |
| 128 | 2 | 2.8e-3 | ~0 | 28.4x |
| 128 | 3 | 9.1e-4 | ~0 | 19.7x |
| 128 | 4 | 2.7e-4 | ~0 | 15.1x |
| 384 | 2 | 1.2e-3 | ~0 | 30.7x |
| 384 | 3 | 7.4e-4 | ~0 | 20.8x |
| 384 | 4 | 6.0e-4 | ~0 | 15.7x |

### Recall@10 (dim=384, 500 vectors)

| Bits | Recall@10 |
|------|-----------|
| 2 | 36.5% |
| 3 | 44.5% |
| 4 | 50.5% |

Note: recall improves significantly with more vectors. The 50-vector demo achieves ~90% recall at similar bit-widths.

## Development

```bash
npm install          # Install dependencies
npm run typecheck    # Type checking
npm test             # Run tests
npm run test:watch   # Watch mode
npm run test:coverage # Coverage report
npm run build        # Build ESM + CJS output
npm run bench        # Run benchmarks
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

## License

[MIT](./LICENSE)
