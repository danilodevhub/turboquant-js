# turboquant-js

[![CI](https://github.com/danilodevhub/turboquant-js/actions/workflows/ci.yml/badge.svg)](https://github.com/danilodevhub/turboquant-js/actions/workflows/ci.yml)
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
// => { totalBits: 23200, bitsPerVector: 1184, compressionRatio: 20.8 }
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
| `memoryUsage` | Compression statistics |

**Options:** `dimension`, `bits` (default 3), `seed` (default 42), `metric` (`'cosine'` | `'ip'`)

### `KVCacheCompressor`

| Method / Property | Description |
|---|---|
| `append(keys, values)` | Compress and store KV pairs |
| `attentionScores(query)` | Unbiased attention score estimates |
| `retrieveValues(indices)` | Dequantize values at given positions |
| `length` | Number of cached pairs |
| `memoryUsage` | Compression statistics |
| `clear()` | Remove all cached entries |

**Options:** `keyDim`, `valueDim`, `keyBits` (default 3), `valueBits` (default 2), `seed` (default 42)

### `TurboQuantMSE`

MSE-optimal scalar quantizer. Randomly rotates input then applies per-coordinate quantization using an optimal codebook.

### `TurboQuantProd`

Two-stage quantizer (MSE + QJL) for unbiased inner product estimation. Uses `(b-1)` bits for MSE and 1 bit for QJL sign-based correction.

## How It Works

1. **Random rotation** — a structured orthogonal rotation (based on randomized Hadamard transforms) spreads vector energy uniformly across coordinates
2. **Lloyd-Max quantization** — each coordinate is scalar-quantized using an optimal codebook for the resulting Beta distribution
3. **QJL correction** (Prod quantizer only) — the quantization residual is sign-bit projected via a Johnson-Lindenstrauss sketch, enabling unbiased inner product estimation

## Development

```bash
npm install          # Install dependencies
npm run typecheck    # Type checking
npm test             # Run tests
npm run test:watch   # Watch mode
npm run test:coverage # Coverage report
npm run build        # Build ESM + CJS output
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

## License

[MIT](./LICENSE)
