# turboquant-js

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

```ts
import { VectorIndex } from 'turboquant-js';

// Create a compressed vector index
const index = new VectorIndex({ dimension: 384, bits: 3, metric: 'cosine' });

// Add vectors
index.add('doc1', embedding1);
index.add('doc2', embedding2);

// Search
const results = index.search(queryEmbedding, 10);
```

## License

MIT
