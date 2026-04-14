// Core quantization

export { KVCacheCompressor, type KVCacheCompressorOptions } from './apps/kv-cache.js';
// Application APIs
export { type SearchResult, VectorIndex, type VectorIndexOptions } from './apps/vector-index.js';
export { clearCodebookCache, getCodebook } from './core/codebook.js';
export { computeDistortion, solveLloydMax } from './core/lloyd-max.js';
export { TurboQuantMSE } from './core/mse-quantizer.js';
export { TurboQuantProd } from './core/prod-quantizer.js';
export { QJL } from './core/qjl.js';
// Core utilities
export { createRotation, type Rotation } from './core/rotation.js';
export { fwht, nextPow2 } from './math/hadamard.js';
// Types
export type {
  Codebook,
  QuantizedMSE,
  QuantizedProd,
  TurboQuantConfig,
} from './core/types.js';
export type { PRNG } from './rng/types.js';
// RNG
export { createPRNG } from './rng/xorshift128.js';
