// Core quantization
export { TurboQuantMSE } from './core/mse-quantizer.js';
export { TurboQuantProd } from './core/prod-quantizer.js';
export { QJL } from './core/qjl.js';

// Core utilities
export { createRotation, type Rotation } from './core/rotation.js';
export { solveLloydMax, computeDistortion } from './core/lloyd-max.js';
export { getCodebook, clearCodebookCache } from './core/codebook.js';

// Types
export type {
  TurboQuantConfig,
  QuantizedMSE,
  QuantizedProd,
  Codebook,
} from './core/types.js';

// Application APIs
export { VectorIndex, type VectorIndexOptions, type SearchResult } from './apps/vector-index.js';
export { KVCacheCompressor, type KVCacheCompressorOptions } from './apps/kv-cache.js';

// RNG
export { createPRNG } from './rng/xorshift128.js';
export type { PRNG } from './rng/types.js';
