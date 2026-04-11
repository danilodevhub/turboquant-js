import { scale, norm as vecNorm } from '../math/vec.js';
import { getCodebook } from './codebook.js';
import { createRotation, type Rotation } from './rotation.js';
import type { Codebook, QuantizedMSE, TurboQuantConfig } from './types.js';

/**
 * TurboQuant Stage 1: MSE-optimal scalar quantizer.
 * Randomly rotates input vectors then applies per-coordinate quantization
 * using an optimal codebook derived from the coordinate distribution.
 */
export class TurboQuantMSE {
  readonly dimension: number;
  readonly bits: number;
  readonly codebook: Codebook;
  private readonly rotation: Rotation;

  constructor(config: TurboQuantConfig) {
    this.dimension = config.dimension;
    this.bits = config.bits;
    this.rotation = createRotation(config.dimension, config.seed ?? 42);
    this.codebook = getCodebook(config.dimension, config.bits);
  }

  /** Quantize a vector. */
  quantize(x: Float64Array): QuantizedMSE {
    let inputNorm: number | undefined;
    let vec = x;

    // Handle non-unit-norm vectors
    const n = vecNorm(x);
    if (Math.abs(n - 1) > 1e-8) {
      inputNorm = n;
      vec = n > 0 ? scale(x, 1 / n) : x;
    }

    // Rotate
    const y = this.rotation.rotate(vec);

    // Quantize each coordinate
    const { centroids, boundaries } = this.codebook;
    const indices = new Uint8Array(this.dimension);
    for (let j = 0; j < this.dimension; j++) {
      indices[j] = findNearest(y[j]!, centroids, boundaries);
    }

    return inputNorm !== undefined ? { indices, norm: inputNorm } : { indices };
  }

  /** Dequantize (reconstruct) a vector. */
  dequantize(q: QuantizedMSE): Float64Array {
    const { centroids } = this.codebook;

    // Rebuild rotated vector from centroids
    const y = new Float64Array(this.dimension);
    for (let j = 0; j < this.dimension; j++) {
      y[j] = centroids[q.indices[j]!]!;
    }

    // Unrotate
    let result = this.rotation.unrotate(y);

    // Restore norm if stored
    if (q.norm !== undefined && q.norm > 0) {
      result = scale(result, q.norm);
    }

    return result;
  }
}

/**
 * Find the index of the nearest centroid using boundary-based search.
 * For small centroid counts (2-16), linear scan over boundaries is
 * faster than binary search due to branch prediction.
 */
function findNearest(value: number, centroids: Float64Array, boundaries: Float64Array): number {
  // Walk boundaries: value < boundary[i] means it belongs to centroid i
  for (let i = 0; i < boundaries.length; i++) {
    if (value < boundaries[i]!) return i;
  }
  return centroids.length - 1;
}
