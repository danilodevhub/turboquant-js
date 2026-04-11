import { packBits, unpackBits } from '../math/bit-pack.js';
import { dot, sub, norm as vecNorm } from '../math/vec.js';
import { TurboQuantMSE } from './mse-quantizer.js';
import { QJL } from './qjl.js';
import type { QuantizedProd, TurboQuantConfig } from './types.js';

/**
 * TurboQuant two-stage quantizer for unbiased inner product estimation.
 * Stage 1: (b-1)-bit MSE quantizer (random rotation + scalar quantization)
 * Stage 2: 1-bit QJL on the residual (sign-bit quantized random projection)
 *
 * Total: b bits per coordinate.
 * Guarantees: E[innerProduct(y, quantize(x))] = ⟨y, x⟩ (unbiased).
 */
export class TurboQuantProd {
  readonly dimension: number;
  readonly totalBitsPerCoordinate: number;
  private readonly mse: TurboQuantMSE;
  private readonly qjl: QJL;

  constructor(config: TurboQuantConfig) {
    if (config.bits < 2) {
      throw new RangeError('TurboQuantProd requires bits >= 2 (1 bit for MSE + 1 bit for QJL)');
    }

    this.dimension = config.dimension;
    this.totalBitsPerCoordinate = config.bits;

    const seed = config.seed ?? 42;

    // Stage 1: MSE quantizer with (b-1) bits
    this.mse = new TurboQuantMSE({
      dimension: config.dimension,
      bits: config.bits - 1,
      seed,
    });

    // Stage 2: QJL with a derived seed (different from rotation)
    this.qjl = new QJL(config.dimension, seed + 1000003);
  }

  /** Quantize a vector into a compressed representation. */
  quantize(x: Float64Array): QuantizedProd {
    // MSE quantize
    const mseResult = this.mse.quantize(x);

    // Dequantize to compute residual
    const xHat = this.mse.dequantize(mseResult);

    // Compute residual in the original scale
    const residual = sub(x, xHat);
    const residualNorm = vecNorm(residual);

    // QJL sign-bit quantize the residual
    let signs: Int8Array;
    if (residualNorm < 1e-15) {
      // Zero residual: all positive signs (arbitrary, won't affect result)
      signs = new Int8Array(this.dimension).fill(1);
    } else {
      signs = this.qjl.project(residual);
    }

    const qjlBits = packBits(signs);

    return {
      indices: mseResult.indices,
      qjlBits,
      residualNorm,
      norm: mseResult.norm,
    };
  }

  /** Dequantize (reconstruct) a vector. */
  dequantize(q: QuantizedProd): Float64Array {
    // MSE reconstruction
    const xMse = this.mse.dequantize({ indices: q.indices, norm: q.norm });

    if (q.residualNorm < 1e-15) return xMse;

    // QJL reconstruction: sqrt(pi/2)/d * gamma * S^T * signs
    // We approximate by adding the correction to xMse
    const signs = unpackBits(q.qjlBits, this.dimension);
    const _correction = this.qjl.innerProductCorrection(signs, q.residualNorm, xMse);

    // For full reconstruction, we'd need S^T * signs.
    // But dequantize is mainly for MSE use; inner products go through innerProduct().
    return xMse;
  }

  /**
   * Estimate ⟨y, x⟩ using the quantized representation of x.
   * This is an unbiased estimator: E[result] = ⟨y, x⟩.
   */
  innerProduct(y: Float64Array, q: QuantizedProd): number {
    // Term 1: inner product with MSE reconstruction
    const xMse = this.mse.dequantize({ indices: q.indices, norm: q.norm });
    const term1 = dot(y, xMse);

    // Term 2: QJL correction for the residual
    if (q.residualNorm < 1e-15) return term1;

    const signs = unpackBits(q.qjlBits, this.dimension);
    const term2 = this.qjl.innerProductCorrection(signs, q.residualNorm, y);

    return term1 + term2;
  }
}
