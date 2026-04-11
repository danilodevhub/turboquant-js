import { TurboQuantMSE } from '../core/mse-quantizer.js';
import { TurboQuantProd } from '../core/prod-quantizer.js';
import type { QuantizedMSE, QuantizedProd } from '../core/types.js';

export interface KVCacheCompressorOptions {
  /** Dimension of key vectors. */
  keyDim: number;
  /** Dimension of value vectors. */
  valueDim: number;
  /** Bit-width for keys. Default: 3. Uses TurboQuantProd (unbiased attention scores). */
  keyBits?: number;
  /** Bit-width for values. Default: 2. Uses TurboQuantMSE (low MSE reconstruction). */
  valueBits?: number;
  /** PRNG seed. Default: 42. */
  seed?: number;
}

/**
 * KV cache compressor for LLM inference.
 * Keys use TurboQuantProd for unbiased attention score estimation.
 * Values use TurboQuantMSE for low-MSE reconstruction.
 */
export class KVCacheCompressor {
  readonly keyDim: number;
  readonly valueDim: number;
  readonly keyBits: number;
  readonly valueBits: number;
  private readonly keyQuantizer: TurboQuantProd;
  private readonly valueQuantizer: TurboQuantMSE;
  private readonly compressedKeys: QuantizedProd[] = [];
  private readonly compressedValues: QuantizedMSE[] = [];

  constructor(options: KVCacheCompressorOptions) {
    this.keyDim = options.keyDim;
    this.valueDim = options.valueDim;
    this.keyBits = options.keyBits ?? 3;
    this.valueBits = options.valueBits ?? 2;

    const seed = options.seed ?? 42;
    this.keyQuantizer = new TurboQuantProd({
      dimension: options.keyDim,
      bits: this.keyBits,
      seed,
    });
    this.valueQuantizer = new TurboQuantMSE({
      dimension: options.valueDim,
      bits: this.valueBits,
      seed: seed + 2000003,
    });
  }

  /**
   * Append key-value pairs to the cache.
   * Each key and value is a Float64Array.
   */
  append(keys: Float64Array[], values: Float64Array[]): void {
    if (keys.length !== values.length) {
      throw new RangeError('keys and values must have the same length');
    }
    for (let i = 0; i < keys.length; i++) {
      this.compressedKeys.push(this.keyQuantizer.quantize(keys[i]!));
      this.compressedValues.push(this.valueQuantizer.quantize(values[i]!));
    }
  }

  /**
   * Compute attention scores: inner product of query with each compressed key.
   * Returns an unbiased estimate of [⟨q, k_0⟩, ⟨q, k_1⟩, ...].
   */
  attentionScores(query: Float64Array): Float64Array {
    const scores = new Float64Array(this.compressedKeys.length);
    for (let i = 0; i < this.compressedKeys.length; i++) {
      scores[i] = this.keyQuantizer.innerProduct(query, this.compressedKeys[i]!);
    }
    return scores;
  }

  /** Retrieve (dequantize) values at the given indices. */
  retrieveValues(indices: number[]): Float64Array[] {
    return indices.map((i) => this.valueQuantizer.dequantize(this.compressedValues[i]!));
  }

  /** Number of cached key-value pairs. */
  get length(): number {
    return this.compressedKeys.length;
  }

  /** Memory usage statistics. */
  get memoryUsage(): {
    keyBitsPerVector: number;
    valueBitsPerVector: number;
    totalBytes: number;
    compressionRatio: number;
  } {
    const keyBitsPerVec = this.keyDim * this.keyBits + 32;
    const valueBitsPerVec = this.valueDim * this.valueBits;
    const totalBits = (keyBitsPerVec + valueBitsPerVec) * this.compressedKeys.length;
    const uncompressedBits = (this.keyDim + this.valueDim) * 64 * this.compressedKeys.length;
    return {
      keyBitsPerVector: keyBitsPerVec,
      valueBitsPerVector: valueBitsPerVec,
      totalBytes: Math.ceil(totalBits / 8),
      compressionRatio: uncompressedBits > 0 ? uncompressedBits / totalBits : 0,
    };
  }

  /** Clear all cached entries. */
  clear(): void {
    this.compressedKeys.length = 0;
    this.compressedValues.length = 0;
  }
}
