import { TurboQuantMSE } from '../core/mse-quantizer.js';
import { TurboQuantProd } from '../core/prod-quantizer.js';
import type { QuantizedMSE, QuantizedProd } from '../core/types.js';
import { packIndices, unpackIndices } from '../math/bit-pack.js';

const KV_CACHE_MAGIC = 0x54514b56; // "TQKV"
const KV_CACHE_VERSION = 1;

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
    actualBytes: number;
  } {
    const keyBitsPerVec = this.keyDim * this.keyBits + 32;
    const valueBitsPerVec = this.valueDim * this.valueBits;
    const totalBits = (keyBitsPerVec + valueBitsPerVec) * this.compressedKeys.length;
    const uncompressedBits = (this.keyDim + this.valueDim) * 64 * this.compressedKeys.length;

    // Actual in-memory per key: Uint8Array indices + packed QJL + residualNorm + norm
    const keyIndicesBytes = this.keyDim;
    const keyQjlBytes = Math.ceil(this.keyDim / 8);
    const keyScalarBytes = 8 + 8; // residualNorm + optional norm
    const perKeyActual = keyIndicesBytes + keyQjlBytes + keyScalarBytes;

    // Actual in-memory per value: Uint8Array indices + norm
    const valueIndicesBytes = this.valueDim;
    const valueScalarBytes = 8; // optional norm
    const perValueActual = valueIndicesBytes + valueScalarBytes;

    const actualBytes = (perKeyActual + perValueActual) * this.compressedKeys.length;

    return {
      keyBitsPerVector: keyBitsPerVec,
      valueBitsPerVector: valueBitsPerVec,
      totalBytes: Math.ceil(totalBits / 8),
      compressionRatio: uncompressedBits > 0 ? uncompressedBits / totalBits : 0,
      actualBytes,
    };
  }

  /** Clear all cached entries. */
  clear(): void {
    this.compressedKeys.length = 0;
    this.compressedValues.length = 0;
  }

  /** Serialize the cache to a compact binary buffer. */
  toBuffer(): ArrayBuffer {
    const keyMseBits = this.keyBits - 1;
    const keyPackedSize = Math.ceil((this.keyDim * keyMseBits) / 8);
    const keyQjlSize = Math.ceil(this.keyDim / 8);
    const valuePackedSize = Math.ceil((this.valueDim * this.valueBits) / 8);

    // Header: 4 + 2 + 2 + 2 + 4 + 4 + 4 = 22 bytes
    let totalSize = 22;
    for (let i = 0; i < this.compressedKeys.length; i++) {
      const key = this.compressedKeys[i]!;
      const value = this.compressedValues[i]!;
      totalSize += keyPackedSize + keyQjlSize + 8 + 1; // indices + qjl + residualNorm + hasNorm
      if (key.norm !== undefined) totalSize += 8;
      totalSize += valuePackedSize + 1; // indices + hasNorm
      if (value.norm !== undefined) totalSize += 8;
    }

    const buffer = new ArrayBuffer(totalSize);
    const view = new DataView(buffer);
    const bytes = new Uint8Array(buffer);
    let offset = 0;

    // Header
    view.setUint32(offset, KV_CACHE_MAGIC, false);
    offset += 4;
    view.setUint16(offset, KV_CACHE_VERSION, false);
    offset += 2;
    view.setUint16(offset, this.keyBits, false);
    offset += 2;
    view.setUint16(offset, this.valueBits, false);
    offset += 2;
    view.setUint32(offset, this.keyDim, false);
    offset += 4;
    view.setUint32(offset, this.valueDim, false);
    offset += 4;
    view.setUint32(offset, this.compressedKeys.length, false);
    offset += 4;

    for (let i = 0; i < this.compressedKeys.length; i++) {
      const key = this.compressedKeys[i]!;
      const value = this.compressedValues[i]!;

      // Key packed MSE indices
      const keyPacked = packIndices(key.indices, keyMseBits);
      bytes.set(keyPacked, offset);
      offset += keyPackedSize;

      // Key QJL bits
      bytes.set(key.qjlBits, offset);
      offset += keyQjlSize;

      // Key residualNorm
      view.setFloat64(offset, key.residualNorm, false);
      offset += 8;

      // Key hasNorm + norm
      if (key.norm !== undefined) {
        view.setUint8(offset, 1);
        offset += 1;
        view.setFloat64(offset, key.norm, false);
        offset += 8;
      } else {
        view.setUint8(offset, 0);
        offset += 1;
      }

      // Value packed indices
      const valuePacked = packIndices(value.indices, this.valueBits);
      bytes.set(valuePacked, offset);
      offset += valuePackedSize;

      // Value hasNorm + norm
      if (value.norm !== undefined) {
        view.setUint8(offset, 1);
        offset += 1;
        view.setFloat64(offset, value.norm, false);
        offset += 8;
      } else {
        view.setUint8(offset, 0);
        offset += 1;
      }
    }

    return buffer;
  }

  /** Deserialize a cache from a buffer. */
  static fromBuffer(buffer: ArrayBuffer, options: KVCacheCompressorOptions): KVCacheCompressor {
    const view = new DataView(buffer);
    const bytes = new Uint8Array(buffer);
    let offset = 0;

    // Header
    const magic = view.getUint32(offset, false);
    offset += 4;
    if (magic !== KV_CACHE_MAGIC) {
      throw new Error(`Invalid buffer: expected magic 0x${KV_CACHE_MAGIC.toString(16)}, got 0x${magic.toString(16)}`);
    }

    const version = view.getUint16(offset, false);
    offset += 2;
    if (version !== KV_CACHE_VERSION) {
      throw new Error(`Unsupported version: ${version}`);
    }

    const keyBits = view.getUint16(offset, false);
    offset += 2;
    const valueBits = view.getUint16(offset, false);
    offset += 2;
    const keyDim = view.getUint32(offset, false);
    offset += 4;
    const valueDim = view.getUint32(offset, false);
    offset += 4;
    const count = view.getUint32(offset, false);
    offset += 4;

    const compressor = new KVCacheCompressor({
      ...options,
      keyDim,
      valueDim,
      keyBits,
      valueBits,
    });

    const keyMseBits = keyBits - 1;
    const keyPackedSize = Math.ceil((keyDim * keyMseBits) / 8);
    const keyQjlSize = Math.ceil(keyDim / 8);
    const valuePackedSize = Math.ceil((valueDim * valueBits) / 8);

    for (let i = 0; i < count; i++) {
      // Key packed MSE indices
      const keyPackedSlice = bytes.slice(offset, offset + keyPackedSize);
      const keyIndices = unpackIndices(keyPackedSlice, keyMseBits, keyDim);
      offset += keyPackedSize;

      // Key QJL bits
      const keyQjlBits = bytes.slice(offset, offset + keyQjlSize);
      offset += keyQjlSize;

      // Key residualNorm
      const keyResidualNorm = view.getFloat64(offset, false);
      offset += 8;

      // Key hasNorm + norm
      const keyHasNorm = view.getUint8(offset);
      offset += 1;
      let keyNorm: number | undefined;
      if (keyHasNorm) {
        keyNorm = view.getFloat64(offset, false);
        offset += 8;
      }

      const quantizedKey: QuantizedProd = { indices: keyIndices, qjlBits: keyQjlBits, residualNorm: keyResidualNorm };
      if (keyNorm !== undefined) {
        quantizedKey.norm = keyNorm;
      }

      // Value packed indices
      const valuePackedSlice = bytes.slice(offset, offset + valuePackedSize);
      const valueIndices = unpackIndices(valuePackedSlice, valueBits, valueDim);
      offset += valuePackedSize;

      // Value hasNorm + norm
      const valueHasNorm = view.getUint8(offset);
      offset += 1;
      let valueNorm: number | undefined;
      if (valueHasNorm) {
        valueNorm = view.getFloat64(offset, false);
        offset += 8;
      }

      const quantizedValue: QuantizedMSE = { indices: valueIndices };
      if (valueNorm !== undefined) {
        quantizedValue.norm = valueNorm;
      }

      compressor.compressedKeys.push(quantizedKey);
      compressor.compressedValues.push(quantizedValue);
    }

    return compressor;
  }
}
