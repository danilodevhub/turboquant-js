import { TurboQuantProd } from '../core/prod-quantizer.js';
import type { QuantizedProd } from '../core/types.js';
import { packIndices, unpackIndices } from '../math/bit-pack.js';
import { normalize } from '../math/vec.js';

const VECTOR_INDEX_MAGIC = 0x54515649; // "TQVI"
const VECTOR_INDEX_VERSION = 1;

export interface VectorIndexOptions {
  /** Vector dimension. */
  dimension: number;
  /** Bit-width per coordinate. Default: 3. */
  bits?: number;
  /** PRNG seed. Default: 42. */
  seed?: number;
  /** Distance metric. Default: 'cosine'. */
  metric?: 'ip' | 'cosine';
}

export interface SearchResult {
  id: string | number;
  score: number;
}

interface StoredEntry {
  id: string | number;
  quantized: QuantizedProd;
}

/**
 * Compressed vector index for approximate nearest neighbor search.
 * Uses TurboQuant for near-optimal compression with unbiased inner product estimates.
 */
export class VectorIndex {
  readonly dimension: number;
  readonly bits: number;
  readonly metric: 'ip' | 'cosine';
  private readonly quantizer: TurboQuantProd;
  private readonly entries: StoredEntry[] = [];
  private readonly idToIndex = new Map<string | number, number>();

  constructor(options: VectorIndexOptions) {
    this.dimension = options.dimension;
    this.bits = options.bits ?? 3;
    this.metric = options.metric ?? 'cosine';
    this.quantizer = new TurboQuantProd({
      dimension: options.dimension,
      bits: this.bits,
      seed: options.seed ?? 42,
    });
  }

  /** Add a vector to the index. */
  add(id: string | number, vector: Float64Array | number[]): void {
    const vec = vector instanceof Float64Array ? vector : new Float64Array(vector);
    if (vec.length !== this.dimension) {
      throw new RangeError(`Expected dimension ${this.dimension}, got ${vec.length}`);
    }

    const input = this.metric === 'cosine' ? normalize(vec) : vec;
    const quantized = this.quantizer.quantize(input);

    if (this.idToIndex.has(id)) {
      // Replace existing
      const idx = this.idToIndex.get(id)!;
      this.entries[idx] = { id, quantized };
    } else {
      this.idToIndex.set(id, this.entries.length);
      this.entries.push({ id, quantized });
    }
  }

  /** Add multiple vectors at once. */
  addBatch(items: Array<{ id: string | number; vector: Float64Array | number[] }>): void {
    for (const item of items) {
      this.add(item.id, item.vector);
    }
  }

  /** Search for the top-k nearest neighbors. */
  search(query: Float64Array | number[], k: number): SearchResult[] {
    const q = query instanceof Float64Array ? query : new Float64Array(query);
    if (q.length !== this.dimension) {
      throw new RangeError(`Expected dimension ${this.dimension}, got ${q.length}`);
    }

    const queryVec = this.metric === 'cosine' ? normalize(q) : q;

    // Score all entries
    const scored: SearchResult[] = [];
    for (const entry of this.entries) {
      const score = this.quantizer.innerProduct(queryVec, entry.quantized);
      scored.push({ id: entry.id, score });
    }

    // Sort descending by score and return top-k
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, k);
  }

  /** Remove a vector by id. Returns true if found and removed. */
  remove(id: string | number): boolean {
    const idx = this.idToIndex.get(id);
    if (idx === undefined) return false;

    // Swap with last and pop (O(1) removal)
    const lastIdx = this.entries.length - 1;
    if (idx !== lastIdx) {
      const lastEntry = this.entries[lastIdx]!;
      this.entries[idx] = lastEntry;
      this.idToIndex.set(lastEntry.id, idx);
    }
    this.entries.pop();
    this.idToIndex.delete(id);
    return true;
  }

  /** Number of vectors in the index. */
  get size(): number {
    return this.entries.length;
  }

  /** Memory usage statistics. */
  get memoryUsage(): {
    totalBits: number;
    bitsPerVector: number;
    compressionRatio: number;
    actualBytes: number;
  } {
    const bitsPerVector = this.dimension * this.bits + 32; // +32 for residual norm float
    const totalBits = bitsPerVector * this.entries.length;
    const uncompressedBitsPerVector = this.dimension * 64; // float64

    // Actual in-memory: Uint8Array indices + packed QJL + residualNorm + norm + overhead
    const indicesBytes = this.dimension; // Uint8Array, 1 byte per coordinate
    const qjlBytes = Math.ceil(this.dimension / 8);
    const scalarBytes = 8 + 8; // residualNorm + optional norm
    const perEntryActual = indicesBytes + qjlBytes + scalarBytes;
    const actualBytes = perEntryActual * this.entries.length;

    return {
      totalBits,
      bitsPerVector,
      compressionRatio: uncompressedBitsPerVector / bitsPerVector,
      actualBytes,
    };
  }

  /** Add a pre-quantized entry, bypassing re-quantization. */
  private _addQuantized(id: string | number, quantized: QuantizedProd): void {
    if (this.idToIndex.has(id)) {
      const idx = this.idToIndex.get(id)!;
      this.entries[idx] = { id, quantized };
    } else {
      this.idToIndex.set(id, this.entries.length);
      this.entries.push({ id, quantized });
    }
  }

  /** Serialize the index to a compact binary buffer. */
  toBuffer(): ArrayBuffer {
    const mseBits = this.bits - 1;
    const packedIndicesSize = Math.ceil((this.dimension * mseBits) / 8);
    const qjlBitsSize = Math.ceil(this.dimension / 8);

    // Calculate total size
    // Header: 4 + 2 + 2 + 4 + 4 + 1 = 17 bytes
    let totalSize = 17;
    for (const entry of this.entries) {
      // idType byte
      totalSize += 1;
      if (typeof entry.id === 'number') {
        totalSize += 8; // float64
      } else {
        const strBytes = new TextEncoder().encode(entry.id);
        totalSize += 2 + strBytes.length; // uint16 strlen + utf8
      }
      totalSize += packedIndicesSize;
      totalSize += qjlBitsSize;
      totalSize += 8; // residualNorm
      totalSize += 1; // hasNorm
      if (entry.quantized.norm !== undefined) {
        totalSize += 8;
      }
    }

    const buffer = new ArrayBuffer(totalSize);
    const view = new DataView(buffer);
    const bytes = new Uint8Array(buffer);
    let offset = 0;

    // Header
    view.setUint32(offset, VECTOR_INDEX_MAGIC, false);
    offset += 4;
    view.setUint16(offset, VECTOR_INDEX_VERSION, false);
    offset += 2;
    view.setUint16(offset, this.bits, false);
    offset += 2;
    view.setUint32(offset, this.dimension, false);
    offset += 4;
    view.setUint32(offset, this.entries.length, false);
    offset += 4;
    view.setUint8(offset, this.metric === 'cosine' ? 0 : 1);
    offset += 1;

    const encoder = new TextEncoder();

    // Per entry
    for (const entry of this.entries) {
      // ID
      if (typeof entry.id === 'number') {
        view.setUint8(offset, 0);
        offset += 1;
        view.setFloat64(offset, entry.id, false);
        offset += 8;
      } else {
        view.setUint8(offset, 1);
        offset += 1;
        const strBytes = encoder.encode(entry.id);
        view.setUint16(offset, strBytes.length, false);
        offset += 2;
        bytes.set(strBytes, offset);
        offset += strBytes.length;
      }

      // Packed MSE indices
      const packed = packIndices(entry.quantized.indices, mseBits);
      bytes.set(packed, offset);
      offset += packedIndicesSize;

      // QJL bits
      bytes.set(entry.quantized.qjlBits, offset);
      offset += qjlBitsSize;

      // residualNorm
      view.setFloat64(offset, entry.quantized.residualNorm, false);
      offset += 8;

      // hasNorm + norm
      if (entry.quantized.norm !== undefined) {
        view.setUint8(offset, 1);
        offset += 1;
        view.setFloat64(offset, entry.quantized.norm, false);
        offset += 8;
      } else {
        view.setUint8(offset, 0);
        offset += 1;
      }
    }

    return buffer;
  }

  /** Deserialize an index from a buffer. */
  static fromBuffer(buffer: ArrayBuffer, options: VectorIndexOptions): VectorIndex {
    const view = new DataView(buffer);
    const bytes = new Uint8Array(buffer);
    let offset = 0;

    // Header
    const magic = view.getUint32(offset, false);
    offset += 4;
    if (magic !== VECTOR_INDEX_MAGIC) {
      throw new Error(`Invalid buffer: expected magic 0x${VECTOR_INDEX_MAGIC.toString(16)}, got 0x${magic.toString(16)}`);
    }

    const version = view.getUint16(offset, false);
    offset += 2;
    if (version !== VECTOR_INDEX_VERSION) {
      throw new Error(`Unsupported version: ${version}`);
    }

    const bits = view.getUint16(offset, false);
    offset += 2;
    const dimension = view.getUint32(offset, false);
    offset += 4;
    const count = view.getUint32(offset, false);
    offset += 4;
    const metricByte = view.getUint8(offset);
    offset += 1;
    const metric: 'cosine' | 'ip' = metricByte === 0 ? 'cosine' : 'ip';

    const index = new VectorIndex({
      ...options,
      dimension,
      bits,
      metric,
    });

    const mseBits = bits - 1;
    const packedIndicesSize = Math.ceil((dimension * mseBits) / 8);
    const qjlBitsSize = Math.ceil(dimension / 8);
    const decoder = new TextDecoder();

    for (let i = 0; i < count; i++) {
      // ID
      const idType = view.getUint8(offset);
      offset += 1;
      let id: string | number;
      if (idType === 0) {
        id = view.getFloat64(offset, false);
        offset += 8;
      } else {
        const strLen = view.getUint16(offset, false);
        offset += 2;
        id = decoder.decode(bytes.slice(offset, offset + strLen));
        offset += strLen;
      }

      // Packed MSE indices
      const packedSlice = bytes.slice(offset, offset + packedIndicesSize);
      const indices = unpackIndices(packedSlice, mseBits, dimension);
      offset += packedIndicesSize;

      // QJL bits
      const qjlBits = bytes.slice(offset, offset + qjlBitsSize);
      offset += qjlBitsSize;

      // residualNorm
      const residualNorm = view.getFloat64(offset, false);
      offset += 8;

      // hasNorm + norm
      const hasNorm = view.getUint8(offset);
      offset += 1;
      let norm: number | undefined;
      if (hasNorm) {
        norm = view.getFloat64(offset, false);
        offset += 8;
      }

      const quantized: QuantizedProd = { indices, qjlBits, residualNorm };
      if (norm !== undefined) {
        quantized.norm = norm;
      }

      index._addQuantized(id, quantized);
    }

    return index;
  }
}
