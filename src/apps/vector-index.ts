import { TurboQuantProd } from '../core/prod-quantizer.js';
import { normalize, dot, norm as vecNorm } from '../math/vec.js';
import type { QuantizedProd } from '../core/types.js';

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
  get memoryUsage(): { totalBits: number; bitsPerVector: number; compressionRatio: number } {
    const bitsPerVector = this.dimension * this.bits + 32; // +32 for residual norm float
    const totalBits = bitsPerVector * this.entries.length;
    const uncompressedBitsPerVector = this.dimension * 64; // float64
    return {
      totalBits,
      bitsPerVector,
      compressionRatio: uncompressedBitsPerVector / bitsPerVector,
    };
  }
}
