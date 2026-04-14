import { describe, expect, it } from 'vitest';
import { VectorIndex } from '../../src/apps/vector-index.js';
import { createPRNG } from '../../src/rng/xorshift128.js';

function randomVector(d: number, rng: ReturnType<typeof createPRNG>): Float64Array {
  const v = new Float64Array(d);
  for (let i = 0; i < d; i++) v[i] = rng.nextGaussian();
  return v;
}

describe('VectorIndex', () => {
  const d = 32;

  it('self-query returns itself as top-1', () => {
    const index = new VectorIndex({ dimension: d, bits: 3, seed: 42 });
    const rng = createPRNG(1);
    const v = randomVector(d, rng);
    index.add('a', v);
    const results = index.search(v, 1);
    expect(results.length).toBe(1);
    expect(results[0]!.id).toBe('a');
  });

  it('returns results sorted by descending score', () => {
    const index = new VectorIndex({ dimension: d, bits: 3, seed: 42 });
    const rng = createPRNG(2);
    for (let i = 0; i < 10; i++) {
      index.add(i, randomVector(d, rng));
    }
    const query = randomVector(d, rng);
    const results = index.search(query, 5);
    for (let i = 1; i < results.length; i++) {
      expect(results[i]!.score).toBeLessThanOrEqual(results[i - 1]!.score);
    }
  });

  it('size is correct', () => {
    const index = new VectorIndex({ dimension: d, bits: 3 });
    expect(index.size).toBe(0);
    index.add('a', new Float64Array(d));
    expect(index.size).toBe(1);
    index.add('b', new Float64Array(d));
    expect(index.size).toBe(2);
  });

  it('remove works', () => {
    const index = new VectorIndex({ dimension: d, bits: 3 });
    const rng = createPRNG(3);
    index.add('a', randomVector(d, rng));
    index.add('b', randomVector(d, rng));
    expect(index.size).toBe(2);
    expect(index.remove('a')).toBe(true);
    expect(index.size).toBe(1);
    expect(index.remove('a')).toBe(false);
  });

  it('addBatch adds multiple vectors', () => {
    const index = new VectorIndex({ dimension: d, bits: 3 });
    const rng = createPRNG(4);
    index.addBatch([
      { id: 'x', vector: randomVector(d, rng) },
      { id: 'y', vector: randomVector(d, rng) },
      { id: 'z', vector: randomVector(d, rng) },
    ]);
    expect(index.size).toBe(3);
  });

  it('accepts number[] as input', () => {
    const index = new VectorIndex({ dimension: 4, bits: 2 });
    index.add('a', [1, 2, 3, 4]);
    const results = index.search([1, 2, 3, 4], 1);
    expect(results.length).toBe(1);
  });

  it('throws on dimension mismatch', () => {
    const index = new VectorIndex({ dimension: d, bits: 3 });
    expect(() => index.add('a', new Float64Array(d + 1))).toThrow();
    expect(() => index.search(new Float64Array(d + 1), 1)).toThrow();
  });

  it('memoryUsage reports compression ratio > 1', () => {
    const index = new VectorIndex({ dimension: d, bits: 3 });
    const rng = createPRNG(5);
    index.add('a', randomVector(d, rng));
    const usage = index.memoryUsage;
    expect(usage.compressionRatio).toBeGreaterThan(1);
    expect(usage.bitsPerVector).toBeLessThan(d * 64);
  });

  it('replacing existing id updates the vector', () => {
    const index = new VectorIndex({ dimension: d, bits: 3, seed: 42 });
    const rng = createPRNG(6);
    const v1 = randomVector(d, rng);
    const v2 = randomVector(d, rng);
    index.add('a', v1);
    const score1 = index.search(v2, 1)[0]!.score;
    index.add('a', v2);
    const score2 = index.search(v2, 1)[0]!.score;
    expect(index.size).toBe(1);
    // Searching with v2 should score higher after replacing with v2
    expect(score2).toBeGreaterThan(score1);
  });

  it('memoryUsage reports actualBytes', () => {
    const index = new VectorIndex({ dimension: d, bits: 3 });
    const rng = createPRNG(10);
    index.add('a', randomVector(d, rng));
    const usage = index.memoryUsage;
    expect(usage.actualBytes).toBeGreaterThan(0);
  });

  describe('toBuffer / fromBuffer', () => {
    it('roundtrip preserves search results', () => {
      const index = new VectorIndex({ dimension: d, bits: 3, seed: 42 });
      const rng = createPRNG(100);
      const vectors: Float64Array[] = [];
      for (let i = 0; i < 10; i++) {
        const v = randomVector(d, rng);
        vectors.push(v);
        index.add(i, v);
      }

      const query = randomVector(d, rng);
      const originalResults = index.search(query, 5);

      const buffer = index.toBuffer();
      const restored = VectorIndex.fromBuffer(buffer, { dimension: d, bits: 3, seed: 42 });

      expect(restored.size).toBe(10);
      const restoredResults = restored.search(query, 5);

      expect(restoredResults.length).toBe(originalResults.length);
      for (let i = 0; i < originalResults.length; i++) {
        expect(restoredResults[i]!.id).toBe(originalResults[i]!.id);
        expect(restoredResults[i]!.score).toBeCloseTo(originalResults[i]!.score, 10);
      }
    });

    it('roundtrip with empty index', () => {
      const index = new VectorIndex({ dimension: d, bits: 3, seed: 42 });
      const buffer = index.toBuffer();
      const restored = VectorIndex.fromBuffer(buffer, { dimension: d, bits: 3, seed: 42 });
      expect(restored.size).toBe(0);
    });

    it('roundtrip with mixed string and number IDs', () => {
      const index = new VectorIndex({ dimension: d, bits: 3, seed: 42 });
      const rng = createPRNG(200);
      index.add('hello', randomVector(d, rng));
      index.add(42, randomVector(d, rng));
      index.add('world', randomVector(d, rng));
      index.add(0, randomVector(d, rng));

      const buffer = index.toBuffer();
      const restored = VectorIndex.fromBuffer(buffer, { dimension: d, bits: 3, seed: 42 });

      expect(restored.size).toBe(4);
      const query = randomVector(d, rng);
      const originalResults = index.search(query, 4);
      const restoredResults = restored.search(query, 4);

      for (let i = 0; i < originalResults.length; i++) {
        expect(restoredResults[i]!.id).toBe(originalResults[i]!.id);
        expect(restoredResults[i]!.score).toBeCloseTo(originalResults[i]!.score, 10);
      }
    });

    it('throws on invalid buffer (wrong magic)', () => {
      const buffer = new ArrayBuffer(17);
      const view = new DataView(buffer);
      view.setUint32(0, 0xdeadbeef, false);
      expect(() => VectorIndex.fromBuffer(buffer, { dimension: d })).toThrow('Invalid buffer');
    });
  });
});
