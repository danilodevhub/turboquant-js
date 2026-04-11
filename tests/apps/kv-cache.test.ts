import { describe, expect, it } from 'vitest';
import { KVCacheCompressor } from '../../src/apps/kv-cache.js';
import { createPRNG } from '../../src/rng/xorshift128.js';

function randomVector(d: number, rng: ReturnType<typeof createPRNG>): Float64Array {
  const v = new Float64Array(d);
  for (let i = 0; i < d; i++) v[i] = rng.nextGaussian();
  return v;
}

describe('KVCacheCompressor', () => {
  const keyDim = 32;
  const valueDim = 32;

  it('append increases length', () => {
    const kv = new KVCacheCompressor({ keyDim, valueDim });
    const rng = createPRNG(1);
    kv.append(
      [randomVector(keyDim, rng), randomVector(keyDim, rng)],
      [randomVector(valueDim, rng), randomVector(valueDim, rng)],
    );
    expect(kv.length).toBe(2);
  });

  it('attentionScores returns correct length', () => {
    const kv = new KVCacheCompressor({ keyDim, valueDim });
    const rng = createPRNG(2);
    const n = 5;
    const keys = Array.from({ length: n }, () => randomVector(keyDim, rng));
    const values = Array.from({ length: n }, () => randomVector(valueDim, rng));
    kv.append(keys, values);
    const query = randomVector(keyDim, rng);
    const scores = kv.attentionScores(query);
    expect(scores.length).toBe(n);
  });

  it('attention scores are finite', () => {
    const kv = new KVCacheCompressor({ keyDim, valueDim, seed: 42 });
    const rng = createPRNG(3);
    const keys = Array.from({ length: 3 }, () => randomVector(keyDim, rng));
    const values = Array.from({ length: 3 }, () => randomVector(valueDim, rng));
    kv.append(keys, values);
    const scores = kv.attentionScores(randomVector(keyDim, rng));
    for (let i = 0; i < scores.length; i++) {
      expect(Number.isFinite(scores[i])).toBe(true);
    }
  });

  it('retrieveValues returns vectors of correct dimension', () => {
    const kv = new KVCacheCompressor({ keyDim, valueDim });
    const rng = createPRNG(4);
    kv.append([randomVector(keyDim, rng)], [randomVector(valueDim, rng)]);
    const values = kv.retrieveValues([0]);
    expect(values.length).toBe(1);
    expect(values[0]!.length).toBe(valueDim);
  });

  it('clear empties the cache', () => {
    const kv = new KVCacheCompressor({ keyDim, valueDim });
    const rng = createPRNG(5);
    kv.append([randomVector(keyDim, rng)], [randomVector(valueDim, rng)]);
    expect(kv.length).toBe(1);
    kv.clear();
    expect(kv.length).toBe(0);
  });

  it('memoryUsage reports compression ratio > 1', () => {
    const kv = new KVCacheCompressor({ keyDim, valueDim });
    const rng = createPRNG(6);
    kv.append([randomVector(keyDim, rng)], [randomVector(valueDim, rng)]);
    const usage = kv.memoryUsage;
    expect(usage.compressionRatio).toBeGreaterThan(1);
    expect(usage.totalBytes).toBeGreaterThan(0);
  });

  it('throws when keys and values length mismatch', () => {
    const kv = new KVCacheCompressor({ keyDim, valueDim });
    const rng = createPRNG(7);
    expect(() => kv.append([randomVector(keyDim, rng)], [])).toThrow();
  });
});
