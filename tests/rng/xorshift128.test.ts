import { describe, it, expect } from 'vitest';
import { createPRNG } from '../../src/rng/xorshift128.js';

describe('xorshift128', () => {
  it('same seed produces identical sequence', () => {
    const a = createPRNG(42);
    const b = createPRNG(42);
    for (let i = 0; i < 1000; i++) {
      expect(a.next()).toBe(b.next());
    }
  });

  it('different seeds produce different sequences', () => {
    const a = createPRNG(42);
    const b = createPRNG(43);
    let same = 0;
    for (let i = 0; i < 100; i++) {
      if (a.next() === b.next()) same++;
    }
    expect(same).toBeLessThan(5);
  });

  it('values are in [0, 1)', () => {
    const rng = createPRNG(123);
    for (let i = 0; i < 10000; i++) {
      const v = rng.next();
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
    }
  });

  it('first 1000 values are all distinct', () => {
    const rng = createPRNG(42);
    const seen = new Set<number>();
    for (let i = 0; i < 1000; i++) {
      seen.add(rng.next());
    }
    expect(seen.size).toBe(1000);
  });

  it('chi-squared test: uniform distribution across 10 buckets', () => {
    const rng = createPRNG(99);
    const n = 10000;
    const buckets = new Array(10).fill(0) as number[];
    for (let i = 0; i < n; i++) {
      const bucket = Math.min(9, Math.floor(rng.next() * 10));
      buckets[bucket]++;
    }
    const expected = n / 10;
    let chiSq = 0;
    for (const count of buckets) {
      chiSq += (count - expected) ** 2 / expected;
    }
    // chi-squared critical value for 9 df at p=0.01 is 21.67
    expect(chiSq).toBeLessThan(21.67);
  });
});
