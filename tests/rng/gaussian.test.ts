import { describe, expect, it } from 'vitest';
import { gaussianArray } from '../../src/rng/gaussian.js';
import { createPRNG } from '../../src/rng/xorshift128.js';

describe('gaussian', () => {
  it('mean is approximately 0', () => {
    const rng = createPRNG(42);
    const samples = gaussianArray(rng, 10000);
    const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
    expect(Math.abs(mean)).toBeLessThan(0.05);
  });

  it('stddev is approximately 1', () => {
    const rng = createPRNG(42);
    const samples = gaussianArray(rng, 10000);
    const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
    let variance = 0;
    for (const s of samples) variance += (s - mean) ** 2;
    variance /= samples.length;
    expect(Math.abs(Math.sqrt(variance) - 1)).toBeLessThan(0.05);
  });

  it('deterministic: same seed gives same array', () => {
    const a = gaussianArray(createPRNG(7), 100);
    const b = gaussianArray(createPRNG(7), 100);
    for (let i = 0; i < 100; i++) {
      expect(a[i]).toBe(b[i]);
    }
  });

  it('gaussianArray returns correct length', () => {
    const rng = createPRNG(0);
    expect(gaussianArray(rng, 50).length).toBe(50);
    expect(gaussianArray(rng, 1).length).toBe(1);
  });
});
