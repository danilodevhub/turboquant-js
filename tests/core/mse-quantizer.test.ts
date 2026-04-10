import { describe, it, expect } from 'vitest';
import { TurboQuantMSE } from '../../src/core/mse-quantizer.js';
import { norm, sub, normalize, dot } from '../../src/math/vec.js';
import { createPRNG } from '../../src/rng/xorshift128.js';

function randomUnitVector(d: number, rng: ReturnType<typeof createPRNG>): Float64Array {
  const v = new Float64Array(d);
  for (let i = 0; i < d; i++) v[i] = rng.nextGaussian();
  return normalize(v);
}

describe('TurboQuantMSE', () => {
  const d = 64;
  const bits = 2;
  const mse = new TurboQuantMSE({ dimension: d, bits, seed: 42 });

  it('quantize returns indices of correct length', () => {
    const x = randomUnitVector(d, createPRNG(1));
    const q = mse.quantize(x);
    expect(q.indices.length).toBe(d);
  });

  it('indices are in valid range [0, 2^bits - 1]', () => {
    const x = randomUnitVector(d, createPRNG(1));
    const q = mse.quantize(x);
    const maxIdx = (1 << bits) - 1;
    for (let i = 0; i < d; i++) {
      expect(q.indices[i]!).toBeGreaterThanOrEqual(0);
      expect(q.indices[i]!).toBeLessThanOrEqual(maxIdx);
    }
  });

  it('dequantize produces vector of same dimension', () => {
    const x = randomUnitVector(d, createPRNG(2));
    const q = mse.quantize(x);
    const xHat = mse.dequantize(q);
    expect(xHat.length).toBe(d);
  });

  it('round-trip MSE is bounded', () => {
    const rng = createPRNG(42);
    const nSamples = 100;
    let totalMse = 0;
    for (let i = 0; i < nSamples; i++) {
      const x = randomUnitVector(d, rng);
      const q = mse.quantize(x);
      const xHat = mse.dequantize(q);
      const error = sub(x, xHat);
      totalMse += dot(error, error);
    }
    const avgMse = totalMse / nSamples;
    // Should be reasonable — well below 1.0 for 2-bit on d=64
    expect(avgMse).toBeLessThan(1.0);
    expect(avgMse).toBeGreaterThan(0);
  });

  it('handles non-unit-norm vectors', () => {
    const x = new Float64Array(d);
    for (let i = 0; i < d; i++) x[i] = (i + 1) * 0.5;
    const originalNorm = norm(x);
    const q = mse.quantize(x);
    expect(q.norm).toBeCloseTo(originalNorm, 8);
    const xHat = mse.dequantize(q);
    // Norm is approximately preserved (quantization introduces some error)
    const normRatio = norm(xHat) / originalNorm;
    expect(normRatio).toBeGreaterThan(0.5);
    expect(normRatio).toBeLessThan(1.5);
  });

  it('unit-norm vectors do not store norm', () => {
    const x = randomUnitVector(d, createPRNG(5));
    const q = mse.quantize(x);
    expect(q.norm).toBeUndefined();
  });

  it('deterministic: same input gives same output', () => {
    const mse1 = new TurboQuantMSE({ dimension: d, bits, seed: 42 });
    const mse2 = new TurboQuantMSE({ dimension: d, bits, seed: 42 });
    const x = randomUnitVector(d, createPRNG(1));
    const q1 = mse1.quantize(x);
    const q2 = mse2.quantize(x);
    expect(Array.from(q1.indices)).toEqual(Array.from(q2.indices));
  });
});
