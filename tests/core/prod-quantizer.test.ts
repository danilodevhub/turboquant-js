import { describe, it, expect } from 'vitest';
import { TurboQuantProd } from '../../src/core/prod-quantizer.js';
import { dot, normalize, norm } from '../../src/math/vec.js';
import { createPRNG } from '../../src/rng/xorshift128.js';

function randomUnitVector(d: number, rng: ReturnType<typeof createPRNG>): Float64Array {
  const v = new Float64Array(d);
  for (let i = 0; i < d; i++) v[i] = rng.nextGaussian();
  return normalize(v);
}

describe('TurboQuantProd', () => {
  const d = 64;
  const bits = 3; // 2-bit MSE + 1-bit QJL

  it('throws for bits < 2', () => {
    expect(() => new TurboQuantProd({ dimension: d, bits: 1 })).toThrow();
  });

  it('quantize returns correct structure', () => {
    const tq = new TurboQuantProd({ dimension: d, bits, seed: 42 });
    const x = randomUnitVector(d, createPRNG(1));
    const q = tq.quantize(x);
    expect(q.indices.length).toBe(d);
    expect(q.qjlBits.length).toBe(Math.ceil(d / 8));
    expect(q.residualNorm).toBeGreaterThan(0);
  });

  it('innerProduct is unbiased (critical test)', () => {
    const tq = new TurboQuantProd({ dimension: d, bits, seed: 42 });
    const rng = createPRNG(123);
    const nSamples = 500;
    let totalError = 0;

    for (let i = 0; i < nSamples; i++) {
      const x = randomUnitVector(d, rng);
      const y = randomUnitVector(d, rng);
      const q = tq.quantize(x);
      const estimated = tq.innerProduct(y, q);
      const exact = dot(y, x);
      totalError += estimated - exact;
    }

    const meanError = totalError / nSamples;
    // Unbiased: mean error should be approximately 0
    // With 500 samples, allow some statistical variance
    expect(Math.abs(meanError)).toBeLessThan(0.1);
  });

  it('innerProduct error variance is bounded', () => {
    const tq = new TurboQuantProd({ dimension: d, bits, seed: 42 });
    const rng = createPRNG(456);
    const nSamples = 200;
    let totalSqError = 0;

    for (let i = 0; i < nSamples; i++) {
      const x = randomUnitVector(d, rng);
      const y = randomUnitVector(d, rng);
      const q = tq.quantize(x);
      const estimated = tq.innerProduct(y, q);
      const exact = dot(y, x);
      totalSqError += (estimated - exact) ** 2;
    }

    const mse = totalSqError / nSamples;
    // MSE should be small for 3-bit quantization
    expect(mse).toBeLessThan(0.5);
  });

  it('handles non-unit-norm vectors', () => {
    const tq = new TurboQuantProd({ dimension: d, bits, seed: 42 });
    const x = new Float64Array(d);
    for (let i = 0; i < d; i++) x[i] = (i + 1) * 0.3;
    const q = tq.quantize(x);
    expect(q.norm).toBeDefined();
    expect(q.norm).toBeCloseTo(norm(x), 5);
  });

  it('deterministic: same input gives same output', () => {
    const tq1 = new TurboQuantProd({ dimension: d, bits, seed: 42 });
    const tq2 = new TurboQuantProd({ dimension: d, bits, seed: 42 });
    const x = randomUnitVector(d, createPRNG(1));
    const y = randomUnitVector(d, createPRNG(2));
    const q1 = tq1.quantize(x);
    const q2 = tq2.quantize(x);
    expect(tq1.innerProduct(y, q1)).toBe(tq2.innerProduct(y, q2));
  });

  it('dequantize produces vector of correct dimension', () => {
    const tq = new TurboQuantProd({ dimension: d, bits, seed: 42 });
    const x = randomUnitVector(d, createPRNG(1));
    const q = tq.quantize(x);
    const xHat = tq.dequantize(q);
    expect(xHat.length).toBe(d);
  });
});
