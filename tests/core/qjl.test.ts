import { describe, it, expect } from 'vitest';
import { QJL } from '../../src/core/qjl.js';
import { normalize, dot } from '../../src/math/vec.js';
import { createPRNG } from '../../src/rng/xorshift128.js';

describe('QJL', () => {
  const d = 32;
  const qjl = new QJL(d, 42);

  it('project returns signs of correct length', () => {
    const r = new Float64Array(d);
    for (let i = 0; i < d; i++) r[i] = Math.sin(i);
    const signs = qjl.project(r);
    expect(signs.length).toBe(d);
  });

  it('all signs are +1 or -1', () => {
    const r = new Float64Array(d);
    for (let i = 0; i < d; i++) r[i] = Math.cos(i * 2.3);
    const signs = qjl.project(r);
    for (let i = 0; i < d; i++) {
      expect(signs[i] === 1 || signs[i] === -1).toBe(true);
    }
  });

  it('deterministic: same input gives same signs', () => {
    const qjl1 = new QJL(d, 42);
    const qjl2 = new QJL(d, 42);
    const r = new Float64Array(d);
    for (let i = 0; i < d; i++) r[i] = i * 0.1;
    expect(Array.from(qjl1.project(r))).toEqual(Array.from(qjl2.project(r)));
  });

  it('innerProductCorrection produces finite values', () => {
    const rng = createPRNG(7);
    const r = new Float64Array(d);
    for (let i = 0; i < d; i++) r[i] = rng.nextGaussian() * 0.1;
    const signs = qjl.project(r);
    const y = normalize(new Float64Array(d).map((_, i) => Math.sin(i)));
    const correction = qjl.innerProductCorrection(signs, 0.5, y);
    expect(Number.isFinite(correction)).toBe(true);
  });

  it('correction is 0 when residualNorm is 0', () => {
    const signs = new Int8Array(d).fill(1);
    const y = new Float64Array(d).fill(1);
    const correction = qjl.innerProductCorrection(signs, 0, y);
    expect(correction).toBeCloseTo(0, 12);
  });
});
