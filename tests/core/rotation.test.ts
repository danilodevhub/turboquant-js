import { describe, expect, it } from 'vitest';
import { createRotation } from '../../src/core/rotation.js';
import { norm } from '../../src/math/vec.js';

function maxAbsDiff(a: Float64Array, b: Float64Array): number {
  let max = 0;
  for (let i = 0; i < a.length; i++) max = Math.max(max, Math.abs(a[i]! - b[i]!));
  return max;
}

describe('rotation', () => {
  it('rotate then unrotate recovers original vector', () => {
    const rot = createRotation(16, 42);
    const x = new Float64Array(16);
    for (let i = 0; i < 16; i++) x[i] = Math.sin(i + 1);
    const y = rot.rotate(x);
    const xBack = rot.unrotate(y);
    expect(maxAbsDiff(x, xBack)).toBeLessThan(1e-10);
  });

  it('rotation preserves norm', () => {
    const rot = createRotation(32, 7);
    const x = new Float64Array(32);
    for (let i = 0; i < 32; i++) x[i] = i * 0.1 - 1.5;
    const y = rot.rotate(x);
    expect(norm(y)).toBeCloseTo(norm(x), 10);
  });

  it('deterministic: same seed gives same rotation', () => {
    const rot1 = createRotation(8, 42);
    const rot2 = createRotation(8, 42);
    const x = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8]);
    const y1 = rot1.rotate(x);
    const y2 = rot2.rotate(x);
    expect(maxAbsDiff(y1, y2)).toBe(0);
  });

  it('different seeds produce different results', () => {
    const rot1 = createRotation(16, 42);
    const rot2 = createRotation(16, 99);
    const x = new Float64Array(16);
    for (let i = 0; i < 16; i++) x[i] = Math.sin(i + 1);
    const y1 = rot1.rotate(x);
    const y2 = rot2.rotate(x);
    expect(maxAbsDiff(y1, y2)).toBeGreaterThan(0.01);
  });

  it('coordinate variance after rotation is ~1/d for unit vector', () => {
    const d = 64;
    const rot = createRotation(d, 42);
    // Unit vector along first axis
    const x = new Float64Array(d);
    x[0] = 1;
    const y = rot.rotate(x);
    // Each coordinate of y should be O(1/sqrt(d))
    let sumSq = 0;
    for (let i = 0; i < d; i++) sumSq += y[i]! * y[i]!;
    const variance = sumSq / d;
    expect(variance).toBeCloseTo(1 / d, 1);
  });

  it('rotated coordinates are approximately uniformly spread', () => {
    // For a unit vector, after RHT the energy should be spread across coordinates
    const d = 128;
    const rot = createRotation(d, 77);
    const x = new Float64Array(d);
    x[0] = 1;
    const y = rot.rotate(x);

    // Check that no single coordinate has too much energy
    // For a well-spread rotation, max |y_i| should be O(sqrt(log(d)/d))
    let maxAbs = 0;
    for (let i = 0; i < d; i++) {
      maxAbs = Math.max(maxAbs, Math.abs(y[i]!));
    }
    // With d=128, sqrt(log(128)/128) ~ 0.19, allow generous bound
    expect(maxAbs).toBeLessThan(0.5);
  });

  it('exposes dimension property', () => {
    const rot = createRotation(42, 1);
    expect(rot.dimension).toBe(42);
  });
});
