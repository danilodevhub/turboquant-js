import { describe, it, expect } from 'vitest';
import { createRotation } from '../../src/core/rotation.js';
import { matMul, transpose, identity } from '../../src/math/mat.js';
import { norm, sub } from '../../src/math/vec.js';

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

  it('rotation matrix is orthogonal', () => {
    const d = 16;
    const rot = createRotation(d, 99);
    const Q = rot.matrix;
    const QtQ = matMul(transpose(Q), Q);
    const I = identity(d);
    expect(maxAbsDiff(QtQ.data, I.data)).toBeLessThan(1e-9);
  });

  it('deterministic: same seed gives same rotation', () => {
    const rot1 = createRotation(8, 42);
    const rot2 = createRotation(8, 42);
    const x = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8]);
    const y1 = rot1.rotate(x);
    const y2 = rot2.rotate(x);
    expect(maxAbsDiff(y1, y2)).toBe(0);
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
});
