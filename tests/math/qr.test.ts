import { describe, it, expect } from 'vitest';
import { qr } from '../../src/math/qr.js';
import { createMat, matMul, transpose, getEl, identity } from '../../src/math/mat.js';

function maxAbsDiff(a: Float64Array, b: Float64Array): number {
  let max = 0;
  for (let i = 0; i < a.length; i++) {
    max = Math.max(max, Math.abs(a[i]! - b[i]!));
  }
  return max;
}

describe('qr', () => {
  it('Q^T * Q = I for 4x4', () => {
    const data = new Float64Array([
      2, -1, 0, 1,
      1, 3, -2, 0,
      0, 1, 4, -1,
      -1, 0, 1, 2,
    ]);
    const m = createMat(4, 4, data);
    const { Q } = qr(m);
    const QtQ = matMul(transpose(Q), Q);
    const I = identity(4);
    expect(maxAbsDiff(QtQ.data, I.data)).toBeLessThan(1e-10);
  });

  it('Q * R reconstructs original matrix', () => {
    const data = new Float64Array([
      1, 2, 3,
      4, 5, 6,
      7, 8, 10,
    ]);
    const m = createMat(3, 3, data);
    const { Q, R } = qr(m);
    const reconstructed = matMul(Q, R);
    expect(maxAbsDiff(reconstructed.data, data)).toBeLessThan(1e-10);
  });

  it('R is upper triangular', () => {
    const data = new Float64Array([
      3, 1, 1,
      1, 3, 1,
      1, 1, 3,
    ]);
    const m = createMat(3, 3, data);
    const { R } = qr(m);
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < i; j++) {
        expect(Math.abs(getEl(R, i, j))).toBeLessThan(1e-10);
      }
    }
  });

  it('R has non-negative diagonal (sign correction)', () => {
    const data = new Float64Array([
      -2, 1, 0, 3,
      4, -1, 2, 1,
      0, 3, -1, 2,
      1, 0, 4, -2,
    ]);
    const m = createMat(4, 4, data);
    const { R } = qr(m);
    for (let i = 0; i < 4; i++) {
      expect(getEl(R, i, i)).toBeGreaterThanOrEqual(-1e-10);
    }
  });

  it('orthogonality at d=16', () => {
    // Pseudo-random deterministic matrix
    const d = 16;
    const data = new Float64Array(d * d);
    for (let i = 0; i < d * d; i++) {
      data[i] = Math.sin(i * 7.3 + 0.1) * 10;
    }
    const m = createMat(d, d, data);
    const { Q } = qr(m);
    const QtQ = matMul(transpose(Q), Q);
    const I = identity(d);
    expect(maxAbsDiff(QtQ.data, I.data)).toBeLessThan(1e-9);
  });
});
