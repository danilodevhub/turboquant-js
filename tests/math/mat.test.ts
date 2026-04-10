import { describe, it, expect } from 'vitest';
import { createMat, identity, matVec, matMul, transpose, getEl, setEl, fromColumns } from '../../src/math/mat.js';

describe('mat', () => {
  it('identity * vector = vector', () => {
    const v = new Float64Array([1, 2, 3]);
    const I = identity(3);
    const result = matVec(I, v);
    expect(Array.from(result)).toEqual([1, 2, 3]);
  });

  it('matVec with known 2x2', () => {
    // [[1, 2], [3, 4]] * [5, 6] = [17, 39]
    const m = createMat(2, 2, new Float64Array([1, 2, 3, 4]));
    const v = new Float64Array([5, 6]);
    const r = matVec(m, v);
    expect(r[0]).toBeCloseTo(17, 12);
    expect(r[1]).toBeCloseTo(39, 12);
  });

  it('matMul with known 2x2', () => {
    const a = createMat(2, 2, new Float64Array([1, 2, 3, 4]));
    const b = createMat(2, 2, new Float64Array([5, 6, 7, 8]));
    const c = matMul(a, b);
    // [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]
    expect(getEl(c, 0, 0)).toBeCloseTo(19, 12);
    expect(getEl(c, 0, 1)).toBeCloseTo(22, 12);
    expect(getEl(c, 1, 0)).toBeCloseTo(43, 12);
    expect(getEl(c, 1, 1)).toBeCloseTo(50, 12);
  });

  it('transpose', () => {
    const m = createMat(2, 3, new Float64Array([1, 2, 3, 4, 5, 6]));
    const t = transpose(m);
    expect(t.rows).toBe(3);
    expect(t.cols).toBe(2);
    expect(getEl(t, 0, 0)).toBe(1);
    expect(getEl(t, 0, 1)).toBe(4);
    expect(getEl(t, 2, 1)).toBe(6);
  });

  it('getEl and setEl', () => {
    const m = createMat(2, 2);
    setEl(m, 1, 0, 42);
    expect(getEl(m, 1, 0)).toBe(42);
  });

  it('fromColumns', () => {
    const c1 = new Float64Array([1, 2]);
    const c2 = new Float64Array([3, 4]);
    const m = fromColumns([c1, c2]);
    expect(getEl(m, 0, 0)).toBe(1);
    expect(getEl(m, 0, 1)).toBe(3);
    expect(getEl(m, 1, 0)).toBe(2);
    expect(getEl(m, 1, 1)).toBe(4);
  });
});
