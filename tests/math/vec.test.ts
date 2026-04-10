import { describe, it, expect } from 'vitest';
import { dot, norm, normalize, add, sub, scale, sign, clone } from '../../src/math/vec.js';

describe('vec', () => {
  const a = new Float64Array([1, 2, 3]);
  const b = new Float64Array([4, 5, 6]);

  it('dot product', () => {
    expect(dot(a, b)).toBe(32); // 4+10+18
  });

  it('dot(a,a) === norm(a)^2', () => {
    expect(dot(a, a)).toBeCloseTo(norm(a) ** 2, 12);
  });

  it('norm', () => {
    expect(norm(a)).toBeCloseTo(Math.sqrt(14), 12);
  });

  it('normalize produces unit vector', () => {
    expect(norm(normalize(a))).toBeCloseTo(1, 12);
  });

  it('normalize zero vector returns zero', () => {
    const zero = new Float64Array(3);
    const n = normalize(zero);
    expect(norm(n)).toBe(0);
  });

  it('add', () => {
    const r = add(a, b);
    expect(Array.from(r)).toEqual([5, 7, 9]);
  });

  it('sub', () => {
    const r = sub(b, a);
    expect(Array.from(r)).toEqual([3, 3, 3]);
  });

  it('scale', () => {
    const r = scale(a, 2);
    expect(Array.from(r)).toEqual([2, 4, 6]);
  });

  it('sign: positive and negative', () => {
    const v = new Float64Array([-3, 0, 5, -0.1]);
    const s = sign(v);
    expect(Array.from(s)).toEqual([-1, 1, 1, -1]);
  });

  it('clone produces independent copy', () => {
    const c = clone(a);
    expect(Array.from(c)).toEqual(Array.from(a));
    c[0] = 999;
    expect(a[0]).toBe(1);
  });
});
