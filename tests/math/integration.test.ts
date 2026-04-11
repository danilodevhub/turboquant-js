import { describe, expect, it } from 'vitest';
import { adaptiveSimpson, simpson } from '../../src/math/integration.js';

describe('simpson', () => {
  it('integral of x^2 from 0 to 1 = 1/3', () => {
    const result = simpson((x) => x * x, 0, 1, 100);
    expect(result).toBeCloseTo(1 / 3, 10);
  });

  it('integral of sin(x) from 0 to pi = 2', () => {
    const result = simpson(Math.sin, 0, Math.PI, 1000);
    expect(result).toBeCloseTo(2, 10);
  });

  it('integral of constant = constant * width', () => {
    const result = simpson(() => 5, 2, 7);
    expect(result).toBeCloseTo(25, 10);
  });
});

describe('adaptiveSimpson', () => {
  it('integral of x^2 from 0 to 1 = 1/3', () => {
    const result = adaptiveSimpson((x) => x * x, 0, 1);
    expect(result).toBeCloseTo(1 / 3, 10);
  });

  it('integral of sin(x) from 0 to pi = 2', () => {
    const result = adaptiveSimpson(Math.sin, 0, Math.PI);
    expect(result).toBeCloseTo(2, 10);
  });

  it('handles sharp peak (Gaussian with small variance)', () => {
    const sigma = 0.01;
    const f = (x: number) => Math.exp(-0.5 * (x / sigma) ** 2) / (sigma * Math.sqrt(2 * Math.PI));
    const result = adaptiveSimpson(f, -1, 1, 1e-8);
    expect(result).toBeCloseTo(1, 5);
  });
});
