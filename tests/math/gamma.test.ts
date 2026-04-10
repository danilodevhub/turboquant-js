import { describe, it, expect } from 'vitest';
import { gamma, lnGamma } from '../../src/math/gamma.js';

describe('gamma', () => {
  it('gamma(1) = 1', () => {
    expect(gamma(1)).toBeCloseTo(1, 12);
  });

  it('gamma(0.5) = sqrt(pi)', () => {
    expect(gamma(0.5)).toBeCloseTo(Math.sqrt(Math.PI), 10);
  });

  it('gamma(5) = 24', () => {
    expect(gamma(5)).toBeCloseTo(24, 10);
  });

  it('gamma(10) = 362880', () => {
    expect(gamma(10)).toBeCloseTo(362880, 5);
  });

  it('gamma(2) = 1', () => {
    expect(gamma(2)).toBeCloseTo(1, 12);
  });

  it('gamma(3) = 2', () => {
    expect(gamma(3)).toBeCloseTo(2, 12);
  });

  it('throws for non-positive integers', () => {
    expect(() => gamma(0)).toThrow();
    expect(() => gamma(-1)).toThrow();
    expect(() => gamma(-3)).toThrow();
  });
});

describe('lnGamma', () => {
  it('lnGamma(1) = 0', () => {
    expect(lnGamma(1)).toBeCloseTo(0, 12);
  });

  it('lnGamma(large) does not overflow', () => {
    // Gamma(256) overflows float64, but lnGamma should be fine
    const result = lnGamma(256);
    expect(Number.isFinite(result)).toBe(true);
    expect(result).toBeGreaterThan(0);
  });

  it('exp(lnGamma(5)) = 24', () => {
    expect(Math.exp(lnGamma(5))).toBeCloseTo(24, 10);
  });

  it('throws for x <= 0', () => {
    expect(() => lnGamma(0)).toThrow();
    expect(() => lnGamma(-1)).toThrow();
  });
});
