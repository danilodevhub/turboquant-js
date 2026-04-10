import { describe, it, expect } from 'vitest';
import { betaPdf, gaussianApproxPdf, coordinatePdf } from '../../src/math/beta-pdf.js';
import { simpson } from '../../src/math/integration.js';

describe('betaPdf', () => {
  it('is zero outside (-1, 1)', () => {
    expect(betaPdf(1, 8)).toBe(0);
    expect(betaPdf(-1, 8)).toBe(0);
    expect(betaPdf(1.5, 8)).toBe(0);
  });

  it('is symmetric: f(x) = f(-x)', () => {
    for (const d of [4, 8, 32]) {
      expect(betaPdf(0.3, d)).toBeCloseTo(betaPdf(-0.3, d), 10);
      expect(betaPdf(0.7, d)).toBeCloseTo(betaPdf(-0.7, d), 10);
    }
  });

  it('integrates to 1 for d=8', () => {
    const integral = simpson((x) => betaPdf(x, 8), -0.999, 0.999, 2000);
    expect(integral).toBeCloseTo(1, 4);
  });

  it('integrates to 1 for d=32', () => {
    const integral = simpson((x) => betaPdf(x, 32), -0.999, 0.999, 2000);
    expect(integral).toBeCloseTo(1, 4);
  });

  it('integrates to 1 for d=128', () => {
    const integral = simpson((x) => betaPdf(x, 128), -0.5, 0.5, 2000);
    expect(integral).toBeCloseTo(1, 3);
  });
});

describe('gaussianApproxPdf', () => {
  it('integrates to ~1 for d=128', () => {
    const sigma = 1 / Math.sqrt(128);
    const integral = simpson((x) => gaussianApproxPdf(x, 128), -5 * sigma, 5 * sigma, 2000);
    expect(integral).toBeCloseTo(1, 3);
  });
});

describe('coordinatePdf', () => {
  it('uses exact for d < 64', () => {
    expect(coordinatePdf(0.1, 16)).toBeCloseTo(betaPdf(0.1, 16), 10);
  });

  it('uses gaussian for d >= 64 by default', () => {
    expect(coordinatePdf(0.01, 128)).toBeCloseTo(gaussianApproxPdf(0.01, 128), 10);
  });

  it('can force exact', () => {
    expect(coordinatePdf(0.01, 128, true)).toBeCloseTo(betaPdf(0.01, 128), 10);
  });
});
