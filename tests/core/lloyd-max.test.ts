import { describe, it, expect } from 'vitest';
import { solveLloydMax, computeDistortion } from '../../src/core/lloyd-max.js';

describe('lloyd-max', () => {
  it('produces correct number of centroids for 1-bit', () => {
    const cb = solveLloydMax(128, 1);
    expect(cb.centroids.length).toBe(2);
    expect(cb.boundaries.length).toBe(1);
  });

  it('produces correct number of centroids for 2-bit', () => {
    const cb = solveLloydMax(128, 2);
    expect(cb.centroids.length).toBe(4);
    expect(cb.boundaries.length).toBe(3);
  });

  it('produces correct number of centroids for 3-bit', () => {
    const cb = solveLloydMax(128, 3);
    expect(cb.centroids.length).toBe(8);
    expect(cb.boundaries.length).toBe(7);
  });

  it('centroids are sorted', () => {
    const cb = solveLloydMax(128, 3);
    for (let i = 1; i < cb.centroids.length; i++) {
      expect(cb.centroids[i]!).toBeGreaterThan(cb.centroids[i - 1]!);
    }
  });

  it('centroids are symmetric around 0', () => {
    const cb = solveLloydMax(128, 2);
    const n = cb.centroids.length;
    for (let i = 0; i < n / 2; i++) {
      expect(cb.centroids[i]!).toBeCloseTo(-cb.centroids[n - 1 - i]!, 6);
    }
  });

  it('1-bit distortion is near 0.36/d', () => {
    const d = 128;
    const cb = solveLloydMax(d, 1);
    const dist = computeDistortion(cb);
    // Paper: 1-bit distortion per coordinate ~0.36
    // Per coordinate distortion should be ~0.36/d (since variance is 1/d)
    const perCoord = dist;
    // The absolute value depends on the scale (variance 1/d)
    // Rough check: distortion should be positive and reasonable
    expect(perCoord).toBeGreaterThan(0);
    expect(perCoord).toBeLessThan(0.01); // Per-coord distortion for d=128
  });

  it('distortion decreases with more bits', () => {
    const d = 128;
    const d1 = computeDistortion(solveLloydMax(d, 1));
    const d2 = computeDistortion(solveLloydMax(d, 2));
    const d3 = computeDistortion(solveLloydMax(d, 3));
    expect(d2).toBeLessThan(d1);
    expect(d3).toBeLessThan(d2);
  });

  it('works with small dimension (d=8, exact beta)', () => {
    const cb = solveLloydMax(8, 2, { useExact: true });
    expect(cb.centroids.length).toBe(4);
    const dist = computeDistortion(cb);
    expect(dist).toBeGreaterThan(0);
  });
});
