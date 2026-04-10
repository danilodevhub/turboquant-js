import { coordinatePdf } from '../math/beta-pdf.js';
import { adaptiveSimpson } from '../math/integration.js';
import type { Codebook } from './types.js';

export interface LloydMaxOptions {
  maxIter?: number;
  tol?: number;
  useExact?: boolean;
}

/**
 * Solve the Lloyd-Max optimal scalar quantizer for the coordinate
 * distribution after random rotation of unit vectors in R^d.
 *
 * Uses iterative centroid/boundary updates with adaptive quadrature
 * for the conditional expectation integrals.
 */
export function solveLloydMax(d: number, bits: number, options?: LloydMaxOptions): Codebook {
  const { maxIter = 200, tol = 1e-10, useExact = false } = options ?? {};
  const numCentroids = 1 << bits;

  const pdf = (x: number) => coordinatePdf(x, d, useExact);

  // Determine the effective range: for Gaussian approx, use ±4*sigma
  const sigma = 1 / Math.sqrt(d);
  const rangeLimit = d < 64 ? 0.999 : 4 * sigma;

  // Initialize centroids uniformly across the range
  const centroids = new Float64Array(numCentroids);
  for (let i = 0; i < numCentroids; i++) {
    centroids[i] = -rangeLimit + (2 * rangeLimit * (i + 0.5)) / numCentroids;
  }

  const boundaries = new Float64Array(numCentroids - 1);

  for (let iter = 0; iter < maxIter; iter++) {
    // Update boundaries: midpoint of adjacent centroids
    for (let i = 0; i < numCentroids - 1; i++) {
      boundaries[i] = (centroids[i]! + centroids[i + 1]!) / 2;
    }

    // Update centroids: conditional expectation within each partition
    let maxDelta = 0;
    for (let i = 0; i < numCentroids; i++) {
      const lo = i === 0 ? -rangeLimit : boundaries[i - 1]!;
      const hi = i === numCentroids - 1 ? rangeLimit : boundaries[i]!;

      if (hi - lo < 1e-18) continue;

      const denom = adaptiveSimpson(pdf, lo, hi, tol * 0.1);

      if (denom < 1e-15) {
        // Negligible probability in this region — keep centroid at midpoint
        const newC = (lo + hi) / 2;
        maxDelta = Math.max(maxDelta, Math.abs(newC - centroids[i]!));
        centroids[i] = newC;
        continue;
      }

      const numer = adaptiveSimpson((x) => x * pdf(x), lo, hi, tol * 0.1);
      const newCentroid = numer / denom;
      maxDelta = Math.max(maxDelta, Math.abs(newCentroid - centroids[i]!));
      centroids[i] = newCentroid;
    }

    if (maxDelta < tol) break;
  }

  // Final boundary computation
  for (let i = 0; i < numCentroids - 1; i++) {
    boundaries[i] = (centroids[i]! + centroids[i + 1]!) / 2;
  }

  return { centroids, boundaries, bits, dimension: d };
}

/**
 * Compute the MSE distortion of a codebook.
 */
export function computeDistortion(codebook: Codebook): number {
  const { centroids, boundaries, dimension: d } = codebook;
  const numCentroids = centroids.length;
  const pdf = (x: number) => coordinatePdf(x, d);
  const sigma = 1 / Math.sqrt(d);
  const rangeLimit = d < 64 ? 0.999 : 4 * sigma;

  let distortion = 0;
  for (let i = 0; i < numCentroids; i++) {
    const lo = i === 0 ? -rangeLimit : boundaries[i - 1]!;
    const hi = i === numCentroids - 1 ? rangeLimit : boundaries[i]!;
    const c = centroids[i]!;
    distortion += adaptiveSimpson((x) => (x - c) ** 2 * pdf(x), lo, hi);
  }

  return distortion;
}
