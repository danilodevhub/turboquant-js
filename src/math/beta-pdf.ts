import { lnGamma } from './gamma.js';

/**
 * Exact PDF of a coordinate after random rotation of a unit vector in R^d.
 * f(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^((d-3)/2)
 * Defined on (-1, 1).
 */
export function betaPdf(x: number, d: number): number {
  if (Math.abs(x) >= 1) return 0;
  const lnCoeff = lnGamma(d / 2) - lnGamma((d - 1) / 2) - 0.5 * Math.log(Math.PI);
  const exponent = (d - 3) / 2;
  return Math.exp(lnCoeff + exponent * Math.log(1 - x * x));
}

/**
 * Gaussian approximation N(0, 1/d) valid for d >= 64.
 */
export function gaussianApproxPdf(x: number, d: number): number {
  const variance = 1 / d;
  return Math.exp(-0.5 * x * x / variance) / Math.sqrt(2 * Math.PI * variance);
}

/**
 * Coordinate PDF dispatcher.
 * Uses exact Beta for d < 64, Gaussian approximation for d >= 64.
 */
export function coordinatePdf(x: number, d: number, useExact = false): number {
  if (useExact || d < 64) {
    return betaPdf(x, d);
  }
  return gaussianApproxPdf(x, d);
}
