/**
 * Lanczos approximation for the log-gamma function.
 * Uses g=7 with standard 9 coefficients. Accurate to ~15 significant digits.
 */
export function lnGamma(x: number): number {
  if (x <= 0) throw new RangeError('lnGamma requires x > 0');

  const g = 7;
  const c = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
    -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6,
    1.5056327351493116e-7,
  ] as const;

  if (x < 0.5) {
    // Reflection formula: Gamma(x)*Gamma(1-x) = pi/sin(pi*x)
    return Math.log(Math.PI / Math.sin(Math.PI * x)) - lnGamma(1 - x);
  }

  x -= 1;
  let sum = c[0]!;
  for (let i = 1; i < g + 2; i++) {
    sum += c[i]! / (x + i);
  }
  const t = x + g + 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(sum);
}

/** Gamma function via exp(lnGamma(x)). */
export function gamma(x: number): number {
  if (x <= 0 && Number.isInteger(x)) {
    throw new RangeError('Gamma is undefined for non-positive integers');
  }
  return Math.exp(lnGamma(x));
}
