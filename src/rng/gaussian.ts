import type { PRNG } from './types.js';

/**
 * Fill a Float64Array with i.i.d. N(0,1) samples from the given PRNG.
 */
export function fillGaussian(prng: PRNG, out: Float64Array): void {
  for (let i = 0; i < out.length; i++) {
    out[i] = prng.nextGaussian();
  }
}

/**
 * Create a Float64Array of n i.i.d. N(0,1) samples.
 */
export function gaussianArray(prng: PRNG, n: number): Float64Array {
  const out = new Float64Array(n);
  fillGaussian(prng, out);
  return out;
}
