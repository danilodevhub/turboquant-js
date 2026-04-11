import { type LloydMaxOptions, solveLloydMax } from './lloyd-max.js';
import type { Codebook } from './types.js';

const cache = new Map<string, Codebook>();

/**
 * Get or compute an optimal codebook for the given dimension and bit-width.
 * Results are cached by (dimension, bits) key.
 */
export function getCodebook(d: number, bits: number, options?: LloydMaxOptions): Codebook {
  const key = `${d}:${bits}`;
  const cached = cache.get(key);
  if (cached) return cached;

  const codebook = solveLloydMax(d, bits, options);
  cache.set(key, codebook);
  return codebook;
}

/** Clear the codebook cache (useful for testing). */
export function clearCodebookCache(): void {
  cache.clear();
}
