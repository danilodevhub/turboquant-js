import { fwht, nextPow2 } from '../math/hadamard.js';
import { createPRNG } from '../rng/xorshift128.js';

export interface Rotation {
  /** Apply rotation: y = R * x */
  rotate(x: Float64Array): Float64Array;
  /** Undo rotation: x = R^T * y */
  unrotate(y: Float64Array): Float64Array;
  /** The dimension of the input vectors. */
  readonly dimension: number;
}

const NUM_ROUNDS = 3;

/**
 * Create a Randomized Hadamard Transform (RHT) rotation of dimension d.
 * Uses NUM_ROUNDS rounds of (diagonal sign flip + Hadamard transform)
 * for O(d log d) application instead of O(d^2) dense matrix multiply.
 */
export function createRotation(d: number, seed: number): Rotation {
  const rng = createPRNG(seed);
  const m = nextPow2(d); // padded dimension
  const invSqrtM = 1 / Math.sqrt(m);

  // Generate random sign vectors for each round
  const signs: Float64Array[] = [];
  for (let r = 0; r < NUM_ROUNDS; r++) {
    const s = new Float64Array(m);
    for (let i = 0; i < m; i++) {
      s[i] = rng.next() < 0.5 ? -1 : 1;
    }
    signs.push(s);
  }

  return {
    dimension: d,

    rotate(x: Float64Array): Float64Array {
      // Pad to power of 2
      const buf = new Float64Array(m);
      buf.set(x);

      // Apply NUM_ROUNDS rounds of D * H
      for (let r = 0; r < NUM_ROUNDS; r++) {
        // Apply diagonal sign flip D_r
        const s = signs[r]!;
        for (let i = 0; i < m; i++) {
          buf[i] *= s[i]!;
        }
        // Apply Walsh-Hadamard and normalize
        fwht(buf);
        for (let i = 0; i < m; i++) {
          buf[i] *= invSqrtM;
        }
      }

      // Return first d elements (unpad)
      return buf.subarray(0, d).slice();
    },

    unrotate(y: Float64Array): Float64Array {
      // Pad to power of 2
      const buf = new Float64Array(m);
      buf.set(y);

      // Apply rounds in reverse order
      for (let r = NUM_ROUNDS - 1; r >= 0; r--) {
        // Apply Hadamard (self-inverse up to scaling) and normalize
        fwht(buf);
        for (let i = 0; i < m; i++) {
          buf[i] *= invSqrtM;
        }
        // Apply diagonal sign flip D_r (sign matrices are self-inverse)
        const s = signs[r]!;
        for (let i = 0; i < m; i++) {
          buf[i] *= s[i]!;
        }
      }

      // Return first d elements
      return buf.subarray(0, d).slice();
    },
  };
}
