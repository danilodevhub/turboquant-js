import { createMat, type Mat, matVec, transpose } from '../math/mat.js';
import { qr } from '../math/qr.js';
import { createPRNG } from '../rng/xorshift128.js';

export interface Rotation {
  /** Apply rotation: y = Pi * x */
  rotate(x: Float64Array): Float64Array;
  /** Undo rotation: x = Pi^T * y */
  unrotate(y: Float64Array): Float64Array;
  /** The orthogonal rotation matrix. */
  readonly matrix: Mat;
}

/**
 * Create a random orthogonal rotation matrix of dimension d.
 * Uses QR decomposition of a Gaussian random matrix to produce
 * a Haar-distributed (uniformly random) orthogonal matrix.
 */
export function createRotation(d: number, seed: number): Rotation {
  const rng = createPRNG(seed);

  // Fill a d×d matrix with N(0,1) entries
  const data = new Float64Array(d * d);
  for (let i = 0; i < d * d; i++) {
    data[i] = rng.nextGaussian();
  }
  const gaussianMat = createMat(d, d, data);

  // QR decomposition with sign correction → Haar-distributed Q
  const { Q } = qr(gaussianMat);
  const Qt = transpose(Q);

  return {
    rotate(x: Float64Array): Float64Array {
      return matVec(Q, x);
    },
    unrotate(y: Float64Array): Float64Array {
      return matVec(Qt, y);
    },
    get matrix() {
      return Q;
    },
  };
}
