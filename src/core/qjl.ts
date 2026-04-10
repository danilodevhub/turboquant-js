import { createMat, matVec, type Mat } from '../math/mat.js';
import { createPRNG } from '../rng/xorshift128.js';

/**
 * Quantized Johnson-Lindenstrauss (QJL) transform.
 * Stage 2 of TurboQuant: applies a random projection then sign-bit quantizes.
 * Provides unbiased inner product estimation with 1 bit per coordinate.
 */
export class QJL {
  readonly dimension: number;
  private readonly projMatrix: Mat;

  constructor(d: number, seed: number) {
    this.dimension = d;
    const rng = createPRNG(seed);

    // Generate d×d random projection matrix with i.i.d. N(0,1) entries
    const data = new Float64Array(d * d);
    for (let i = 0; i < d * d; i++) {
      data[i] = rng.nextGaussian();
    }
    this.projMatrix = createMat(d, d, data);
  }

  /**
   * Project and sign-quantize a residual vector.
   * Returns sign(S * r) as Int8Array of +1/-1.
   */
  project(r: Float64Array): Int8Array {
    const projected = matVec(this.projMatrix, r);
    const signs = new Int8Array(this.dimension);
    for (let i = 0; i < this.dimension; i++) {
      signs[i] = projected[i]! >= 0 ? 1 : -1;
    }
    return signs;
  }

  /**
   * Compute the QJL inner product correction term.
   *
   * Instead of materializing S^T * qjlSigns and dotting with y,
   * we compute S * y and dot with qjlSigns.
   * Both equal qjlSigns^T * S * y.
   *
   * Returns: sqrt(pi/2) / d * residualNorm * sum_j(sign_j * (S*y)_j)
   */
  innerProductCorrection(
    qjlSigns: Int8Array,
    residualNorm: number,
    y: Float64Array,
  ): number {
    const sy = matVec(this.projMatrix, y);
    let dotProduct = 0;
    for (let i = 0; i < this.dimension; i++) {
      dotProduct += qjlSigns[i]! * sy[i]!;
    }
    return (Math.sqrt(Math.PI / 2) / this.dimension) * residualNorm * dotProduct;
  }
}
