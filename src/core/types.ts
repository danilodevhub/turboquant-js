export interface TurboQuantConfig {
  /** Vector dimension. Must be >= 2. */
  dimension: number;
  /** Bit-width per coordinate. Typically 2-4. */
  bits: number;
  /** PRNG seed for deterministic rotation/projection matrices. Default: 42. */
  seed?: number;
}

export interface QuantizedMSE {
  /** Per-coordinate codebook indices (b-bit values stored as uint8). */
  indices: Uint8Array;
  /** Original vector L2 norm, stored only if the input was not unit-norm. */
  norm?: number;
}

export interface QuantizedProd {
  /** Per-coordinate (b-1)-bit MSE stage indices. */
  indices: Uint8Array;
  /** Bit-packed QJL sign array (1 bit per coordinate). */
  qjlBits: Uint8Array;
  /** L2 norm of the residual (float scalar). */
  residualNorm: number;
  /** Original vector L2 norm, stored only if the input was not unit-norm. */
  norm?: number;
}

export interface Codebook {
  /** Sorted centroid values. Length = 2^bits. */
  centroids: Float64Array;
  /** Decision boundaries between adjacent centroids. Length = 2^bits - 1. */
  boundaries: Float64Array;
  /** Bit-width this codebook was solved for. */
  bits: number;
  /** Dimension used for the PDF during optimization. */
  dimension: number;
}
