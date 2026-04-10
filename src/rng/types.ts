export interface PRNG {
  /** Returns a uniform random number in [0, 1). */
  next(): number;
  /** Returns a Gaussian N(0, 1) sample via Box-Muller. */
  nextGaussian(): number;
}
