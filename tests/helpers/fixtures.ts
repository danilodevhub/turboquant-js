/**
 * Small-dimension test fixtures for regression testing.
 * These provide known-correct values to anchor the implementation.
 */

/** A known 8-dimensional unit vector for testing. */
export const FIXTURE_VECTOR_8D = new Float64Array([
  0.35355339, 0.35355339, 0.35355339, 0.35355339,
  0.35355339, 0.35355339, 0.35355339, 0.35355339,
]);
// norm = 1.0 (1/sqrt(8) * sqrt(8) = 1)

/** A second 8-dimensional unit vector (orthogonal direction). */
export const FIXTURE_QUERY_8D = new Float64Array([
  0.5, 0.5, 0.5, 0.5,
  -0.5, -0.5, -0.5, -0.5,
]);
// norm = sqrt(8*0.25) = sqrt(2) ≈ 1.414, not unit — will be normalized by cosine metric

/** Expected inner product of FIXTURE_VECTOR_8D and FIXTURE_QUERY_8D. */
export const FIXTURE_INNER_PRODUCT = 0; // orthogonal: 4*0.5*0.353.. + 4*(-0.5)*0.353.. = 0
