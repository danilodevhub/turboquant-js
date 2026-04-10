/** Dot product of two vectors. */
export function dot(a: Float64Array, b: Float64Array): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += a[i]! * b[i]!;
  }
  return sum;
}

/** L2 norm of a vector. */
export function norm(a: Float64Array): number {
  return Math.sqrt(dot(a, a));
}

/** Return a unit-length copy. */
export function normalize(a: Float64Array): Float64Array {
  const n = norm(a);
  if (n === 0) return new Float64Array(a.length);
  return scale(a, 1 / n);
}

/** Element-wise addition: a + b. */
export function add(a: Float64Array, b: Float64Array): Float64Array {
  const out = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) {
    out[i] = a[i]! + b[i]!;
  }
  return out;
}

/** Element-wise subtraction: a - b. */
export function sub(a: Float64Array, b: Float64Array): Float64Array {
  const out = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) {
    out[i] = a[i]! - b[i]!;
  }
  return out;
}

/** Scalar multiplication: a * s. */
export function scale(a: Float64Array, s: number): Float64Array {
  const out = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) {
    out[i] = a[i]! * s;
  }
  return out;
}

/** Element-wise sign: +1 for x >= 0, -1 for x < 0. */
export function sign(a: Float64Array): Float64Array {
  const out = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) {
    out[i] = a[i]! >= 0 ? 1 : -1;
  }
  return out;
}

/** Clone a vector. */
export function clone(a: Float64Array): Float64Array {
  return new Float64Array(a);
}
