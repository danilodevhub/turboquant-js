/**
 * Composite Simpson's rule for integrating f over [a, b].
 * @param n Number of subintervals (must be even, will be rounded up).
 */
export function simpson(f: (x: number) => number, a: number, b: number, n = 1000): number {
  if (n % 2 !== 0) n++;
  const h = (b - a) / n;
  let sum = f(a) + f(b);
  for (let i = 1; i < n; i++) {
    const x = a + i * h;
    sum += (i % 2 === 0 ? 2 : 4) * f(x);
  }
  return (h / 3) * sum;
}

/**
 * Adaptive Simpson's rule. Recursively refines until the error
 * estimate is below `tol`.
 */
export function adaptiveSimpson(
  f: (x: number) => number,
  a: number,
  b: number,
  tol = 1e-12,
): number {
  const maxDepth = 50;

  function simpsonSingle(a: number, b: number): number {
    const mid = (a + b) / 2;
    return ((b - a) / 6) * (f(a) + 4 * f(mid) + f(b));
  }

  function recurse(a: number, b: number, whole: number, depth: number): number {
    const mid = (a + b) / 2;
    const left = simpsonSingle(a, mid);
    const right = simpsonSingle(mid, b);
    const refined = left + right;
    if (depth >= maxDepth || Math.abs(refined - whole) <= 15 * tol) {
      return refined + (refined - whole) / 15;
    }
    return recurse(a, mid, left, depth + 1) + recurse(mid, b, right, depth + 1);
  }

  return recurse(a, b, simpsonSingle(a, b), 0);
}
