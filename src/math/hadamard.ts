/**
 * In-place Fast Walsh-Hadamard Transform.
 * Input length must be a power of 2.
 */
export function fwht(x: Float64Array): void {
  const n = x.length;
  for (let h = 1; h < n; h *= 2) {
    for (let i = 0; i < n; i += h * 2) {
      for (let j = i; j < i + h; j++) {
        const a = x[j]!;
        const b = x[j + h]!;
        x[j] = a + b;
        x[j + h] = a - b;
      }
    }
  }
}

/** Return the smallest power of 2 that is >= n. */
export function nextPow2(n: number): number {
  let p = 1;
  while (p < n) p *= 2;
  return p;
}
