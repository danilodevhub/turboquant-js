import { type Mat, createMat, getEl, setEl } from './mat.js';

/**
 * Householder QR decomposition.
 * Returns Q (orthogonal) and R (upper triangular) such that A = Q * R.
 * For square matrices with Gaussian random entries, applying the sign
 * correction produces a Haar-distributed random orthogonal matrix.
 */
export function qr(a: Mat): { Q: Mat; R: Mat } {
  const m = a.rows;
  const n = a.cols;
  const k = Math.min(m, n);

  // Work on a copy (becomes R)
  const R = createMat(m, n, new Float64Array(a.data));

  // Accumulate Q = H_1 * H_2 * ... * H_k, start with identity
  const Q = createMat(m, m);
  for (let i = 0; i < m; i++) setEl(Q, i, i, 1);

  for (let j = 0; j < k; j++) {
    // Extract the column below the diagonal
    const colLen = m - j;
    const x = new Float64Array(colLen);
    for (let i = 0; i < colLen; i++) {
      x[i] = getEl(R, i + j, j);
    }

    // Compute the Householder vector
    let xNorm = 0;
    for (let i = 0; i < colLen; i++) xNorm += x[i]! * x[i]!;
    xNorm = Math.sqrt(xNorm);

    if (xNorm < 1e-15) continue;

    const s = x[0]! >= 0 ? -1 : 1;
    x[0]! -= s * xNorm;

    // Normalize v
    let vNorm = 0;
    for (let i = 0; i < colLen; i++) vNorm += x[i]! * x[i]!;
    vNorm = Math.sqrt(vNorm);
    if (vNorm < 1e-15) continue;
    for (let i = 0; i < colLen; i++) x[i]! /= vNorm;

    // Apply H = I - 2*v*v^T to R (columns j..n-1, rows j..m-1)
    for (let col = j; col < n; col++) {
      let dotVCol = 0;
      for (let i = 0; i < colLen; i++) {
        dotVCol += x[i]! * getEl(R, i + j, col);
      }
      for (let i = 0; i < colLen; i++) {
        setEl(R, i + j, col, getEl(R, i + j, col) - 2 * x[i]! * dotVCol);
      }
    }

    // Apply H to Q (all rows, columns j..m-1)
    for (let row = 0; row < m; row++) {
      let dotVRow = 0;
      for (let i = 0; i < colLen; i++) {
        dotVRow += x[i]! * getEl(Q, row, i + j);
      }
      for (let row2 = 0; row2 < colLen; row2++) {
        setEl(Q, row, row2 + j, getEl(Q, row, row2 + j) - 2 * x[row2]! * dotVRow);
      }
    }
  }

  // Sign correction: ensure R has positive diagonal entries.
  // This makes Q Haar-distributed when the input is Gaussian.
  for (let i = 0; i < k; i++) {
    if (getEl(R, i, i) < 0) {
      // Flip sign of R row i and Q column i
      for (let j = i; j < n; j++) {
        setEl(R, i, j, -getEl(R, i, j));
      }
      for (let row = 0; row < m; row++) {
        setEl(Q, row, i, -getEl(Q, row, i));
      }
    }
  }

  return { Q, R };
}
