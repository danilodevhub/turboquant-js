/** Row-major matrix stored as a flat Float64Array. */
export interface Mat {
  data: Float64Array;
  rows: number;
  cols: number;
}

/** Create a matrix from dimensions and data. */
export function createMat(rows: number, cols: number, data?: Float64Array): Mat {
  return { data: data ?? new Float64Array(rows * cols), rows, cols };
}

/** Identity matrix of size n. */
export function identity(n: number): Mat {
  const m = createMat(n, n);
  for (let i = 0; i < n; i++) {
    m.data[i * n + i] = 1;
  }
  return m;
}

/** Matrix-vector product: M * v. */
export function matVec(m: Mat, v: Float64Array): Float64Array {
  const out = new Float64Array(m.rows);
  for (let i = 0; i < m.rows; i++) {
    let sum = 0;
    const rowOff = i * m.cols;
    for (let j = 0; j < m.cols; j++) {
      sum += m.data[rowOff + j]! * v[j]!;
    }
    out[i] = sum;
  }
  return out;
}

/** Matrix-matrix product: A * B. */
export function matMul(a: Mat, b: Mat): Mat {
  const out = createMat(a.rows, b.cols);
  for (let i = 0; i < a.rows; i++) {
    for (let k = 0; k < a.cols; k++) {
      const aik = a.data[i * a.cols + k]!;
      if (aik === 0) continue;
      for (let j = 0; j < b.cols; j++) {
        out.data[i * b.cols + j]! += aik * b.data[k * b.cols + j]!;
      }
    }
  }
  return out;
}

/** Transpose a matrix. */
export function transpose(m: Mat): Mat {
  const out = createMat(m.cols, m.rows);
  for (let i = 0; i < m.rows; i++) {
    for (let j = 0; j < m.cols; j++) {
      out.data[j * m.rows + i] = m.data[i * m.cols + j]!;
    }
  }
  return out;
}

/** Get element at (row, col). */
export function getEl(m: Mat, row: number, col: number): number {
  return m.data[row * m.cols + col]!;
}

/** Set element at (row, col). */
export function setEl(m: Mat, row: number, col: number, val: number): void {
  m.data[row * m.cols + col] = val;
}

/** Build a matrix from column vectors. */
export function fromColumns(cols: Float64Array[]): Mat {
  const rows = cols[0]!.length;
  const nCols = cols.length;
  const m = createMat(rows, nCols);
  for (let j = 0; j < nCols; j++) {
    const col = cols[j]!;
    for (let i = 0; i < rows; i++) {
      m.data[i * nCols + j] = col[i]!;
    }
  }
  return m;
}
