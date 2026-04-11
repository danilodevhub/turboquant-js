export { betaPdf, coordinatePdf, gaussianApproxPdf } from './beta-pdf.js';
export { packBits, packIndices, unpackBits, unpackIndices } from './bit-pack.js';
export { gamma, lnGamma } from './gamma.js';
export { adaptiveSimpson, simpson } from './integration.js';
export {
  createMat,
  fromColumns,
  getEl,
  identity,
  type Mat,
  matMul,
  matVec,
  setEl,
  transpose,
} from './mat.js';
export { qr } from './qr.js';
export { add, clone, dot, norm, normalize, scale, sign, sub } from './vec.js';
