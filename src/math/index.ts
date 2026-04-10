export { dot, norm, normalize, add, sub, scale, sign, clone } from './vec.js';
export { type Mat, createMat, identity, matVec, matMul, transpose, getEl, setEl, fromColumns } from './mat.js';
export { qr } from './qr.js';
export { gamma, lnGamma } from './gamma.js';
export { betaPdf, gaussianApproxPdf, coordinatePdf } from './beta-pdf.js';
export { simpson, adaptiveSimpson } from './integration.js';
export { packBits, unpackBits, packIndices, unpackIndices } from './bit-pack.js';
