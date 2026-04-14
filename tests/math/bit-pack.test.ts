import { describe, expect, it } from 'vitest';
import { packBits, packIndices, unpackBits, unpackIndices } from '../../src/math/bit-pack.js';

describe('packBits / unpackBits', () => {
  it('round-trip identity', () => {
    const signs = new Int8Array([1, -1, 1, 1, -1, -1, 1, -1, 1, 1]);
    const packed = packBits(signs);
    const unpacked = unpackBits(packed, signs.length);
    expect(Array.from(unpacked)).toEqual(Array.from(signs));
  });

  it('all positive', () => {
    const signs = new Int8Array([1, 1, 1, 1, 1, 1, 1, 1]);
    const packed = packBits(signs);
    expect(packed[0]).toBe(0xff);
    const unpacked = unpackBits(packed, 8);
    expect(Array.from(unpacked)).toEqual(Array.from(signs));
  });

  it('all negative', () => {
    const signs = new Int8Array([-1, -1, -1, -1, -1, -1, -1, -1]);
    const packed = packBits(signs);
    expect(packed[0]).toBe(0x00);
    const unpacked = unpackBits(packed, 8);
    expect(Array.from(unpacked)).toEqual(Array.from(signs));
  });

  it('non-byte-aligned length', () => {
    const signs = new Int8Array([1, -1, 1]);
    const packed = packBits(signs);
    const unpacked = unpackBits(packed, 3);
    expect(Array.from(unpacked)).toEqual([1, -1, 1]);
  });
});

describe('packIndices / unpackIndices', () => {
  it('round-trip 2-bit', () => {
    const indices = new Uint8Array([0, 1, 2, 3, 0, 2, 1, 3]);
    const packed = packIndices(indices, 2);
    const unpacked = unpackIndices(packed, 2, indices.length);
    expect(Array.from(unpacked)).toEqual(Array.from(indices));
  });

  it('round-trip 3-bit', () => {
    const indices = new Uint8Array([0, 1, 2, 3, 4, 5, 6, 7]);
    const packed = packIndices(indices, 3);
    const unpacked = unpackIndices(packed, 3, indices.length);
    expect(Array.from(unpacked)).toEqual(Array.from(indices));
  });

  it('round-trip 4-bit', () => {
    const indices = new Uint8Array([0, 5, 10, 15, 3, 7, 12, 1]);
    const packed = packIndices(indices, 4);
    const unpacked = unpackIndices(packed, 4, indices.length);
    expect(Array.from(unpacked)).toEqual(Array.from(indices));
  });

  it('round-trip 1-bit', () => {
    const indices = new Uint8Array([0, 1, 1, 0, 1, 0, 0, 1, 1]);
    const packed = packIndices(indices, 1);
    const unpacked = unpackIndices(packed, 1, indices.length);
    expect(Array.from(unpacked)).toEqual(Array.from(indices));
  });

  describe('roundtrip at all bit-widths [1..8]', () => {
    const lengths = [1, 7, 8, 9, 16, 33, 64, 100];

    for (const bits of [1, 2, 3, 4, 5, 6, 7, 8]) {
      for (const len of lengths) {
        it(`bits=${bits}, length=${len}`, () => {
          const maxVal = (1 << bits) - 1;
          const indices = new Uint8Array(len);
          for (let i = 0; i < len; i++) {
            indices[i] = i % (maxVal + 1);
          }
          const packed = packIndices(indices, bits);
          const unpacked = unpackIndices(packed, bits, len);
          expect(Array.from(unpacked)).toEqual(Array.from(indices));
        });
      }
    }
  });
});
