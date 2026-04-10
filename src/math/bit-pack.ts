/**
 * Pack an array of sign values (+1/-1 or 1/0) into a bit-packed Uint8Array.
 * Bit 0 of byte 0 holds signs[0], bit 7 of byte 0 holds signs[7], etc.
 */
export function packBits(signs: Int8Array): Uint8Array {
  const numBytes = Math.ceil(signs.length / 8);
  const packed = new Uint8Array(numBytes);
  for (let i = 0; i < signs.length; i++) {
    // Positive sign (>= 0) → bit = 1, negative → bit = 0
    if (signs[i]! >= 0) {
      packed[i >> 3]! |= 1 << (i & 7);
    }
  }
  return packed;
}

/**
 * Unpack bit-packed signs back to an Int8Array of +1/-1.
 */
export function unpackBits(packed: Uint8Array, length: number): Int8Array {
  const signs = new Int8Array(length);
  for (let i = 0; i < length; i++) {
    signs[i] = (packed[i >> 3]! >> (i & 7)) & 1 ? 1 : -1;
  }
  return signs;
}

/**
 * Pack an array of b-bit unsigned indices tightly.
 * Each index is in [0, 2^bits - 1].
 */
export function packIndices(indices: Uint8Array, bits: number): Uint8Array {
  const totalBits = indices.length * bits;
  const numBytes = Math.ceil(totalBits / 8);
  const packed = new Uint8Array(numBytes);

  let bitPos = 0;
  for (let i = 0; i < indices.length; i++) {
    let val = indices[i]!;
    let bitsLeft = bits;
    while (bitsLeft > 0) {
      const byteIdx = bitPos >> 3;
      const bitOffset = bitPos & 7;
      const bitsInThisByte = Math.min(bitsLeft, 8 - bitOffset);
      const mask = (1 << bitsInThisByte) - 1;
      packed[byteIdx]! |= (val & mask) << bitOffset;
      val >>= bitsInThisByte;
      bitPos += bitsInThisByte;
      bitsLeft -= bitsInThisByte;
    }
  }
  return packed;
}

/**
 * Unpack tightly packed b-bit indices.
 */
export function unpackIndices(packed: Uint8Array, bits: number, length: number): Uint8Array {
  const indices = new Uint8Array(length);
  let bitPos = 0;
  for (let i = 0; i < length; i++) {
    let val = 0;
    let bitsLeft = bits;
    let shift = 0;
    while (bitsLeft > 0) {
      const byteIdx = bitPos >> 3;
      const bitOffset = bitPos & 7;
      const bitsInThisByte = Math.min(bitsLeft, 8 - bitOffset);
      const mask = (1 << bitsInThisByte) - 1;
      val |= ((packed[byteIdx]! >> bitOffset) & mask) << shift;
      shift += bitsInThisByte;
      bitPos += bitsInThisByte;
      bitsLeft -= bitsInThisByte;
    }
    indices[i] = val;
  }
  return indices;
}
