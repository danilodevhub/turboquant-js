import type { PRNG } from './types.js';

/**
 * xorshift128+ PRNG.
 * Uses Uint32Array(4) to represent two 64-bit state words as pairs of 32-bit values.
 * Deterministic: same seed produces identical sequences across all JS runtimes.
 */
export function createPRNG(seed: number): PRNG {
  // Initialize state from seed using splitmix32-like seeding
  const state = new Uint32Array(4);
  let s = seed >>> 0;
  for (let i = 0; i < 4; i++) {
    s = (s + 0x9e3779b9) >>> 0;
    let z = s;
    z = (Math.imul(z ^ (z >>> 16), 0x85ebca6b)) >>> 0;
    z = (Math.imul(z ^ (z >>> 13), 0xc2b2ae35)) >>> 0;
    z = (z ^ (z >>> 16)) >>> 0;
    state[i] = z;
  }

  // Ensure state is not all zeros
  if (state[0] === 0 && state[1] === 0 && state[2] === 0 && state[3] === 0) {
    state[0] = 1;
  }

  // Cached gaussian value for Box-Muller pairs
  let hasSpare = false;
  let spare = 0;

  function next(): number {
    // s0 = state[0..1], s1 = state[2..3]
    let s1Lo = state[0]!;
    let s1Hi = state[1]!;
    const s0Lo = state[2]!;
    const s0Hi = state[3]!;

    // result = s0 + s1 (64-bit addition via 32-bit halves)
    const resultLo = (s0Lo + s1Lo) >>> 0;
    const carry = ((s0Lo >>> 1) + (s1Lo >>> 1) + ((s0Lo & s1Lo) & 1)) >>> 31;
    const resultHi = (s0Hi + s1Hi + carry) >>> 0;

    // state[0..1] = s0
    state[0] = s0Lo;
    state[1] = s0Hi;

    // s1 ^= s1 << 23 (shift left 23 of 64-bit value)
    const shiftedLo = s1Lo << 23;
    const shiftedHi = (s1Hi << 23) | (s1Lo >>> 9);
    s1Lo ^= shiftedLo;
    s1Hi ^= shiftedHi;

    // s1 = s1 ^ s0 ^ (s1 >> 17) ^ (s0 >> 26)
    // s1 >> 17 (logical right shift of 64-bit)
    const s1r17Lo = (s1Lo >>> 17) | (s1Hi << 15);
    const s1r17Hi = s1Hi >>> 17;
    // s0 >> 26
    const s0r26Lo = (s0Lo >>> 26) | (s0Hi << 6);
    const s0r26Hi = s0Hi >>> 26;

    state[2] = (s1Lo ^ s0Lo ^ s1r17Lo ^ s0r26Lo) >>> 0;
    state[3] = (s1Hi ^ s0Hi ^ s1r17Hi ^ s0r26Hi) >>> 0;

    // Convert to [0, 1): use upper 53 bits of the 64-bit result
    // Upper 32 bits contribute 2^21 values, lower 32 bits contribute the top 21 bits
    const upper = resultHi >>> 0;
    const lower21 = resultLo >>> 11;
    return (upper * 2097152 + lower21) / 9007199254740992; // 2^53
  }

  function nextGaussian(): number {
    if (hasSpare) {
      hasSpare = false;
      return spare;
    }

    let u: number, v: number, s: number;
    do {
      u = 2 * next() - 1;
      v = 2 * next() - 1;
      s = u * u + v * v;
    } while (s >= 1 || s === 0);

    const mul = Math.sqrt(-2 * Math.log(s) / s);
    spare = v * mul;
    hasSpare = true;
    return u * mul;
  }

  return { next, nextGaussian };
}
