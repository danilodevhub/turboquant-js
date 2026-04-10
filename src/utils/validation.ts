/** Convert number[] | Float64Array to Float64Array. */
export function toFloat64Array(input: Float64Array | number[]): Float64Array {
  return input instanceof Float64Array ? input : new Float64Array(input);
}
