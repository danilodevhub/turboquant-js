// TextEncoder/TextDecoder are available in all modern runtimes (Node.js, browsers, Deno, Bun)
// but not included in the ES2022 TypeScript lib. Declare them here to avoid pulling in DOM types.
declare class TextEncoder {
  encode(input?: string): Uint8Array;
}

declare class TextDecoder {
  decode(input?: ArrayBufferView | ArrayBuffer): string;
}
