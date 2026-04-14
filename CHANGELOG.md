# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-04-14

### Added

- **Randomized Hadamard Transform (RHT)** — O(d log d) rotation replacing O(d^3) dense QR decomposition
- **Serialization** — `toBuffer()` / `fromBuffer()` on `VectorIndex` and `KVCacheCompressor` for compact binary persistence
- **`actualBytes`** field in `memoryUsage` reporting real in-memory footprint
- **THEORY.md** — detailed mapping of implementation to the TurboQuant paper with deviation notes
- **Benchmarks** — `benchmarks/bench.ts` covering compression quality, recall@k, and performance timing
- **Use Cases** section and comparison table in README
- Improved test rigor: z-tests for unbiasedness, edge cases (d=4, metric:'ip'), bit-pack roundtrips (196 tests)

### Changed

- `Rotation` interface no longer exposes `matrix` property (internal detail)
- `TurboQuantProd.dequantize()` documented as MSE-only reconstruction; removed dead QJL code path

### Fixed

- README "How It Works" now accurately describes the RHT rotation method

## [0.1.0] - 2026-04-10

### Added

- `TurboQuantMSE` — MSE-optimal scalar quantizer with random rotation and Lloyd-Max codebooks
- `TurboQuantProd` — two-stage quantizer (MSE + QJL) for unbiased inner product estimation
- `VectorIndex` — compressed vector index for approximate nearest-neighbor search
- `KVCacheCompressor` — KV cache compressor for LLM inference
- `QJL` — Johnson-Lindenstrauss sign-bit projection
- Random rotation via structured orthogonal transforms
- Codebook generation with Beta-distribution-based Lloyd-Max optimization
- Dual build output (ESM + CJS) with full type declarations
- CI pipeline (Node 18, 20, 22)
- Comprehensive test suite

[0.1.0]: https://github.com/danilodevhub/turboquant-js/releases/tag/v0.1.0
