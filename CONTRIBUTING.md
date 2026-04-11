# Contributing to turboquant-js

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing.

## Getting Started

1. **Fork** the repository
2. **Clone** your fork:
   ```bash
   git clone https://github.com/<your-username>/turboquant-js.git
   cd turboquant-js
   ```
3. **Install** dependencies:
   ```bash
   npm install
   ```
4. **Run tests** to make sure everything works:
   ```bash
   npm test
   ```

## Development Workflow

### Branch naming

- `feat/short-description` for new features
- `fix/short-description` for bug fixes
- `docs/short-description` for documentation changes

### Making changes

1. Create a new branch from `main`
2. Make your changes
3. Add or update tests as needed
4. Ensure all checks pass:
   ```bash
   npm run typecheck   # Type checking
   npm run test        # Unit tests
   npm run build       # Build output
   ```

### Code style

- TypeScript strict mode is enabled
- No runtime dependencies — keep the bundle lean
- Use `Float64Array` for vector data (not `number[]` internally)
- No Node.js-specific APIs in `src/` — the library must work in browsers

### Testing

- Tests live in `tests/` mirroring the `src/` structure
- Use `vitest` — run with `npm test` or `npm run test:watch`
- Aim for meaningful tests: verify mathematical properties, round-trip identities, and statistical guarantees (e.g., unbiasedness)
- Run coverage with `npm run test:coverage`

## Submitting a Pull Request

1. Push your branch to your fork
2. Open a PR against `main` on the upstream repo
3. Fill out the PR template
4. Ensure CI passes (tests on Node 18, 20, 22)

### What makes a good PR

- Focused on a single change
- Includes tests for new functionality
- Doesn't break existing tests
- Has a clear description of *why* the change is needed

## Reporting Bugs

Use the [bug report template](https://github.com/danilodevhub/turboquant-js/issues/new?template=bug_report.md). Include:

- Steps to reproduce
- Expected vs actual behavior
- Environment details (Node version, runtime, OS)

## Feature Requests

Use the [feature request template](https://github.com/danilodevhub/turboquant-js/issues/new?template=feature_request.md).

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
