# TurboQuant: Theory and Implementation Details

## Paper Citation

> Zandieh, A., Daliri, A., Hadian, A., & Mirrokni, V. (2026). **TurboQuant: Online Vector Quantization**. In *Proceedings of the International Conference on Learning Representations (ICLR 2026)*.
>
> arXiv: [https://arxiv.org/abs/2504.19874](https://arxiv.org/abs/2504.19874)

## Algorithm Overview

TurboQuant is a data-oblivious vector quantization algorithm that compresses d-dimensional vectors using b bits per coordinate. It achieves near-optimal mean squared error (MSE) distortion -- within approximately 2.7x of the information-theoretic lower bound -- without requiring any training data or codebook learning.

The algorithm operates in two stages:

- **Stage 1 (MSE quantization):** A random rotation followed by per-coordinate scalar quantization using an optimal Lloyd-Max codebook. This uses (b-1) bits per coordinate in the Prod quantizer, or all b bits in the MSE-only quantizer.
- **Stage 2 (QJL correction):** A 1-bit Quantized Johnson-Lindenstrauss sketch of the quantization residual that provides an unbiased correction term for inner product estimation.

The two-stage design guarantees that inner product estimates are unbiased: E[IP_estimate(y, Q(x))] = <y, x>.

## Stage 1: Random Rotation (Randomized Hadamard Transform)

### Purpose

Random rotation is the key insight enabling per-coordinate scalar quantization to be near-optimal. A random orthogonal rotation spreads the energy of any vector uniformly across all coordinates, so that after rotation each coordinate follows a known distribution regardless of the input vector's structure.

### Implementation

The rotation uses a **Randomized Hadamard Transform (RHT)** consisting of 3 rounds of the form H * D, where:

- **D** is a diagonal matrix with random +1/-1 entries (a "sign flip" matrix)
- **H** is the normalized Walsh-Hadamard transform (WHT), applied via the Fast Walsh-Hadamard Transform (FWHT)

The full rotation is: R = (H * D_3) * (H * D_2) * (H * D_1)

**Zero-padding:** The Walsh-Hadamard transform requires input length to be a power of 2. For non-power-of-2 dimensions d, the input is zero-padded to m = nextPow2(d) before rotation, and the output is truncated back to d elements. The sign vectors are also generated at length m.

**Normalization:** After each FWHT application, the result is scaled by 1/sqrt(m) to maintain the orthogonal property. Over 3 rounds this means each round applies (1/sqrt(m)) * H * D.

**Complexity:**
- Time: O(d log d) per round, O(d log d) total (constant 3 rounds)
- Space: O(d) for the working buffer and O(d) per sign vector (3 sign vectors total)

**Inverse rotation:** The inverse uses the same sign vectors in reverse order. Since both H (up to scaling) and D are self-inverse, the inverse of round r is D_r * H * (1/sqrt(m)).

### Why 3 Rounds?

Three rounds of randomized Hadamard is a well-studied construction that provides sufficient randomness to make the rotation behave like a uniformly random orthogonal matrix for the purposes of coordinate-wise analysis. A single round would not provide enough mixing.

### Coordinate Distribution After Rotation

For a unit vector x in R^d, after applying a random orthogonal rotation, each coordinate of the rotated vector follows the distribution:

f(t) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - t^2)^((d-3)/2)

for t in (-1, 1). This is a rescaled Beta distribution: if Y ~ Beta(1/2, (d-1)/2), then t = 2Y - 1.

## Stage 2: Lloyd-Max Quantization

### Overview

Each coordinate of the rotated vector is independently scalar-quantized using an optimal codebook. The codebook is computed once for a given (dimension, bits) pair and cached.

### Lloyd-Max Algorithm

The Lloyd-Max algorithm finds the optimal scalar quantizer (centroids and decision boundaries) that minimizes the MSE distortion for a given probability distribution. It alternates:

1. **Update boundaries:** Set each boundary to the midpoint of adjacent centroids: b_i = (c_i + c_{i+1}) / 2
2. **Update centroids:** Set each centroid to the conditional expectation within its partition: c_i = E[X | X in partition_i] = integral(x * f(x) dx) / integral(f(x) dx)

The integrals are computed using adaptive Simpson quadrature with tolerance 1e-10.

### PDF Selection

- **d < 64:** Uses the exact Beta PDF described above, with range (-0.999, 0.999)
- **d >= 64:** Uses a Gaussian approximation N(0, 1/d), with range +/- 4*sigma = +/- 4/sqrt(d)

The Gaussian approximation is justified by the central limit theorem: as d grows, the Beta distribution converges to a Gaussian with variance 1/d. The threshold d=64 provides a good trade-off between accuracy and computation.

### Initialization

Centroids are initialized uniformly across the effective range: c_i = -L + 2L*(i + 0.5)/K for i = 0, ..., K-1, where K = 2^bits and L is the range limit.

### Convergence

The algorithm runs for up to 200 iterations with a convergence tolerance of 1e-10 on the maximum centroid displacement. In practice, convergence is achieved in far fewer iterations.

### Quantization

Given a rotated coordinate value, the nearest centroid is found by a linear scan over the decision boundaries. For the small centroid counts used in practice (2-16 for 1-4 bits), linear scan is faster than binary search due to branch prediction.

## Stage 3: QJL Correction

### Purpose

Scalar quantization introduces a residual r = x - x_hat that is correlated with the quantization process. Simply using the MSE reconstruction x_hat for inner products would yield biased estimates. The QJL stage corrects this bias.

### Random Projection Matrix

A dense d x d matrix S is generated with i.i.d. N(0,1) entries. This is stored as a flat Float64Array in row-major order.

**Important cost note:** This matrix requires O(d^2) storage and O(d^2) time for matrix-vector products. For large d (e.g., d=1024), this means ~8MB of storage for the matrix alone. This is the dominant cost in the system for large dimensions.

### Sign-Bit Quantization

The residual r = x - x_hat is projected through S, then sign-quantized:

q_j = sign((S * r)_j) for j = 1, ..., d

This produces d sign bits (stored as a packed Uint8Array), consuming 1 bit per coordinate.

### Inner Product Correction

For a query vector y, the QJL correction term is:

correction = sqrt(pi/2) / d * ||r|| * sum_j(q_j * (S * y)_j)

The total inner product estimate is:

<y, x>_est = <y, x_hat> + correction

### Proof of Unbiasedness

The key identity is:

E[sign(g) * g'] = sqrt(2/pi) * rho

where g, g' are jointly Gaussian with correlation rho = E[g * g']. In our setting:

- g = (S * r / ||r||), which has i.i.d. N(0,1) entries
- For the j-th coordinate: E[q_j * (S * y)_j] = sqrt(2/pi) * <r, y> / ||r|| * ... (by the Gaussian correlation structure)

Taking the expectation over the random matrix S:

E[correction] = sqrt(pi/2) / d * ||r|| * d * sqrt(2/pi) * <r/||r||, y> = <r, y>

Therefore:

E[<y, x>_est] = <y, x_hat> + <r, y> = <y, x_hat + r> = <y, x>

The estimator is unbiased.

### Implementation Detail

Instead of computing S^T * signs and then dotting with y, the implementation computes S * y first and dots with signs. Both compute the same quantity: signs^T * S * y. This avoids materializing the transposed product.

## Bit Budget Allocation

For the TurboQuantProd quantizer with total budget b bits per coordinate:

- **(b-1) bits** are allocated to the MSE stage (Lloyd-Max scalar quantization)
- **1 bit** is allocated to the QJL correction stage

For example, with b=3 total bits:
- 2 bits for MSE quantization (4 centroids per coordinate)
- 1 bit for QJL sign (residual correction)

The TurboQuantMSE quantizer (no inner product guarantees) uses all b bits for the MSE stage.

## Non-Unit-Norm Handling

The theoretical analysis assumes unit-norm input vectors. For non-unit-norm vectors, the implementation:

1. Records the original L2 norm of the input
2. Normalizes the input to unit norm before rotation and quantization
3. Stores the norm as side information (one float64 scalar)
4. On dequantization, scales the reconstructed unit vector by the stored norm

This adds a small constant overhead (8 bytes per vector) but allows the algorithm to handle arbitrary vectors while maintaining the optimality guarantees of the unit-sphere analysis.

## Known Limitations

### QJL Projection is O(d^2)

The QJL stage uses a dense d x d Gaussian random matrix. This means:
- **Storage:** O(d^2) -- for d=384, the matrix is ~1.1 MB; for d=1024, ~8 MB
- **Computation:** Each quantize() and innerProduct() call requires a matrix-vector product in O(d^2)
- The rotation stage (O(d log d)) and scalar quantization (O(d)) are dominated by this cost

Possible mitigations (not yet implemented):
- Sparse JL projections (reduces to O(d) but with worse constants)
- Structured random projections (e.g., subsampled randomized Hadamard for JL)

### RHT Padding for Non-Power-of-2 Dimensions

When d is not a power of 2, the input is zero-padded to m = nextPow2(d). This means:
- The FWHT operates on m-length vectors (up to 2x the input dimension)
- The sign vectors are m-length
- The padded coordinates contribute to the rotation but are discarded in the output

For d=384 (next power of 2 is 512), this is a 33% overhead in the rotation step. For d=256 or d=128 (already powers of 2), there is no overhead.

### Brute-Force Search

The VectorIndex performs a linear scan over all stored vectors for each query. There is no approximate nearest neighbor (ANN) index structure (e.g., HNSW, IVF). Search time is O(n * d) where n is the number of stored vectors.

### No SIMD Acceleration

The implementation is pure TypeScript/JavaScript with no WebAssembly or SIMD intrinsics. For production workloads with large d, a native implementation would be significantly faster.

## Theoretical Guarantees

### MSE Bound

For a b-bit per-coordinate quantizer on unit vectors in R^d, TurboQuant achieves:

E[||x - Q(x)||^2] <= C * d * D_b

where D_b is the optimal scalar quantization distortion for the coordinate distribution and C is a constant close to 1. The distortion D_b decreases exponentially with b (roughly halving for each additional bit).

### Unbiased Inner Products

For the TurboQuantProd quantizer:

E[<y, Q_prod(x)>] = <y, x>

for any vectors x, y. The expectation is over the random rotation and QJL matrices (which are fixed at construction time -- the guarantee holds in expectation over a random seed).

### Compression Ratio

The theoretical compression ratio for the Prod quantizer is:

ratio = 64 / (b + 32/d)

where the 32/d term accounts for the residual norm float stored per vector. For large d this approaches 64/b (e.g., ~21x for b=3).

## References

1. Zandieh, A., Daliri, A., Hadian, A., & Mirrokni, V. (2026). TurboQuant: Online Vector Quantization. ICLR 2026. [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)

2. Lloyd, S. P. (1982). Least squares quantization in PCM. IEEE Transactions on Information Theory, 28(2), 129-137.

3. Max, J. (1960). Quantizing for minimum distortion. IRE Transactions on Information Theory, 6(1), 7-12.

4. Ailon, N., & Chazelle, B. (2009). The fast Johnson-Lindenstrauss transform and approximate nearest neighbors. SIAM Journal on Computing, 39(1), 302-322.

5. Johnson, W. B., & Lindenstrauss, J. (1984). Extensions of Lipschitz mappings into a Hilbert space. Contemporary Mathematics, 26, 189-206.

6. Tropp, J. A. (2011). Improved analysis of the subsampled randomized Hadamard transform. Advances in Adaptive Data Analysis, 3(1-2), 115-126.
