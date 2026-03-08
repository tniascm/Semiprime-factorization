# CADO-NFS Parity Design: Parameter Alignment + Advanced Polyselect + Block Wiedemann

**Date:** 2026-03-08
**Goal:** Dominate CADO-NFS on c30 benchmark reproducibly on all metrics (single-threaded CPU, multi-threaded wall).
**Current state:** Rust 5548ms vs CADO 2149ms single-threaded (2.6x gap). Rust 1923ms vs CADO 7534ms multi-threaded wall (Rust 3.9x faster).
**Target:** Single-threaded CPU < 2000ms. Multi-threaded wall < 1500ms.

---

## Root Cause Analysis

Three compounding mismatches vs CADO params.c30:

| Issue | Rust-NFS | CADO c30 | Impact |
|-------|----------|----------|--------|
| Sieve area (log_i) | 7 (area=32K) | 9 (area=524K) | 16x less area per q, need 16x more q's |
| Sieve threshold | mfb=36 (2LP bump) | mfb=18 (no bump) | 9x more false survivors (230k vs ~30k) |
| Polynomial search | 5 base-m variants | admax=5000, ropteffort=1 | Lower yield per q |

Profiling breakdown (single-threaded, 4416ms sieve):
- Setup/scatter: 2515ms (57%) — runs 2704 times (once per q)
- Cofactoring: 1394ms (32%) — processes 230k false-positive survivors
- Region scan: 487ms (11%) — norm(45ms) + small_sieve(150ms) + bucket_apply(240ms)

## Part 1: Parameter Alignment

### 1a. Fix log_i for c30

Change `params.rs` c30 preset: `log_i: 7` -> `log_i: 9`.

This matches CADO's `tasks.I = 9` which gives `half_width = 2^9 = 512`, `sieve_width = 1024`, `max_j = 512`, area = 524,288 per special-q.

**Effect:** Each q covers 16x more (a,b) pairs. Relations per q increases ~16x, requiring ~16x fewer q's. Per-q setup time increases ~4x (more scatter updates), but total sieve time drops ~4x.

### 1b. Decouple sieve threshold from 2LP mfb

The 2LP merge bumps mfb from 18 to 36. This inflates the sieve bound from 26 to 52, creating 230k survivors where ~30k would suffice. CADO uses mfb=18 for both sieve AND cofactoring (no 2LP for c30).

**Implementation:** Add `sieve_mfb0` / `sieve_mfb1` fields to NfsParams, initialized to the original mfb values BEFORE the 2LP bump. Use these for sieve threshold computation. Keep bumped mfb for cofactoring decisions.

In `sieve/mod.rs` threshold computation:
```rust
let rat_bound = ((params.sieve_mfb0 as f64) * scale).min(255.0) as u8;
let alg_bound = ((params.sieve_mfb1 as f64) * scale).min(255.0) as u8;
```

**Effect:** Survivors drop from 230k to ~30k. Cofactoring time drops from 1394ms to ~200ms.

### 1c. Parameter preset update

Update all parameter presets (c30, c35, c40, c45) to match CADO's parameter files where available. Cross-reference with `/Users/andriipotapov/cado-nfs/parameters/factor/params.c*`.

## Part 2: Advanced Polynomial Selection

### 2a. Non-monic polynomials

Current code forces leading coefficient = 1 (monic). CADO searches over leading coefficients `ad` from 1 to `admax=5000` with `incr=20`.

For degree 3: `f(x) = ad*x^3 + c2*x^2 + c1*x + c0` where:
- Choose `ad` and compute `m = floor((N/ad)^(1/d))`
- Expand: `ad*m^d + c_{d-1}*m^{d-1} + ... + c_0 = N`
- All coefficients scale as `~N^(1/d) / ad^((d-1)/d)` which is smaller when `ad > 1`

**Implementation:** In `gnfs/src/polyselect.rs`:
1. Add `select_polynomial_with_ad(n, degree, ad)` that generates a polynomial with given leading coefficient
2. Search over `ad` in `[1, admax]` with step `incr`
3. For each `ad`, generate the polynomial and score it

### 2b. Murphy alpha scoring

Murphy alpha measures the "root property" of a polynomial — how many roots it has modulo small primes. More roots mod p means primes divide the norm more often, producing smoother values.

```
alpha(f) = -sum_{p prime <= B} (1/p) * (r_p / p - 1) * log(p)
```

where `r_p` = number of roots of f mod p (including projective root if p | leading coeff).

More negative alpha = better polynomial (more smooth values).

**Implementation:** In `gnfs/src/polyselect.rs`:
1. Add `murphy_alpha(f_coeffs, bound)` that computes alpha for primes up to `bound` (typically 2000)
2. Use `avg_log_norm + alpha` as the combined quality score
3. Keep top `nrkeep` polynomials by combined score

### 2c. Root optimization (rotation)

After finding a good `(f, g)` pair, apply linear rotations `f' = f + k*g` (for integer k) that preserve the number field but may improve root properties and reduce norms.

**Implementation:** In `gnfs/src/polyselect.rs`:
1. Add `optimize_roots(f, g, effort)` that tries rotations `f + k*g` for `k` in `[-K, K]`
2. Score each rotation by `avg_log_norm + murphy_alpha`
3. Keep the best rotation

### 2d. Combined polynomial search

```
for ad in (1..admax).step_by(incr):
    f, g = generate_polynomial(N, degree, ad)
    f, g = optimize_roots(f, g, ropteffort)
    score = avg_log_norm(f) + murphy_alpha(f) + murphy_alpha(g)
    keep top nrkeep by score
```

**Expected yield improvement:** 2-5x more relations per q from better polynomial quality.

## Part 3: Block Wiedemann for Linear Algebra

### 3a. Why Block Wiedemann

Current: Dense Gaussian Elimination, O(n^3) where n = matrix dimension.
For c30 matrix (8466 x 8359): GE takes ~542ms. Acceptable.
For c45+ matrices (50K+): GE would take minutes-hours. BW is O(n^2).

### 3b. Algorithm outline

Block Wiedemann over GF(2) for sparse matrix A (nrows x ncols):
1. Choose blocking parameter m (typically 64 for u64 word size)
2. Generate random m-column matrix X
3. Compute sequence: A_i = X^T * A^i * Y for i = 0..2n/m
4. Find minimal polynomial of the sequence using Berlekamp-Massey
5. Extract nullspace vectors

The key operation is sparse matrix-vector multiply (SpMV), which is O(nnz) per iteration. With ~2n/m iterations, total is O(n * nnz / m).

### 3c. Sparse matrix representation

Convert from dense BitRow to CSR (Compressed Sparse Row) format:
- `row_ptr[i]` = start of row i's column indices
- `col_idx[j]` = column index of j-th nonzero

SpMV with 64-wide blocking: process 64 RHS vectors simultaneously using u64 word ops.

### 3d. Implementation scope

For c30 (n~8K), BW is ~2x faster than GE. For c45+ (n~50K+), BW is essential. Implementation in `gnfs/src/linalg.rs`:
1. Add `SparseMatrix` type (CSR format)
2. Add `block_wiedemann(matrix, ncols, block_size)` function
3. Wire into pipeline with size-based gate (use BW for n > 5000)

---

## Implementation Priority

1. **Part 1a + 1b** (Parameter + threshold fix): Highest ROI, simplest changes. Expected to cut sieve from 4400ms to ~400ms.
2. **Part 2a-2d** (Polynomial selection): Medium ROI, moderate complexity. Expected additional 2-5x yield improvement.
3. **Part 3** (Block Wiedemann): Lower ROI for c30 (saves ~300ms), essential for c45+.

## Success Criteria

- c30 single-threaded: < 2000ms (currently 5548ms)
- c30 multi-threaded wall: < 1500ms (currently 1923ms)
- Correct factorization on all test cases
- All existing tests pass
