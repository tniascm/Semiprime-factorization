# E20: Boolean Polynomial Degree Audit for Eisenstein Channels

## Motivation

The Amy-Stinchcombe paper (arXiv:2408.02778) shows that hidden shift quantum circuits
are classically poly-time simulable precisely when their path sum polynomial has
**bounded degree** over F2.  The project's Barrier Theorem shows that Ring+Jacobi
circuits of constant depth d produce functions with CRT rank ≤ 3^d — which implies
spectral flatness and no factoring signal.

**Structural parallel:**
- Amy-Stinchcombe: bounded path sum degree → poly-time classical simulation (positive)
- Barrier Theorem:  bounded CRT rank       → spectral flatness (negative)

**Open question (BARRIER_THEOREM.md §3, §4.1):**
The barrier theorem is only *proven* for constant-depth circuits.  For poly-depth
circuits (where 3^{poly(n)} > √N), spectral flatness is empirical, not proven.

**This experiment:** Test whether the Boolean polynomial degree of the Eisenstein
channel target function f_{k,ℓ}(N) = (σ_{k-1}(N) mod ℓ) mod 2 grows as:

| Degree scaling | Implication |
|---|---|
| O(1) — bounded constant | Amy-Stinchcombe regime: poly-time factoring via path sum rewriting |
| O(n^c), c < 1/3 | Sub-GNFS: beats L[1/3] factoring complexity |
| O(n^{1/3}) | GNFS regime: consistent with known complexity |
| O(n^{1/2}) | Random function baseline |
| Ω(n) | Linear: barrier extends to all poly-depth circuits (new proof) |

## Reduction chain (why this matters for factoring)

From BARRIER_THEOREM.md §13 (E13b reduction theorem):

> poly(log N) access to σ_{k-1}(N) mod ℓ ⟹ poly(log N) factoring

Therefore: if path sum rewriting of degree-d polynomials is poly(n^d)-time
(Amy-Stinchcombe), and if deg(f_{k,ℓ}) = d is bounded, then we get poly-time
factoring.  Any sublinear degree growth gives better-than-GNFS.

## Measurements

### 1. Correlation lower bound (all n_bits, d = 1..max_degree)

For each bit size n and degree d:
- Sample 2000 balanced semiprimes N = pq
- Compute target bit: f_{k,ℓ}(N) = (σ_{k-1}(N) mod ℓ) mod 2
- Sample 200 random degree-d monomials M_S(N) = ∏_{i ∈ S} bit_i(N)
- Measure maximum |correlation(M_S, f_{k,ℓ})| in ±1 encoding
- If max|corr| > 3/√2000 ≈ 0.067: degree ≥ d confirmed (lower bound)

### 2. Minimum fitting degree d ≤ 2 (n_bits ≤ 32)

For d = 0, 1, 2:
- Build feature matrix Φ ∈ F2^{m × C(n,≤d)}
- Check via Gaussian elimination whether a degree-d polynomial over F2 fits all samples
- Report minimum fitting degree (upper bound on true degree for the sample)

### 3. CRT rank (n_bits ≤ 20)

- Enumerate all balanced semiprimes for exact n-bit range
- Build matrix M[p][q] = f_{k,ℓ}(pq) over F2
- Compute rank via Gaussian elimination
- Compare with log2(degree lower bound) to test degree-rank equivalence

## Running

```bash
# Quick smoke test (2 bit sizes, small sample, ~5 seconds)
cargo run -p path-sum-degree --release -- --mode=quick

# Full audit for small n (CRT rank + correlation, ~2 minutes)
cargo run -p path-sum-degree --release -- --mode=audit

# Full scaling scan (11 bit sizes, ~15 minutes)
cargo run -p path-sum-degree --release -- --mode=scale

# Custom parameters
cargo run -p path-sum-degree --release -- \
    --mode=scale \
    --bits=14,16,20,24,32,40,48 \
    --samples=5000 \
    --monomials=500 \
    --max-degree=6 \
    --seed=12345

# Tests
cargo test -p path-sum-degree
```

## Output

Results are written to `data/E20_audit_results.json` or `data/E20_scaling_results.json`.

The printed summary shows:
1. Power-law fits: deg_lower_bound(n) ≈ a·n^b per channel
2. Correlation table: max|corr| vs (n_bits, degree) per channel
3. CRT rank table (for small n)

## Connection to Ryan Williams

Williams' algorithmic method: a faster-than-brute-force algorithm for circuit
class C → lower bound showing C can't compute all functions.

Applied here: if path sum rewriting (bounded-degree confluent rewriting) is
poly-time for degree-d polynomials over n input bits, and f_{k,ℓ} has degree d,
then f_{k,ℓ} is evaluable in poly-time → poly-time factoring (via E13b).

Conversely: if degree d grows as Ω(n/2), then no bounded-degree path sum algorithm
can evaluate f_{k,ℓ} in sub-exponential time — a new lower bound, extending the
barrier theorem from constant-depth (proven) to poly-depth (empirical → proven).

## Status

- [ ] Quick smoke test passes
- [ ] Full audit results collected (n = 14..20)
- [ ] Scaling law b measured (n = 14..48)
- [ ] Degree-rank equivalence hypothesis tested
- [ ] Results interpreted vs GNFS/poly-time thresholds
