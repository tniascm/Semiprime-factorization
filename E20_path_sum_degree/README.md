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

- [x] Quick smoke test passes
- [x] Full audit results collected (n = 14..20)  → `data/E20_audit_results.json`
- [x] Scaling law b measured (n = 14..48)        → `data/E20_scaling_results.json`
- [x] Degree-rank equivalence hypothesis tested
- [x] Results interpreted vs GNFS/poly-time thresholds

## Results (2026-02-23)

### CRT Rank (reliable signal)

| n_bits | n_primes | mean rank | mean frac |
|--------|----------|-----------|-----------|
| 14 | 32 | 23.4 | 0.732 |
| 16 | 59 | 43.1 | 0.731 |
| 18 | 104 | 78.9 | 0.758 |
| 20 | 187 | 144.0 | 0.770 |

Rank fraction is stable at **~0.75** across all n.  The absolute rank scales
as rank ≈ 0.75 × π(2^{n/2}).  By the prime number theorem, π(2^{n/2}) ≈
2^{n/2}/(n·ln2), so log₂(rank) ≈ n/2 − log₂(n).

Under the degree-rank equivalence hypothesis this implies **deg_F2 ≈ n/2**
(linear in n).  Amy-Stinchcombe rewriting for a degree-d polynomial takes
poly(circuit^d) time; with d ≈ n/2 that is super-exponential — no improvement
over brute force.

### Correlation lower bound (inconclusive)

The random monomial test (200 monomials, threshold 3/√m ≈ 0.067 at m=2000)
is too noisy to resolve the degree scaling law.  With 200 samples the expected
maximum correlation under the null hypothesis is ~0.080 > threshold, giving a
false-positive rate of ~42% per (channel, n, degree) cell.  The resulting d*
values are dominated by this noise and the power-law fits are unreliable.

Observed significant cells at n ≥ 24 (max over 42 cells per bit size):
n=24: 14/42 · n=28: 2/42 · n=32: 5/42 · n=36: 1/42 · n=40: 2/42 · n=44: 10/42 · n=48: 1/42

The alternating pattern (high → low → moderate → low → ...) is inconsistent
with any monotone degree scaling law and is consistent with sampling artefact.

### Implication for factoring

CRT rank ≈ 0.75·π(2^{n/2}) → log₂(rank) ≈ n/2 → deg_F2 ≈ n/2 (if degree-rank
equivalence holds).  This is the **LINEAR / full hardness** regime.  Path sum
rewriting provides no sub-exponential speedup.  Combined with the E13b reduction
(poly(log N) access to σ_{k-1}(N) mod ℓ ⟹ poly(log N) factoring), the result
is consistent with the barrier theorem extending to poly-depth circuits via the
degree-rank equivalence.

**Verdict:** No evidence of bounded or sub-GNFS degree.  The Amy-Stinchcombe
path sum framework does not yield poly-time or better-than-GNFS factoring for
the Eisenstein channels with the current measurement resolution.

### What would sharpen the measurement

1. ~~Calibrated null threshold~~ ✓ done (E20b)
2. ~~Extend CRT rank to n=22,24~~ ✓ done (E20b)
3. ~~Real SVD spectrum~~ ✓ done (E20b) — see dramatic finding below

## Results (2026-02-23, E20b — improved analysis)

### Improvement 1: Calibrated null threshold eliminates all spurious hits

With the 99th-percentile permutation threshold (~0.080 at m=2000 vs old 0.067):

| n_bits | sig/42 cells | calibrated threshold |
|--------|-------------|---------------------|
| 14 | 1/42 | 0.272 |
| 16 | 5/42 | 0.157 |
| 18 | 0/42 | 0.086 |
| 20–48 | 0–6/42 | ~0.080 |

**Verdict:** All previous `*` were false positives (42% FPR with old threshold).
The correlation test finds **no significant degree structure** at any n ≤ 48 or
degree ≤ 6.  Spuradic 1–6/42 hits are consistent with the expected 1% FPR.

### Improvement 2: CRT rank extended to n=22,24

| n_bits | n_primes | mean rank fraction |
|--------|----------|--------------------|
| 14 | 32 | 0.732 |
| 16 | 59 | 0.731 |
| 18 | 104 | 0.758 |
| 20 | 187 | 0.770 |
| 24 | 636 | **0.787** |

Rank fraction increases from 0.73 to 0.79 across n=14..24. Not converging to 1
(no simple compressible structure) and not converging to 0 (no low-rank shortcut).

### Improvement 3: Real-valued SVD — KEY NEW FINDING

Stable rank = ‖M‖_F² / σ_max² across all n and all 7 channels:

| n_bits | n_primes | mean stable_rank | frac_stable |
|--------|----------|-----------------|-------------|
| 14 | 32 | 3.71 | 0.116 |
| 16 | 59 | 3.61 | 0.061 |
| 18 | 104 | 3.68 | 0.035 |
| 20 | 187 | 3.65 | 0.020 |
| 24 | 636 | 3.74 | 0.006 |

**Stable rank ≈ 3.7 is CONSTANT across all n and all 7 channels.**

Top-5 |eigenvalues| at n=20 (k=12, ℓ=691): `[39.32, 28.22, 16.04, 11.24, 9.02]`
- σ₁² captures 27.5% of ‖M‖_F²; top-5 capture 50%

This is a property of the **function** g(a,b) = ((1+a)(1+b) mod ℓ) mod 2, not
of the sample size.  Its harmonic structure has ~4 dominant real directions
regardless of how many primes are tested.

### Reconciliation: F2 rank vs real stable rank

| Measure | Value | Growth | Implication |
|---------|-------|--------|-------------|
| F2 rank fraction | ~0.75–0.79 | linear in n_primes | Barrier holds over GF(2) |
| Real stable rank | ~3.7 | constant | ≈4 dominant real directions |

These are complementary.  The parity function mod 2 destroys smooth real structure
(creating many GF(2)-independent rows) while the dominant harmonic components of g
remain fixed.

### Open question — can stable rank ≈ 4 be exploited?

The dominant eigenvectors of M have structure from the harmonic analysis of
((1+p^{k-1})(1+q^{k-1}) mod ℓ) mod 2.  Power iteration on M would extract this
structure — but building M requires knowing all (p,q) factorizations.

Key obstacle: we cannot access M without knowing the factors.  Whether the
dominant direction is computable from N alone (without factors) is the open
question.  This connects to the "analytic continuation" corridor in
ACCESS_MODEL_REQUIREMENTS.md and is a distinct direction from the GF(2)/path-sum
barrier — the real-valued structure is qualitatively simpler than GF(2) suggests.
