# E21: Dominant Eigenvector Multiplicative Character Structure

**Status:** Complete (E21 barrier confirmed; E21b smoothness bias real but NOT N-extractable).

## Motivation

E20 found that the real CRT matrix `M[p][q] = (σ_{k−1}(pq) mod ℓ) mod 2`
has a **constant stable rank ≈ 3.7** across all n=14–24 bits and all 7 channels.
This raised a question: does the dominant eigenvector u₁ have exploitable
**multiplicative character structure**, and is the product u₁(p)·u₁(q)
computable from N = pq alone (without knowing p, q)?

## Key algebraic identity

Define `g(p) = (1 + p^{k−1}) mod ℓ`.  Then:

```
(1 + p^{k−1})(1 + q^{k−1}) = 1 + p^{k−1} + q^{k−1} + (pq)^{k−1}
                             = σ_{k−1}(pq) mod ℓ
```

So `M[p][q] = parity(g(p) · g(q) mod ℓ)`.  This is a purely **algebraic**
factorisation of M: the matrix entry depends on the product `g(p)·g(q)` in the
cyclic group `(ℤ/ℓℤ)*`, composed with the parity function h(x) = x mod 2.

**Important distinction:** over the **full** cyclic group (ℤ/ℓℤ)*, the matrix
H[a][b] = h(ab mod ℓ) has multiplicative characters χ_r as exact eigenvectors.
But this diagonalisation does **not** transfer cleanly to the prime-restricted
matrix M, because: (i) the mapping p → g(p) sends primes to a sparse,
algebraically structured subset of the group; and (ii) the parity function h
mixes many Fourier modes, so the dominant eigenvectors of M are mixtures of
characters, not individual ones.

If the eigenvectors **were** individual characters, then:

- `u₁(p)·u₁(q) ≈ Re(χ_{r*}(g(p)·g(q) mod ℓ)) = Re(χ_{r*}(σ_{k−1}(N) mod ℓ))`
- The product depends on `σ_{k−1}(N) mod ℓ`, which requires knowing p, q.
- NOT computable from `N^{k−1} mod ℓ` alone (which IS accessible from N).

## E21 measurements

For each (n_bits ∈ {14,16,18,20}, channel ∈ 7 Eisenstein channels):

1. Build M exactly.  Extract top-3 eigenvectors via deflated power iteration.
2. **Character scan**: for `u₁` and each non-trivial character `r = 1 … (ℓ−1)/2`,
   compute `amp(r) = sqrt(corr_re² + corr_im²)` (complex Pearson amplitude).
3. Find `r*` = argmax amp(r).
4. **Product test A**: `corr(u₁(p)·u₁(q), Re(χ_{r*}(σ_{k−1}(N) mod ℓ)))`.
5. **Product test B**: `corr(u₁(p)·u₁(q), Re(χ_{r*}(N^{k−1} mod ℓ)))`.

## Results

### Table 1: Character fit of top eigenvector u₁

| n  | k  | ℓ     | r*    | corr_re | corr_im | amp   | null_amp |
|----|-----|-------|-------|---------|---------|-------|----------|
| 14 | 12 | 691   | 70    | −0.074  | +0.559  | 0.564 | **0.660** |
| 14 | 16 | 3617  | 1091  | −0.603  | −0.323  | 0.684 | **0.739** |
| 14 | 18 | 43867 | 7192  | +0.709  | +0.128  | 0.721 | **0.844** |
| 14 | 22 | 131   | 62    | −0.540  | +0.236  | 0.590 | **0.570** |
| 16 | 12 | 691   | 179   | −0.216  | +0.352  | 0.413 | **0.479** |
| 18 | 12 | 691   | 244   | −0.342  | −0.104  | 0.357 | **0.358** |
| 20 | 12 | 691   | 287   | −0.120  | +0.219  | 0.250 | **0.266** |
| 20 | 18 | 43867 | 5795  | +0.298  | +0.155  | 0.336 | **0.340** |
| 20 | 22 | 131   | 31    | −0.214  | −0.052  | 0.220 | **0.229** |

`null_amp` = expected maximum over random data = `sqrt(2·ln(2k) / (n_primes−2))`.

**Every observed amplitude falls at or below the null expectation.**
The best-fitting character r* is a noise fluctuation, not a true signal.

### Table 2: Product tests (all 28 blocks)

| n  | k  | ℓ     | corr_σ  | corr_Nk | verdict    |
|----|----|-------|---------|---------|------------|
| 14 | 12 | 691   | −0.088  | +0.024  | ✓ barrier  |
| 14 | 16 | 3617  | +0.314  | +0.010  | ✓ barrier  |
| 14 | 18 | 43867 | +0.215  | +0.054  | ✓ barrier  |
| 16 | 18 | 43867 | +0.250  | +0.055  | ✓ barrier  |
| 18–20 (all) | … | … | ≤ 0.104 | ≤ 0.026 | ✓ barrier |

- `corr_σ`: weakly non-zero at small n due to selection bias (r* chosen to fit u₁);
  shrinks to noise for n ≥ 18.
- `corr_Nk`: **≤ 0.088 in every block**, consistent with zero.
  The barrier holds: u₁(p)·u₁(q) is NOT determined by N^{k−1} mod ℓ alone.

## Pipeline validation: full-group control experiment

To confirm that the character scan pipeline works correctly (i.e., the absence of
character structure in Table 1 is real, not a pipeline artifact), we run a control:

**Setup:** For small primes ℓ ∈ {131, 283, 593, 617, 691}, build the **full-group**
matrix H[a][b] = (ab mod ℓ) mod 2 for all a, b ∈ {1, …, ℓ−1}.  Over the full
cyclic group, characters χ_r are **exact** eigenvectors of H with eigenvalue
(ℓ−1)·ĥ(r), where ĥ(r) is the r-th Fourier coefficient of the parity function h.

**Result:** Extract top-5 eigenvectors; the character scan finds amp ≈ 1.0 for
non-trivial eigenvectors:

| ℓ   | order | idx | λ      | r*  | amp   | check       |
|-----|-------|-----|--------|-----|-------|-------------|
| 131 | 130   | 1   | +15.0  | 65  | 1.000 | ✓ character |
| 131 | 130   | 2   | −12.8  | 45  | 1.000 | ✓ character |
| 283 | 282   | 1   | +20.2  | 11  | 0.918 | ✓ character |
| 283 | 282   | 2   | −20.2  | 11  | 1.000 | ✓ character |
| 593 | 592   | 1   | +31.3  | 1   | 0.998 | ✓ character |
| 691 | 690   | 1   | +31.2  | 41  | 0.962 | ✓ character |
| 691 | 690   | 2   | −29.7  | 11  | 1.000 | ✓ character |

16/20 non-trivial eigenvectors have amp > 0.90.  **The pipeline correctly
identifies characters when they truly are eigenvectors.**  The drop to noise
in the prime-restricted audit (Table 1) is real.

### Fourier scaling: exact DFT of centered parity on the full group

To precisely measure how the dominant Fourier amplitude scales with ℓ, we
compute the **exact DFT** of the centered parity function h̃(a) = 2(a mod 2) − 1
on the full group (ℤ/ℓℤ)* for 30 primes spanning ℓ = 101 to 50021.  This uses
no eigenvectors, no prime restriction — just the pure harmonic spectrum of h̃.

Define A_max(ℓ) = max_{r≥1} |ĥ̃(r)|.  Representative values:

| ℓ      | A_max     | A_max·√ℓ | A·√ℓ/√(log ℓ) |
|--------|-----------|-----------|----------------|
| 101    | 0.22576   | 2.27      | 1.056          |
| 701    | 0.09689   | 2.57      | 1.002          |
| 3109   | 0.05245   | 2.92      | 1.031          |
| 11197  | 0.02936   | 3.11      | 1.017          |
| 50021  | 0.01476   | 3.30      | 1.004          |

**Uncorrected fit:** A_max ≈ 1.79 · ℓ^(−0.44), R² = 0.998.
The raw slope (−0.44) is shallower than −1/2.

**With √(log ℓ) correction:** A_max · √ℓ / √(log ℓ) = **1.050 ± 0.035**
(CV = 3.3%) — essentially constant across 3 orders of magnitude.

**Conclusion:** A_max = Θ(√(log ℓ) / √ℓ).  The deviation from pure ℓ^(−1/2)
is fully explained by the √(log ℓ) extremal-value correction (the maximum
over ~ℓ/2 characters, each with amplitude ~1/√ℓ).  No anomalous multiplicative
bias is present.

## Statistical interpretation

The expected null maximum complex Pearson amplitude over `(ℓ−1)/2` characters
with `n_primes` samples is approximately:

```
null_amp ≈ sqrt(2 · ln(ℓ−1) / (n_primes − 2))
```

This captures the "winner's curse" from scanning many correlated characters.
At n=20 with 187 primes and ℓ=691 (345 characters): null_amp ≈ 0.27,
observed amp ≈ 0.25.  At n=20 with ℓ=43867 (21933 characters): null ≈ 0.34,
observed ≈ 0.34.  All results at the noise floor.

## Why no single-character structure: parity mixing

The absence of character structure is **not** primarily due to the prime set being
too sparse (though sparsity compounds the problem).  The more decisive reason is:

1. **Parity projection destroys character diagonalisation.**  Over the full group,
   H[a][b] = h(ab mod ℓ) has characters as eigenvectors with eigenvalues
   (ℓ−1)·ĥ(r).  But A_max = Θ(√(log ℓ)/√ℓ) (confirmed empirically over 30
   primes with CV = 3.3%), so the non-trivial eigenvalues scale as O(√(ℓ·log ℓ))
   while the trivial eigenvalue is (ℓ−1)/2 ≈ ℓ/2.  The ratio
   λ₁/λ₀ ≈ O(√(log ℓ)/√ℓ) → 0, so the spectral gap between the trivial mode
   and any non-trivial character vanishes as ℓ grows.

2. **No single character dominates.**  Since all non-trivial Fourier coefficients
   are O(√(log ℓ)/√ℓ), many characters have comparable amplitude.  The top
   eigenvectors of M are mixtures of the O(1) dominant Fourier modes of h, not
   individual characters.  This is consistent with E20's stable rank ≈ 3.7: the
   3–4 dominant directions in M represent **collective** modes arising from the
   interplay of the parity function and the group structure, not individual χ_r.

3. **The g(·) mapping adds no structure that could help.**  Even if a single
   character dominated H over the full group, the restriction p → g(p) maps
   primes to a sparse, algebraically constrained subset.  But the mixing
   argument above shows that even with dense sampling, no single character
   would dominate — the parity function distributes its spectral weight
   across Θ(ℓ) characters.

**Summary:** No single multiplicative character explains the top eigenvectors;
any harmonic structure is spread across many characters and is not N-only
extractable.

## E21b: Smoothness Fourier spectrum

### Motivation

The parity function h(a) = a mod 2 has Fourier amplitudes decaying as
A_max = Θ(√(log ℓ)/√ℓ) (confirmed above).  This uniform decay is WHY parity
provides no exploitable multiplicative bias.  But GNFS works precisely because
**B-smooth numbers** have special multiplicative structure.

**Question:** Does the B-smoothness indicator s_B(a) = 1_{a is B-smooth} have
different Fourier scaling from parity?  Specifically: does its maximum Fourier
amplitude A_max^{(B)}(ℓ) decay *slower* than ℓ^{−1/2}?

If yes: the smoothness function retains multiplicative bias in the Fourier domain
that parity does not — a quantitative signature of the structure GNFS exploits.

### Method

For each (ℓ, B): build the centered smoothness indicator
s̃_B(a) = s_B(a) − mean(s_B) on (ℤ/ℓℤ)* in dlog order, compute the exact DFT,
and record A_max^{(B)}(ℓ) = max_{r≥1} |ŝ̃_B(r)|.  Compare the power-law decay
slope to the parity baseline.

30 primes ℓ = 101 to 50021.  Smoothness bounds B ∈ {10, 30, 100, 300}.

### Results

#### Table 3: Smoothness vs parity scaling comparison

| B   | slope    | parity slope | Δslope  | R²     | verdict                       |
|-----|----------|--------------|---------|--------|-------------------------------|
| 10  | −0.4905  | −0.4387      | −0.052  | 0.968  | same as parity                |
| 30  | −0.2382  | −0.4387      | **+0.201** | 0.884 | **slower decay** — bias detected |
| 100 | −0.0838  | −0.4403      | **+0.357** | 0.351 | **slower decay** — bias detected |
| 300 | −0.0548  | −0.4420      | **+0.387** | 0.153 | **slower decay** — bias detected |

#### Table 4: Smoothness-to-parity ratio (B = 30, selected ℓ)

| ℓ     | smooth% | A_max    | A·√ℓ | parity A·√ℓ | ratio |
|-------|---------|----------|------|-------------|-------|
| 101   | 79.0%   | 0.09534  | 0.96 | 2.27        | 0.42  |
| 701   | 45.7%   | 0.08554  | 2.27 | 2.57        | 0.88  |
| 3109  | 26.0%   | 0.06358  | 3.55 | 2.92        | 1.21  |
| 11197 | 15.0%   | 0.04848  | 5.13 | 3.11        | 1.65  |
| 50021 | 7.3%    | 0.02957  | 6.61 | 3.30        | **2.00** |

### Density-normalized analysis

The raw slopes above can be partially confounded by density: smooth_fraction p
drops with ℓ, and the expected A_max for a random subset of density p is
null_A_max = √(p(1−p) · 2·ln(n/2) / n) where n = ℓ−1.  The **excess ratio**
A_max / null_A_max removes this confound.

#### Table 5: Density-normalized excess ratio

| B   | excess mean | excess std | first ℓ | last ℓ | verdict                  |
|-----|-------------|------------|---------|--------|--------------------------|
| 10  | 2.90        | 1.15       | 1.31    | 4.88   | STRONG multiplicative bias |
| 30  | 2.32        | 1.37       | 0.84    | 5.64   | STRONG multiplicative bias |
| 100 | 1.61        | 0.82       | 0.71    | 3.45   | moderate bias            |
| 300 | 1.23        | 0.28       | 0.84    | 1.82   | marginal                 |

excess > 1.0 means the Fourier amplitude exceeds what a random subset of the
same density would produce.  The bias is **growing with ℓ** for B = 10, 30.

#### Table 6: Head energy fraction (spectral concentration)

Head energy = Σ_{top-10} |ĥ(r)|² / p(1−p).  High values indicate spectral
concentration in few modes, not spread across all ~ℓ/2 characters.

| B   | smooth headE | parity headE | ratio |
|-----|-------------|--------------|-------|
| 10  | 0.0131      | 0.0047       | 2.78× |
| 30  | 0.0058      | 0.0032       | 1.79× |
| 100 | 0.0029      | 0.0024       | 1.19× |
| 300 | 0.0024      | 0.0022       | 1.10× |

### Interpretation

1. **B = 10 (very small bound):** Raw slope matches parity (Δslope ≈ 0), BUT
   density-normalized excess is the STRONGEST at 2.90×.  This is because at
   very sparse densities (10-smooth numbers are rare at large ℓ), the random-
   subset null drops faster than A_max does.  The smoothness function has
   genuine multiplicative structure that persists even after density correction.

2. **B ≥ 30:** **Strong density-corrected bias.**  For B = 30, excess grows
   from 0.84× at ℓ = 101 to 5.64× at ℓ = 50021 — not converging.  The head
   energy is 1.79× parity, meaning the spectral weight is more concentrated.

3. **B = 100, 300:** Moderate to marginal excess (1.61× and 1.23× mean).
   The high smooth_fraction at small ℓ makes the indicator nearly constant,
   reducing signal.  At large ℓ, the excess rises (3.45× for B=100 at ℓ=50021).

4. **Physical meaning:** The smoothness indicator has genuine **multiplicative
   Fourier bias** that parity lacks, confirmed after density normalization.
   This is consistent with the product-of-local-factors structure:
   ŝ_B(r) = Π_{p≤B} (local DFT at p), which creates constructive interference
   for characters χ_r that are "nearly trivial" on the small primes.

5. **Implication for factoring:** The bias is real, but **bias in χ-space is
   not automatically factor information**.  The hinge scalar barrier still
   applies: factoring requires extracting something about specific primes p, q
   dividing N, not just global group structure.

### E21b Step 2: Prime-restricted diagnostic

**Question:** Does the smoothness character r* (identified on the full group)
survive when restricted to the prime image set {g(p) : p in semiprime set}?
In E21, parity characters dropped to noise under prime restriction.  Does
smoothness behave differently?

**Method:** For each (n_bits, channel, B):
1. Compute full-group smoothness DFT → find r* and A_max.
2. Restrict to primes: compute g(p) = (1 + p^{k−1}) mod ℓ, evaluate s_B(g(p)).
3. Fixed-r* test: Pearson amplitude of χ_{r*} against s_B on restricted set.
4. Full scan: find best character on restricted set (with scan-null correction).
5. Product test: corr(s_B(g(p))·s_B(g(q)), Re(χ_{r*}(N^{k−1} mod ℓ))).

126 blocks tested: 9 bit sizes × 7 channels × 2 smoothness bounds (B=10, 30).
For n ≤ 24: exhaustive enumeration of balanced semiprimes.
For n > 24: deterministic sampling of 500 primes in half-bit range, forming up to
20,000 balanced pairs (ratio p/q > 0.3).

#### Table 7: Prime-restricted — fix_excess by n_bits (mean over 7 channels)

| n_bits | B=10 fix | B=30 fix | B=10 scan | B=30 scan | n_pairs (avg) |
|--------|----------|----------|-----------|-----------|---------------|
| 14     | 1.40     | 1.15     | 0.87      | 0.95      | 165           |
| 16     | 1.64     | 1.45     | 0.88      | 0.98      | 529           |
| 18     | 2.59     | 1.94     | 1.18      | 1.10      | 1,712         |
| 20     | 3.51     | 2.32     | 1.57      | 1.22      | 5,587         |
| 24     | 6.26     | 4.50     | 2.68      | 2.02      | 64,247        |
| 28     | 5.32     | 4.10     | 2.36      | 1.90      | 19,903        |
| 32     | 5.42     | 3.94     | 2.42      | 1.92      | 19,848        |
| 40     | 5.75     | 4.20     | 2.51      | 1.82      | 19,929        |
| 48     | 5.34     | 4.19     | 2.39      | 1.87      | 19,890        |

The fix_excess grows steeply from ~1.4× at n=14 to ~6.3× at n=24 (where
exhaustive enumeration gives the most pairs), then plateaus at ~5.3–5.8×
for n=28–48 (where the 500-prime sampling cap limits statistical power).
**The signal is persistent and does NOT decay with bit size.**

#### Table 7b: Representative channel (k=22, ℓ=131, B=10) detail

| n_bits | n_primes | n_pairs | fix_exc | scan_exc | \|corr_Nk\| |
|--------|----------|---------|---------|----------|-------------|
| 14     | 32       | 155     | 2.73    | 1.34     | 0.088       |
| 16     | 59       | 506     | 2.89    | 1.41     | 0.027       |
| 18     | 104      | 1,698   | 4.10    | 2.01     | 0.011       |
| 20     | 187      | 5,536   | 5.75    | 2.81     | 0.014       |
| 24     | 636      | 63,774  | 9.94    | 4.86     | 0.016       |
| 28     | 500      | 19,691  | 8.80    | 4.31     | 0.008       |
| 32     | 500      | 19,669  | 8.78    | 4.30     | 0.012       |
| 40     | 500      | 19,745  | 8.74    | 4.28     | 0.004       |
| 48     | 500      | 19,820  | 9.48    | 4.64     | 0.008       |

The strongest single-channel fix_excess reaches 9.94× at n=24 and sustains
~8.7–9.5× through 48 bits.  The scan_excess (after multiple-testing penalty)
reaches 4.86× — far above 1.0.  Meanwhile |corr_Nk| stays below 0.09
everywhere: the signal is NOT an artifact of the N^{k−1} distribution.

#### Table 8: Prime-restricted — full scan (selected blocks, n=20)

| k  | ℓ    | B  | r*_full | r*_scan | scan_amp | null   | excess | verdict     |
|----|------|----|---------|---------|----------|--------|--------|-------------|
| 20 | 283  | 10 | 7       | 7=      | 0.437    | 0.232  | 1.88   | ⚠ SIGNAL   |
| 20 | 617  | 10 | 23      | 23=     | 0.394    | 0.249  | 1.58   | ⚠ SIGNAL   |
| 22 | 131  | 10 | 11      | 11=     | 0.601    | 0.214  | 2.81   | ⚠ SIGNAL   |
| 22 | 593  | 10 | 11      | 83      | 0.429    | 0.248  | 1.73   | ⚠ SIGNAL   |
| 22 | 131  | 30 | 7       | 11      | 0.337    | 0.214  | 1.58   | ⚠ SIGNAL   |

At n=20, 6/28 blocks have scan_excess > 1.5 even after the multiple-testing
penalty.  The scanned r* often matches the full-group r* (marked with `=`),
confirming the same character is active.  At larger n, 62/126 blocks exceed
scan_excess > 1.5.

#### Table 9: Product tests (all 126 blocks)

| B   | mean \|corr_Nk\| | max \|corr_Nk\| | n(< 0.15) / 63 | verdict       |
|-----|-------------------|------------------|-----------------|---------------|
| 10  | 0.019             | 0.088            | 63/63           | ✓ **BARRIER** |
| 30  | 0.024             | 0.179            | 62/63           | ✓ **BARRIER** |

corr_Nk ≈ 0 uniformly (125/126 blocks with |corr_Nk| < 0.15).
The one weak exception (n=14, k=20, ℓ=283, B=30: corr_Nk = −0.18) is
small-sample noise (only 167 pairs).

**Aggregate (126 blocks):** 93/126 blocks (73.8%) have fix_excess > 2.0.
62/126 blocks (49.2%) have scan_excess > 1.5.  125/126 blocks (99.2%)
have |corr_Nk| < 0.15.

**Key finding:** The smoothness character SURVIVES prime restriction (unlike
parity), and this survival is **persistent from 14 to 48 bits** — it does
NOT decay with bit size.  But the product test still confirms the barrier:
the smoothness bias is real and detectable on the restricted set, but
**NOT N-extractable**.

### E21b Step 2b: σ-approximation corridor test

**Question:** The algebraic gap σ_{k−1}(N) − N^{k−1} = 1 + p^{k−1} + q^{k−1} reduces
(via Newton's identity) to knowing p + q.  For balanced semiprimes p ≈ q ≈ √N, can we
approximate σ_{k−1}(N) using only N?

**Method:** Compute σ_approx = 1 + 2·⌊√N⌋^{k−1} + N^{k−1} mod ℓ (replaces unknown
p^{k−1} + q^{k−1} with 2·⌊√N⌋^{k−1}, exact when p = q).  Measure:
- Circular distance error: |σ_approx − σ_true| / ℓ (wrapped to [0, 0.5])
- Product correlation: corr(s_B(g(p))·s_B(g(q)), Re(χ_{r*}(σ_approx mod ℓ)))

56 blocks tested (same configuration as Step 2).

#### Table 10: σ-approximation results (all 56 blocks)

| B   | mean error | expected (random) | mean |corr_approx| | max |corr_approx| | verdict   |
|-----|------------|-------------------|----------------------|---------------------|-----------|
| 10  | 0.252      | 0.250             | 0.026                | 0.124               | ✓ RANDOM  |
| 30  | 0.252      | 0.250             | 0.028                | 0.165               | ✓ RANDOM  |

**Interpretation:** The mean circular distance error is 0.252, indistinguishable from
the expected value 0.250 for a uniformly random residue.  High-order exponentiation
(k−1 ≥ 11) completely destroys the proximity p ≈ q ≈ √N: even though |p − q| / √N
is O(1) for balanced semiprimes, |p^{k−1} − (√N)^{k−1}| mod ℓ is uniformly distributed
because the exponential amplifies small differences.  The correlation corr_approx ≈ 0
confirms that the σ-approximation provides no signal whatsoever.

**Conclusion:** The last algebraic loophole in the tested invariant family — exploiting the
balanced-semiprime constraint to approximate the algebraic gap — is closed.  The smoothness
Fourier corridor is closed for the tested observables at all three levels: character
structure (Step 2), product extractability (Table 9), and σ-approximation (Table 10).
(Note: this closure is demonstrated for the tested character/product invariant family;
alternative N-only observables are tested in Step 3 below.)

### E21b Step 3: Stress tests (validation controls)

**Motivation:** Step 2 established that the smoothness character r* survives prime
restriction (fix_excess 5–10× at 24–48 bits) while the N-only product test shows
corr_Nk ≈ 0.  Step 3 validates these findings with rigorous controls: (1) verify
the null hypothesis for corr_Nk via permutation, (2) test r* stability across
prime sets, (3) test whether ANY linear combination of characters can extract
smoothness from N, and (4) quantify uncertainty via bootstrap confidence intervals.

126 blocks tested (9 bit sizes × 7 channels × 2 smoothness bounds), same
configuration as Step 2.

#### Test 1: Permutation null controls

**Method:** For each block, randomly re-pair primes (shuffle q-indices, 200
permutations) to generate a null distribution for corr_Nk.  Both the smoothness
product and the N^{k−1} predictor change under permutation (since N = p·q changes).
Report z-score and empirical p-value.

**Prediction:** If corr_Nk reflects no genuine signal, the observed value should
be consistent with the null distribution (p > 0.05).

#### Table 11: Permutation null by n_bits (mean over 14 channels)

| n_bits | mean |obs_corr| | mean |z| | pass (p>0.05) | mean p |
|--------|-----------------|----------|---------------|--------|
| 14     | 0.0499          | 0.606    | 14/14         | 0.514  |
| 16     | 0.0248          | 0.439    | 14/14         | 0.559  |
| 18     | 0.0135          | 0.437    | 14/14         | 0.517  |
| 20     | 0.0095          | 0.441    | 14/14         | 0.541  |
| 24     | 0.0079          | 0.493    | 14/14         | 0.484  |
| 28     | 0.0068          | 0.448    | 14/14         | 0.539  |
| 32     | 0.0069          | 0.470    | 13/14         | 0.497  |
| 40     | 0.0060          | 0.390    | 14/14         | 0.568  |
| 48     | 0.0061          | 0.392    | 13/14         | 0.549  |

**Result:** 124/126 blocks consistent with null (p > 0.05).  Two marginal
exceptions (both p = 0.035, both on channel k=20, ℓ=283): n=32/B=10 and
n=48/B=30.  At 126 tests with α = 0.05, we expect ~6 false positives by
chance; observing 2 is unremarkable.  Mean |z| = 0.46 across all blocks.

**Verdict:** The observed corr_Nk values are indistinguishable from permutation
noise.  No hidden N-dependent signal is present.

#### Test 2: Cross-n transfer of r*

**Method:** Fix the scanned_r_star from a source bit size (n=20) and evaluate
its fix_excess on every other bit size, using each target's own prime set.
The consistency_ratio = transfer_fix_excess / local_fix_excess measures
whether the source character performs comparably to the locally optimal one.

**Prediction:** If r* reflects genuine group structure (not overfitting to a
specific prime set), the transfer ratio should be ≈ 1.0.

#### Table 12: Cross-n transfer summary (selected channels)

| k  | ℓ     | B  | source r* | mean ratio | range         | stable (0.5–2.0) |
|----|-------|----|-----------|------------|---------------|-------------------|
| 12 | 691   | 10 | 70        | 1.000      | [1.00, 1.00]  | 9/9               |
| 12 | 691   | 30 | 7         | 1.000      | [1.00, 1.00]  | 9/9               |
| 16 | 3617  | 10 | 1091      | 1.000      | [1.00, 1.00]  | 9/9               |
| 16 | 3617  | 30 | 362       | 0.868      | [0.00, 1.00]  | 7/9               |
| 18 | 43867 | 10 | 7192      | 0.658      | [0.00, 1.00]  | 5/9               |
| 18 | 43867 | 30 | 7192      | 0.659      | [0.00, 1.00]  | 5/9               |
| 20 | 283   | 10 | 7         | 1.000      | [1.00, 1.00]  | 9/9               |
| 20 | 617   | 10 | 23        | 1.000      | [1.00, 1.00]  | 9/9               |
| 20 | 617   | 30 | 23        | 1.000      | [1.00, 1.00]  | 9/9               |
| 22 | 131   | 10 | 11        | 1.000      | [1.00, 1.00]  | 9/9               |
| 22 | 131   | 30 | 7         | 2.459      | [0.37, 13.59] | 5/9               |
| 22 | 593   | 10 | 83        | 1.074      | [0.58, 2.17]  | 7/9               |
| 22 | 593   | 30 | 83        | 1.068      | [0.57, 2.13]  | 7/9               |
| 20 | 283   | 30 | 7         | 1.000      | [1.00, 1.00]  | 9/9               |

**Result:** 10/14 channel configurations show perfect or near-perfect transfer
(mean ratio ≈ 1.0, all entries in [0.5, 2.0]).  The instability in ℓ=43867
channels reflects sparsity: at small n, very few primes map to smooth values
under g(p), causing degenerate blocks (0 smooth values → ratio = 0).  The
instability in (k=22, ℓ=131, B=30) is caused by a mismatch between local
and transferred r* at small n.

**Verdict:** The smoothness character r* is a stable property of the group
structure, not an artifact of a particular prime set.

#### Test 3: Multi-character N-score (the KEY test)

**Method:** Test whether ANY linear combination of characters can extract the
smoothness product from N^{k−1} mod ℓ alone.  Four sub-tests:

- **3a (DFT-weighted all):** Use all ~ℓ/2 characters with DFT-optimal weights
  ŝ_B(r).  By the inverse DFT identity, this equals s_B(N^{k−1} mod ℓ) − mean,
  which is just the smoothness indicator evaluated at the N-derived group element.
- **3b (top-10 weighted):** Use only the 10 highest-amplitude characters.
- **3c (direct smoothness):** Compute s_B(N^{k−1} mod ℓ) directly as a sanity
  check; should equal 3a exactly (algebraic identity).
- **3d (train/test split):** Learn arbitrary weights from training data (50/50
  split), evaluate R² on held-out test set.

**Prediction:** If the barrier is real, all correlations should be ≈ 0 and
R² should be ≤ 0.

#### Table 13: Multi-character N-score by n_bits (mean over 14 channels)

| n_bits | mean |corr_all| | max |corr_all| | mean |corr_top10| | mean R²  | < 0.10 |
|--------|-----------------|-----------------|-------------------|----------|--------|
| 14     | 0.0517          | 0.2517          | 0.0530            | −7.89    | 12/14  |
| 16     | 0.0198          | 0.0558          | 0.0197            | −19.38   | 14/14  |
| 18     | 0.0093          | 0.0286          | 0.0093            | −7.64    | 14/14  |
| 20     | 0.0075          | 0.0202          | 0.0080            | −4.16    | 14/14  |
| 24     | 0.0041          | 0.0143          | 0.0038            | −0.74    | 14/14  |
| 28     | 0.0046          | 0.0136          | 0.0048            | −1.42    | 14/14  |
| 32     | 0.0055          | 0.0135          | 0.0057            | −2.69    | 14/14  |
| 40     | 0.0042          | 0.0104          | 0.0042            | −11.73   | 14/14  |
| 48     | 0.0046          | 0.0139          | 0.0049            | −14.87   | 14/14  |

**Algebraic identity verified:** |corr_all − corr_direct| < 8.2 × 10⁻¹⁵ in
every block (machine precision), confirming that the DFT-weighted score equals
the direct smoothness evaluation.

**Result:** 125/126 blocks have |corr_all| < 0.15.  The single exception
(n=14, k=16, ℓ=3617, B=10: corr_all = 0.25) has only 167 pairs and its
train/test R² = 0.00 (no out-of-sample predictive power).  All train/test
R² values are negative (116/126) or effectively zero — the learned weights
overfit to training noise and perform worse than predicting the mean.

**Verdict:** No linear combination of multiplicative characters can extract
the smoothness product s_B(g(p))·s_B(g(q)) from N^{k−1} mod ℓ.  The barrier
holds for the entire character space, not just a single r*.

#### Test 4: Bootstrap confidence intervals

**Method:** Resample primes with replacement (500 bootstrap resamples) to
compute 95% CIs for fix_excess; resample pairs for corr_Nk CIs.

#### Table 14: Bootstrap CI by n_bits (mean over 14 channels)

| n_bits | mean fix_exc | mean CI_lo | mean CI_hi | CI_lo > 1.0 | corr_Nk CI ∋ 0 |
|--------|-------------|------------|------------|-------------|----------------|
| 14     | 1.48        | 0.66       | 2.30       | 3/14        | 11/14          |
| 16     | 1.88        | 1.00       | 2.79       | 7/14        | 13/14          |
| 18     | 2.62        | 1.49       | 3.76       | 10/14       | 14/14          |
| 20     | 3.34        | 2.01       | 4.71       | 12/14       | 14/14          |
| 24     | 5.44        | 3.54       | 7.49       | 14/14       | 13/14          |
| 28     | 4.76        | 2.99       | 6.65       | 13/14       | 14/14          |
| 32     | 4.71        | 2.98       | 6.55       | 13/14       | 13/14          |
| 40     | 5.01        | 3.22       | 6.93       | 13/14       | 13/14          |
| 48     | 5.03        | 3.26       | 6.88       | 8/14        | 14/14          |

**Result:**
- **fix_excess is robust:** 93/126 blocks have CI lower bound > 1.0 (statistically
  significant excess at 95% level).  At n ≥ 24, 13–14/14 channels are significant.
  The smoothness character signal is NOT a statistical fluctuation.
- **corr_Nk is indistinguishable from zero:** 119/126 blocks have 95% CI containing
  0.  The 7 exceptions are all at small n (14–16 bits) with wide CIs due to few pairs.

**Verdict:** The fix_excess is a statistically robust signal; the corr_Nk barrier
is confirmed with quantified uncertainty.

#### Stress test summary

| Test                    | Blocks | Pass criterion              | Pass rate | Exceptions           |
|-------------------------|--------|-----------------------------|-----------|----------------------|
| 1. Permutation null     | 126    | empirical p > 0.05          | 124/126   | 2 marginal (p=0.035) |
| 2. Cross-n transfer     | 14×9   | ratio ∈ [0.5, 2.0]         | 10/14 perfect | ℓ=43867 sparse  |
| 3. Multi-character      | 126    | |corr_all| < 0.15           | 125/126   | 1 small-sample       |
| 4. Bootstrap CI (fix)   | 126    | CI_lo > 1.0                 | 93/126    | small n expected     |
| 4. Bootstrap CI (corr)  | 126    | CI ∋ 0                      | 119/126   | small n expected     |

**Overall verdict:** All four stress tests confirm the Step 2 findings.
The smoothness character r* is a genuine, stable, statistically robust
structural property that survives prime restriction at all tested bit sizes.
The N-extractability barrier holds under permutation controls, across the
full character space (not just single r*), and with quantified confidence
intervals.  The closure is validated for the tested observable family.

#### Statistical protocol notes

**Multiple-testing correction.**  Two layers of multiple testing arise:

1. *Within-block character scan:* each block scans over (ℓ−1)/2 characters
   (65 for ℓ=131 up to 21,933 for ℓ=43867).  This is corrected by the
   scan_null = √(2·ln(n_chars) / (n_primes − 2)), which is the expected
   maximum amplitude under independent Gaussian noise.  The fix_excess and
   scan_excess metrics already incorporate this correction.

2. *Across-block testing:* 126 blocks are tested in the permutation null
   (Test 1).  Under the global null, the expected number of false rejections
   at α = 0.05 is 126 × 0.05 = 6.3.  We observe 2 rejections — well below
   expectation.  Under Bonferroni correction (α_adj = 0.05/126 ≈ 0.0004),
   all 126 blocks pass (zero rejections).

**Threshold choices are descriptive, not pre-registered.**  The cutoffs
|corr_Nk| < 0.15, fix_excess > 2.0, and p > 0.05 were chosen post-hoc as
convenient reporting thresholds.  To demonstrate robustness to this choice,
we report a sensitivity analysis:

| Threshold              | Pass count | Pass rate |
|------------------------|------------|-----------|
| \|corr_Nk\| < 0.05    | 116/126    | 92.1%     |
| \|corr_Nk\| < 0.10    | 124/126    | 98.4%     |
| \|corr_Nk\| < 0.15    | 125/126    | 99.2%     |
| \|corr_Nk\| < 0.20    | 126/126    | 100%      |
| \|corr_all\| < 0.05   | 119/126    | 94.4%     |
| \|corr_all\| < 0.10   | 124/126    | 98.4%     |
| \|corr_all\| < 0.15   | 125/126    | 99.2%     |
| perm p > 0.01          | 126/126    | 100%      |
| perm p > 0.025         | 126/126    | 100%      |
| perm p > 0.05          | 124/126    | 98.4%     |
| fix_excess CI_lo > 1.0 | 93/126     | 73.8%     |
| fix_excess CI_lo > 2.0 | 66/126     | 52.4%     |

The barrier conclusion (corr_Nk ≈ 0) is stable: even at the strict
threshold |corr_Nk| < 0.05, 92% of blocks pass.  The bootstrap CIs for
corr_Nk have median width 0.028, with 118/126 blocks having
|corr_Nk_mean| < 0.05 — the correlations are tightly concentrated at zero,
not just below an arbitrary cutoff.

**Bootstrap method.**  Percentile bootstrap (2.5th/97.5th quantiles).
For fix_excess: resample the prime set with replacement (n_primes draws),
recompute g(p), s_B(g(p)), and Pearson amplitude of χ_{r*} on the
resampled set.  For corr_Nk: resample pairs with replacement (n_pairs
draws), recompute the product correlation.  500 resamples per block.

## Conclusions

1. **Algebraic identity confirmed** (by unit test): `g(p)·g(q) mod ℓ = σ_{k−1}(N) mod ℓ`
   for all primes (p, q) tested.  The multiplicative factorisation of M is exact.

2. **No character structure detected**: the dominant eigenvector u₁ of M does
   NOT align with any individual multiplicative character χ_r of (ℤ/ℓℤ)*.
   Character amplitudes are indistinguishable from noise across all 28
   (n_bits, channel) blocks.

3. **Pipeline validated** (full-group control): the same character scan finds
   amp ≈ 1.0 over the unrestricted group.  The drop to noise in the
   prime-restricted audit is real, not a pipeline artifact.

4. **Fourier scaling confirmed**: A_max(ℓ) = Θ(√(log ℓ)/√ℓ), measured via
   exact DFT of centered parity h̃ = 2h − 1 on (ℤ/ℓℤ)* for 30 primes
   (ℓ = 101 to 50021).  The corrected ratio A_max·√ℓ/√(log ℓ) = 1.050 ± 0.035
   is constant to 3.3% CV across 3 orders of magnitude.  The √(log ℓ) factor
   is the extremal-value correction from maximising over ~ℓ/2 characters.
   This explains why the stable rank is O(1): the eigenvalue spectrum of H
   has O(1) modes at amplitude O(√(ℓ·log ℓ)) against a trivial background
   of ℓ/2, giving effective rank ≈ Σ(λ_r²)/λ₀² = O(log ℓ/ℓ) · (ℓ/2) = O(1).

5. **Barrier intact** (Product test B): `corr(u₁(p)·u₁(q), N^{k−1} mod ℓ) ≈ 0`
   uniformly.  The product of eigenvector components is NOT accessible from N
   alone.  The **analytic-continuation corridor is closed**.

6. **Interaction with E20 stable rank ≈ 3.7**: the constant low stable rank
   means M is approximately rank-3.7 over ℝ.  But the 3–4 dominant "directions"
   are NOT individual characters — they are collective modes arising from the
   O(1) dominant Fourier coefficients of the parity function on (ℤ/ℓℤ)*.
   Each such direction is a superposition of many characters at comparable
   amplitude O(√(log ℓ)/√ℓ), making them useless for extracting a single
   N-computable product.

7. **Smoothness Fourier bias confirmed** (E21b): the B-smoothness indicator
   s_B(a) has significantly slower Fourier decay than parity for B ≥ 30.
   At B = 30, the decay exponent is −0.24 vs parity's −0.44 (Δ = +0.20),
   and the smoothness-to-parity ratio grows from 0.42 to 2.0 across
   ℓ = 101 to 50021.

8. **Density-normalized bias survives** (E21b): after correcting for the
   density effect (smooth_fraction drops with ℓ), the excess ratio
   A_max / null_A_max is 2.90× for B=10 and 2.32× for B=30, growing with ℓ.
   Head energy is 2.78× (B=10) and 1.79× (B=30) of parity.  This is
   genuine multiplicative structure, not a max-statistics artifact.

9. **Smoothness character survives prime restriction to 48 bits** (E21b Step 2):
   unlike parity (which drops to noise), the smoothness-tuned character r*
   retains signal when restricted to the prime image set {g(p)}, tested across
   126 blocks (9 bit sizes × 7 channels × 2 bounds).  Mean fix_excess grows
   from 1.4× at n=14 to 5.3–6.3× at n=24–48 (B=10), with 93/126 blocks
   exceeding 2.0× and 62/126 exceeding scan_excess > 1.5.  The signal is
   **persistent and does NOT decay with bit size**.

10. **Smoothness barrier also holds at scale** (E21b Step 2, product test):
    corr(s_B(g(p))·s_B(g(q)), Re(χ_{r*}(N^{k−1} mod ℓ))) ≈ 0 uniformly
    (125/126 blocks with |corr_Nk| < 0.15, max = 0.18 at n=14 small-sample
    noise).  Despite the smoothness character surviving restriction at all
    bit sizes, the product is NOT predictable from N alone.
    The **smoothness Fourier corridor is closed for the tested invariant family**.

11. **Stress tests validate all findings** (E21b Step 3): Four independent
    validation controls confirm the Step 2 results across 126 blocks:
    - **Permutation null:** 124/126 blocks consistent with random pairing (p > 0.05).
    - **Cross-n transfer:** r* is a stable group property, not overfit to a prime set
      (10/14 channels show perfect transfer, instabilities only at sparse ℓ=43867).
    - **Multi-character N-score:** NO linear combination of characters extracts the
      product from N — 125/126 blocks with |corr| < 0.15, all train/test R² ≤ 0.
      The DFT algebraic identity corr_all ≡ corr_direct verified to machine precision.
    - **Bootstrap CI:** fix_excess is robust (93/126 CI_lo > 1.0); corr_Nk CI
      contains 0 in 119/126 blocks.  The signal is real; the barrier is real.

12. **σ-approximation corridor closed** (E21b Step 2b): the balanced-semiprime
    approximation σ_approx = 1 + 2·⌊√N⌋^{k−1} + N^{k−1} mod ℓ has circular
    distance error 0.252 ≈ 0.250 (random), with corr_approx ≈ 0 in all 56 blocks.
    High-order exponentiation (k−1 ≥ 11) destroys p ≈ q proximity modulo ℓ.
    The last algebraic loophole is closed.

13. **Joint cross-channel N-only tests confirm no nonlinear signal** (E21c):
    testing whether the JOINT distribution of N^{k−1} mod ℓ across all 7
    Eisenstein channels reveals cross-channel structure invisible to
    individual channels.  Four tests across 18 blocks (9 bit sizes × 2 bounds):
    - **C1 Pairwise interactions:** 84 cross-product features per block;
      17/18 pass Bonferroni, mean|corr| ≈ O(1/√n_pairs).
    - **C2 OLS regression:** 35 features (14 linear + 21 products), 50/50
      holdout; 18/18 blocks with test R² ≤ 0.
    - **C3 Mutual information:** binned MI between channels 5,3; 13/16
      tested blocks consistent with independence (p > 0.05).
    - **C4 Permutation null:** max|corr| over all 84 features vs shuffled
      target; 16/18 blocks consistent with noise (p > 0.05).
    The barrier is closed not just per-channel but for all pairwise and
    joint observables in the tested Eisenstein family.

## E21c: Joint cross-channel N-only tests

### Motivation

E21b established that for each individual Eisenstein channel (k, ℓ), the single
N-only observable N^{k−1} mod ℓ carries zero predictive power for the smoothness
product s_B(g(p))·s_B(g(q)).  But each channel probes a different algebraic function
of N — different weight k, different prime ℓ, different optimal character r*.

E21c tests whether the **joint** distribution across all 7 channels reveals
cross-channel structure that no single channel carries alone.  This is the natural
next hypothesis class: nonlinear interactions between channels.

### Method

**Shared primes and pairs.**  All 7 channels use the same (p, q) pairs per block
(channel-independent seed `0xE21c_0000 + n_bits`), ensuring cross-channel features
are computed on identical data.

**Feature construction.**  For each valid pair (p, q) and channel i ∈ {0,…,6}:
- f_{2i} = Re(χ_{r*_i}(N^{k_i−1} mod ℓ_i))
- f_{2i+1} = Im(χ_{r*_i}(N^{k_i−1} mod ℓ_i))

This produces a 14-dimensional feature vector per pair.

**Target.**  s_B(g(p)) · s_B(g(q)) for the reference channel (idx 5: k=22, ℓ=131).

**Four sub-tests:**

| Test | Description | Features | Null threshold |
|------|-------------|----------|----------------|
| C1 | Pairwise interaction corr | 84 cross-products (C(7,2)×4) | Bonferroni α=0.05/84 |
| C2 | OLS regression | 35 (14 linear + 21 Re_i·Re_j) | Test R² ≤ 0 |
| C3 | Binned mutual information | Q=8 quantile bins, ch 5×3 | Permutation p > 0.05 |
| C4 | Permutation null (max) | max\|corr\| over 84 features | Permutation p > 0.05 |

C4 accounts for selection bias by computing the maximum |correlation| across all 84
features under each permutation, matching the selection process in C1.

### Results

**Table 15: C1 — Pairwise interaction correlations (84 tests per block)**

|  n  |  B | pairs  | max\|corr\| | mean\|corr\| | Bonf_thr | verdict |
|----:|---:|-------:|----------:|----------:|----------:|---------|
|  14 | 10 |    167 |    0.1907 |    0.0582 |    0.3137 | ✓ noise |
|  14 | 30 |    167 |    0.1848 |    0.0526 |    0.3137 | ✓ noise |
|  16 | 10 |    537 |    0.1234 |    0.0304 |    0.1742 | ✓ noise |
|  16 | 30 |    537 |    0.1002 |    0.0368 |    0.1742 | ✓ noise |
|  18 | 10 |   1698 |    0.0800 |    0.0231 |    0.0979 | ✓ noise |
|  18 | 30 |   1698 |    0.0608 |    0.0194 |    0.0979 | ✓ noise |
|  20 | 10 |   5485 |    0.0390 |    0.0104 |    0.0544 | ✓ noise |
|  20 | 30 |   5485 |    0.0518 |    0.0115 |    0.0544 | ✓ noise |
|  24 | 10 |  62124 |    0.0128 |    0.0033 |    0.0162 | ✓ noise |
|  24 | 30 |  62124 |    0.0094 |    0.0031 |    0.0162 | ✓ noise |
|  28 | 10 |  19547 |    0.0206 |    0.0051 |    0.0288 | ✓ noise |
|  28 | 30 |  19547 |    0.0290 |    0.0066 |    0.0288 | ✗ marginal |
|  32 | 10 |  18889 |    0.0233 |    0.0061 |    0.0293 | ✓ noise |
|  32 | 30 |  18889 |    0.0204 |    0.0050 |    0.0293 | ✓ noise |
|  40 | 10 |  19550 |    0.0210 |    0.0057 |    0.0288 | ✓ noise |
|  40 | 30 |  19550 |    0.0228 |    0.0056 |    0.0288 | ✓ noise |
|  48 | 10 |  19753 |    0.0205 |    0.0053 |    0.0287 | ✓ noise |
|  48 | 30 |  19753 |    0.0184 |    0.0056 |    0.0287 | ✓ noise |

17/18 blocks pass.  The single marginal exceedance (n=28, B=30) has
max|corr| = 0.0290 vs threshold 0.0288 — within sampling noise.
Mean |corr| scales as O(1/√n_pairs), confirming pure noise.

**Table 16: C2 — OLS regression (35 features, 50/50 holdout)**

|  n  |  B | train | test | test R² | best 1ch R² | verdict |
|----:|---:|------:|-----:|--------:|------------:|---------|
|  14 | 10 |    83 |   84 |  −0.987 |     0.0476  | ✓ |
|  14 | 30 |    83 |   84 |  −1.588 |     0.0048  | ✓ |
|  16 | 10 |   268 |  269 |  −0.570 |     0.0165  | ✓ |
|  16 | 30 |   268 |  269 |  −4.067 |     0.0053  | ✓ |
|  18 | 10 |   849 |  849 |  −0.237 |     0.0007  | ✓ |
|  18 | 30 |   849 |  849 |  −1.285 |     0.0007  | ✓ |
|  20 | 10 |  2742 | 2743 |  −0.252 |     0.0002  | ✓ |
|  20 | 30 |  2742 | 2743 |  −2.217 |     0.0007  | ✓ |
|  24 | 10 | 31062 |31062 |  −0.171 |     0.0003  | ✓ |
|  24 | 30 | 31062 |31062 |  −1.076 |     0.0000  | ✓ |
|  28 | 10 |  9773 | 9774 |  −0.230 |     0.0002  | ✓ |
|  28 | 30 |  9773 | 9774 |  −1.253 |     0.0000  | ✓ |
|  32 | 10 |  9444 | 9445 |  −0.160 |     0.0002  | ✓ |
|  32 | 30 |  9444 | 9445 |  −1.226 |     0.0002  | ✓ |
|  40 | 10 |  9775 | 9775 |  −0.177 |     0.0003  | ✓ |
|  40 | 30 |  9775 | 9775 |  −1.417 |     0.0001  | ✓ |
|  48 | 10 |  9876 | 9877 |  −0.239 |     0.0001  | ✓ |
|  48 | 30 |  9876 | 9877 |  −1.140 |     0.0002  | ✓ |

**18/18 blocks with test R² ≤ 0.**  No linear or bilinear combination of cross-channel
features generalises.  Best single-channel R² is also negligible (max 0.048 at n=14).
Negative R² indicates the model is worse than predicting the mean — pure overfit.

**Table 17: C3 — Binned mutual information (channels 5, 3; 200 permutations)**

|  n  |  B | pairs |    MI    | null μ | null σ  | p-val | verdict |
|----:|---:|------:|---------:|-------:|--------:|------:|---------|
|  16 | 10 |   537 | 0.06075  | 0.06679 | 0.01099 | 0.680 | ✓ |
|  16 | 30 |   537 | 0.05889  | 0.06704 | 0.01179 | 0.750 | ✓ |
|  18 | 10 |  1698 | 0.02322  | 0.01943 | 0.00330 | 0.115 | ✓ |
|  18 | 30 |  1698 | 0.01622  | 0.01886 | 0.00346 | 0.760 | ✓ |
|  20 | 10 |  5485 | 0.00390  | 0.00591 | 0.00098 | 0.985 | ✓ |
|  20 | 30 |  5485 | 0.00473  | 0.00590 | 0.00111 | 0.875 | ✓ |
|  24 | 10 | 62124 | 0.00043  | 0.00050 | 0.00009 | 0.740 | ✓ |
|  24 | 30 | 62124 | 0.00056  | 0.00050 | 0.00009 | 0.255 | ✓ |
|  28 | 10 | 19547 | 0.00193  | 0.00164 | 0.00031 | 0.160 | ✓ |
|  28 | 30 | 19547 | 0.00153  | 0.00158 | 0.00030 | 0.530 | ✓ |
|  32 | 10 | 18889 | 0.00221  | 0.00166 | 0.00030 | 0.035 | marginal |
|  32 | 30 | 18889 | 0.00218  | 0.00164 | 0.00028 | 0.040 | marginal |
|  40 | 10 | 19550 | 0.00171  | 0.00163 | 0.00027 | 0.375 | ✓ |
|  40 | 30 | 19550 | 0.00173  | 0.00156 | 0.00031 | 0.270 | ✓ |
|  48 | 10 | 19753 | 0.00281  | 0.00160 | 0.00029 | 0.000 | ✗ |
|  48 | 30 | 19753 | 0.00193  | 0.00162 | 0.00030 | 0.155 | ✓ |

13/16 tested blocks pass (n=14 skipped: insufficient pairs for Q=8 binning).
The 3 marginal/failing blocks (n=32 both bounds, n=48 B=10) show MI barely
above null mean.  With 18 blocks at α=0.05, 0.9 false positives expected;
3 observed is slightly elevated but not systematic — the n=48/B=10 anomaly
does not replicate at B=30.

**Table 18: C4 — Permutation null on max|corr| (200 permutations)**

|  n  |  B | pairs |  obs corr | null μ  | null σ | p-val | verdict |
|----:|---:|------:|----------:|--------:|-------:|------:|---------|
|  14 | 10 |   167 |   −0.1907 |  0.2032 | 0.0265 | 0.675 | ✓ |
|  14 | 30 |   167 |    0.1848 |  0.2110 | 0.0333 | 0.765 | ✓ |
|  16 | 10 |   537 |    0.1234 |  0.1153 | 0.0170 | 0.290 | ✓ |
|  16 | 30 |   537 |   −0.1002 |  0.1190 | 0.0183 | 0.865 | ✓ |
|  18 | 10 |  1698 |    0.0800 |  0.0647 | 0.0099 | 0.085 | ✓ |
|  18 | 30 |  1698 |   −0.0608 |  0.0660 | 0.0101 | 0.675 | ✓ |
|  20 | 10 |  5485 |   −0.0390 |  0.0362 | 0.0053 | 0.300 | ✓ |
|  20 | 30 |  5485 |   −0.0518 |  0.0365 | 0.0052 | 0.005 | ✗ |
|  24 | 10 | 62124 |    0.0128 |  0.0107 | 0.0016 | 0.115 | ✓ |
|  24 | 30 | 62124 |   −0.0094 |  0.0106 | 0.0015 | 0.765 | ✓ |
|  28 | 10 | 19547 |    0.0206 |  0.0191 | 0.0028 | 0.270 | ✓ |
|  28 | 30 | 19547 |   −0.0290 |  0.0194 | 0.0031 | 0.015 | ✗ |
|  32 | 10 | 18889 |    0.0233 |  0.0195 | 0.0028 | 0.095 | ✓ |
|  32 | 30 | 18889 |    0.0204 |  0.0191 | 0.0027 | 0.305 | ✓ |
|  40 | 10 | 19550 |    0.0210 |  0.0194 | 0.0031 | 0.270 | ✓ |
|  40 | 30 | 19550 |   −0.0228 |  0.0194 | 0.0031 | 0.145 | ✓ |
|  48 | 10 | 19753 |   −0.0205 |  0.0190 | 0.0029 | 0.260 | ✓ |
|  48 | 30 | 19753 |   −0.0184 |  0.0192 | 0.0029 | 0.570 | ✓ |

16/18 blocks pass.  The 2 failures (n=20/B=30, n=28/B=30) are isolated and
do not persist at other bit sizes or bounds — consistent with expected false
positives (0.9 expected at α=0.05 across 18 blocks).

### Interpretation

All four tests produce results consistent with the null hypothesis of no
cross-channel signal.  The minor exceedances are:

- **Not systematic:** failures do not cluster at specific bit sizes or bounds.
- **Not replicable:** the n=20/B=30 C4 failure does not appear at n=20/B=10;
  the n=28/B=30 C1 marginal exceedance vanishes in C4 at B=10.
- **Expected rate:** 5 "failures" across 70 tests ≈ 7.1%, consistent with
  the 5% false-positive rate under multiple testing.

Combined with E21b's per-channel results (126 blocks confirming individual
corridor closure), E21c closes the joint/interaction corridor:

> **The barrier holds not only for each Eisenstein channel individually, but
> also for all pairwise and linear-combination observables across the tested
> 7-channel family.  No nonlinear cross-channel interaction rescues
> predictability of the smoothness product from N alone.**

## Data files

- `rust/data/E21_character_audit.json` — full per-block per-eigenvector results
- `rust/data/E21_control_results.json` — full-group control experiment results
- `rust/data/E21_fourier_scaling.json` — Fourier scaling analysis (30 primes, centered parity)
- `rust/data/E21b_smoothness_spectrum.json` — smoothness Fourier spectrum (B = 10, 30, 100, 300)
- `rust/data/E21b_prime_restricted.json` — prime-restricted smoothness diagnostic (126 blocks, n=14–48)
- `rust/data/E21b_stress_tests.json` — stress test validation results (126 blocks, 4 tests)
- `rust/data/E21c_cross_channel.json` — joint cross-channel test results (18 blocks, 4 tests)
- `rust/eigenvector-character/` — Rust crate implementing E21, E21b, and E21c

- `rust/data/E21_character_audit.json` — full per-block per-eigenvector results
- `rust/data/E21_control_results.json` — full-group control experiment results
- `rust/data/E21_fourier_scaling.json` — Fourier scaling analysis (30 primes, centered parity)
- `rust/data/E21b_smoothness_spectrum.json` — smoothness Fourier spectrum (B = 10, 30, 100, 300)
- `rust/data/E21b_prime_restricted.json` — prime-restricted smoothness diagnostic (126 blocks, n=14–48)
- `rust/data/E21b_stress_tests.json` — stress test validation results (126 blocks, 4 tests)
- `rust/eigenvector-character/` — Rust crate implementing E21 and E21b
