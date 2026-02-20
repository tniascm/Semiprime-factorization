# Semiprime Factorization via Automorphic Methods — Research Summary

## 1. Research Program Overview

**Question:** Can modular forms, trace formulas, and Langlands program constructs provide computational shortcuts for factoring semiprimes N = pq?

**Approach:** A series of computational experiments (E4–E8) systematically test whether various automorphic/arithmetic objects carry factor-related spectral information that is both *detectable* (peaks at factor frequencies) and *computable* (accessible without already knowing p, q).

**Recommended execution order:** E5 → E7 → E6 → E4 (refined during execution).

---

## 2. Experiment Results

### E5: Martin Dimension Formula (KILLED)

**File:** `E5_martin_dimension/run_E5.sage`

**What it tests:** Whether the dimension of spaces of modular forms S_k(Γ₀(N)) encodes factor information through its dependence on divisors of N.

**Result:** The dimension formula requires summing over all divisors of N. For semiprimes N = pq, this involves terms depending on p and q individually. Computing these without knowing the factors requires enumerating all d | N, which is equivalent to factoring. The computation scales as O(N^1.91).

**Verdict:** Killed — the access model is circular (needs factors to compute the formula that supposedly reveals factors).

---

### E7: Beyond Endoscopy / Orbital Integral DFT

**File:** `E7_altug_beyond_endoscopy/run_E7.sage`
**Data:** `data/E7_cancellation_results.json`

**What it tests:** DFT of the orbital integral O(t, N) = ∏_{p|N} (1 + kronecker(t²−4, p)) over the trace parameter t ∈ Z/NZ. Inspired by Altug's beyond-endoscopy approach to the trace formula.

**Key finding: 3-tier Fourier spectrum.** The DFT coefficients Ô(ξ) organize into three magnitude tiers based on gcd(ξ, N):
- **gcd(ξ, N) = q:** |Ô| ~ 1/√q (loud, q−1 modes)
- **gcd(ξ, N) = p:** |Ô| ~ 1/√p (loud, p−1 modes)
- **gcd(ξ, N) = 1:** |Ô| ~ 1/√N (bulk, (p−1)(q−1) modes)

Peak-to-bulk ratio ~ N^{1/4} for balanced semiprimes. This grows without bound.

**Critical caveat:** O(t, N) requires knowing the individual Legendre symbols L_p(t²−4) and L_q(t²−4), which requires knowing p and q. This is "oracle-conditional."

---

### E7b: Spectral Sparsity & Factor Extraction

**File:** `E7_altug_beyond_endoscopy/run_E7b_spectral_sparsity.sage`
**Data:** `data/E7b_spectral_sparsity_results.json`

**What it tests:** How sparse/concentrated the orbital DFT spectrum is, and whether the dominant modes encode factor information.

**Key findings:**
1. **Participation ratio ~ N^{0.507}** (effective spectral dimension grows as √N, not N). The spectrum is sparse.
2. **Exact 2/3 energy split:** E(gcd > 1) / E(total) = 2/3 to 6+ decimal places for all tested semiprimes. This is the (p−1 + q−1) / (N−1) fraction with a clean 2/3 limit for balanced p ≈ q.
3. **100% top-mode factor extraction:** For every semiprime tested, gcd(argmax|Ô(ξ)|, N) yields a non-trivial factor. The top DFT mode always sits at a factor-related frequency.
4. **Jacobi preliminary:** Replacing the oracle O(t,N) with J(t) = kronecker(t²−4, N) destroys all peaks.

---

### E7c: Jacobi-Only Spectral Probes

**File:** `E7_altug_beyond_endoscopy/run_E7c_jacobi_probes.sage`
**Data:** `data/E7c_jacobi_probes_results.json`

**What it tests:** Whether *any* nonlinear function of Jacobi-computable quantities has factor-related DFT peaks.

**10 observables tested:**

| Signal | Type | Peak Scaling | E(gcd>1) | Top-1 | Verdict |
|--------|------|:---:|:---:|:---:|---------|
| O(t,N) | oracle | N^{0.25} | 0.667 | 100% | Factor peaks — but oracle |
| L_p+L_q | oracle | N^{0.25} | 1.000 | 100% | Cuspidal residual — oracle |
| J(t) | computable | N^{−0.25} | ~0 | 0% | Anti-peak (shrinks with N) |
| T_+(t) | computable | ~1.0 | ~0.01 | 0% | Dead flat |
| T_−(t) | computable | ~1.0 | ~0.01 | 0% | Dead flat |
| \|J(t)\| | computable | N^{0.25} | ~0.99 | 100% | = gcd indicator (see below) |
| gcd>1 | computable | N^{0.25} | ~0.99 | 100% | Real peaks, but ≡ trial division |
| J·J(t+1) | computable | ~1.0 | ~0.01 | 0% | Dead flat |
| J·J(t+2) | computable | ~1.0 | ~0.01 | 0% | Dead flat |
| J×J₂ | computable | ~1.0 | ~0.01 | 0% | Dead flat |

**The CRT product structure argument:** For N = pq, any pointwise function of Jacobi symbols S(t) = Φ(kronecker(f₁(t), N), ..., kronecker(f_k(t), N)) factors as h(a)·g(b) in CRT coordinates (a mod p, b mod q). Its DFT is therefore flat: |Ŝ(ξ)| ~ 1/√N at all frequencies.

This covers J(t), T_±(t), shifted products J(t)·J(t+Δ), and cross-polynomial products.

**The gcd indicator exception:** G(t) = 1_{gcd(t²−4, N) > 1} has CRT form u(a)+v(b)−u(a)v(b) — a *sum*, not a product — creating real N^{1/4} peaks with R² = 0.9999. But finding when gcd(t²−4, N) > 1 is computationally equivalent to trial division.

**The quadratic residuosity bottleneck:** Among t values with J(t) = +1 and gcd(t²−4, N) = 1, distinguishing O(t,N) = 4 (both QR) from O(t,N) = 0 (both QNR) is the quadratic residuosity problem, believed equivalent to factoring.

---

### E6: Braverman-Kazhdan Transform

**File:** `E6_luo_ngo_kernel/run_E6.sage`
**Data:** `data/E6_bk_transform_results.json`

**What it tests:** Whether the BK "nonabelian Fourier transform" on GL₂ provides information beyond the abelian (Jacobi) setting.

**4-part analysis:**

**Part A — BK = Orbital equivalence:** At Iwahori level, the BK kernel evaluates to:

    f̂_p(t) = (1 + kronecker(t²−4, p)) / (p+1)

This is the orbital integral O_p(t) divided by (p+1). Since this factors over primes dividing N via an Euler product, BK(t, N) = O_p(t)/(p+1) · O_q(t)/(q+1). Confirmed numerically at N = 77 with exact agreement.

**Part B — GL₂ trace count LOSES signal:** The GL₂ conjugacy class count (number of elements with given trace) has factor-related features, but the energy fraction E(gcd>1) drops from 0.80 at N = 221 to 0.007 at N = 47053 — it gets diluted as N grows. Only 4 of 20 semiprimes had top-1 factor extraction.

**Part C — 2D (trace, det) extension:** Extending to the full conjugacy class parametrized by (trace, det) recovers ~97% factor energy, but this 2D observable is oracle-conditional (requires L_p, L_q separately).

**Part D — Access model test:** The "collapsed" BK transform (using J(t) = kronecker(t²−4, N) instead of the individual L_p, L_q) has peak ~0.4 vs oracle peak ~2.4. Factor signal is destroyed.

**Verdict:** E6 collapses entirely to E7. The "nonabelian" in nonabelian Fourier transform refers to the GL₂ group structure, not to any non-multiplicativity in the CRT sense. At finite places, everything factors over primes.

---

### E7d: Global Analytic Separators

**File:** `E7_altug_beyond_endoscopy/run_E7d_global_separators.sage`
**Data:** `data/E7d_global_separators_results.json`

**What it tests:** Whether "global" automorphic objects (theta functions, Kloosterman sums) avoid the CRT product obstruction.

**Part A — Theta function:**

θ(a/N) = Σ_n e^{2πi a n²/N} = N · DFT(h)(a)

where h(m) = #{n : n² ≡ m mod N} = (1 + L_p(m))(1 + L_q(m)).

Since h(m) is a CRT product of local Legendre data, the theta function has the same 3-tier spectrum as E7's orbital DFT. Correlation between theta and orbital peaks: R = 0.999. The Jacobi-collapsed version (using 1 + J(m) instead of (1+L_p)(1+L_q)) has peak ≈ 0 (spectrally dead).

**Part B — Kloosterman sums:**

S(m, 1; N) = Σ_{x coprime to N} e^{2πi(mx + x⁻¹)/N}

Multiplicative: S(m, 1; pq) = S(m mod p, 1; p) · S(m mod q, 1; q).

Results for N = 3127 (= 53 × 59): |S| at factor primes ≈ 7.2 (predicted ~ N^{1/4} ≈ 7.5), |S| at non-factors ≈ 41.8 (predicted ~ N^{1/2} ≈ 55.9). Factor primes rank in the bottom 10 of 46 tested primes by |S| magnitude.

**But:** Each Kloosterman sum evaluation costs O(N), and testing O(√N) candidate primes gives total cost O(N^{3/2}) — worse than trial division at O(√N).

**Part C — Blind detection:** Factor primes are consistently anomalously small, confirming the multiplicative structure. But the computational cost makes this useless.

---

### E7e: Analytic Proxy Tests

**File:** `E7_altug_beyond_endoscopy/run_E7e_analytic_proxies.sage`
**Data:** `data/E7e_analytic_proxies_results.json`

**What it tests:** Whether any computable analytic proxy — arithmetic weightings, Dirichlet character twists, multi-discriminant elimination, or functional equation reflection — can extract factor information from Jacobi-accessible data.

**13 signals tested, all failed (0% top-1 factor extraction):**
- J*Lambda (von Mangoldt weighting), J*mu (Mobius weighting), Gaussian-smoothed J
- J*chi_m twists for m in {3,4,5,7,8}
- Multi-discriminant elimination with 12 discriminants d=1,...,12 using equal weights, alternating weights, SVD projection, max-kurtosis optimization
- Cross-discriminant coherence analysis

**Key discoveries:**
1. Multi-discriminant anti-peaks scale as N^{-0.25} (R²=0.98-0.99): opposite of factor peaks
2. Jacobsthal sum gives f̂_d(0) = -1/p universally across discriminants, causing structural suppression
3. Coherence ratio factor/bulk = 0.92 — no cross-discriminant resonance at factor frequencies
4. J(t) = J(N-t) makes functional equation reflection trivial (zero information)

**Verdict:** Empirically closes the "cheap analytic proxy" corridor for single-/few-pass linear or mildly nonlinear signal transforms on Jacobi-accessible sequences.

---

### E8a: Twisted GL(2) L-function Tomography

**File:** `E8_global_projector/run_E8a_Lfunction_tomography.sage`
**Data:** `data/E8a_Lfunction_tomography_results.json`

**What it tests:** Whether a genuinely global analytic object — L(s, Delta x chi_N), the Ramanujan Delta function twisted by the Kronecker character mod N — carries factor information beyond gcd/trial-division through its global analytic properties (zeros, central value, functional equation).

**Setup:** Fixed form Delta (weight 12, level 1), twist chi_N = kronecker(., N). Evaluated using approximate functional equation near critical line (s=6+it) and partial sums in convergent region (s=8+it). 12 semiprimes from N=3127 to N=126727, 17 s-values, 5 confusable pairs.

**Results:**

| Test | Finding | Factor Signal? |
|------|---------|:-:|
| A. L-value profiles | Values computed at 17 s-points | Computed successfully |
| B. Confusable pairs | mag_diff 0.28-0.37, all same epsilon | Not systematically exploitable |
| C. Root number | epsilon = +1 for all 12 semiprimes | No factor signal beyond N mod 4 |
| D. Euler factor scoring | Factors always rank #1, #2 | YES — but = gcd test (chi_N(m)=0 iff m\|N) |
| E. L-value sensitivity | max |r| = 0.44 with p, PC1 explains 43% variance | Weak, no consistent direction |

**Critical structural observation:** chi_N(p) = 0 for p|N, so the Euler product at factor primes is trivial: L_p(s) = (1 + p^{11-2s})^{-1}. Testing chi_N(m) = 0 IS gcd computation. The global properties (zeros, central value) do depend on which primes are "missing," but extracting this would require resolving O(N log N) zeros to O(1/log N) precision — cost O(N²).

**Verdict:** The locally-computable part of L-function values reduces to trial division. Global coupling exists in principle but is not extractable at sub-exponential cost through L-value sampling.

---

### E8b: Multi-Form L-function Amplification

**File:** `E8_global_projector/run_E8b_multi_form_amplification.sage`
**Data:** `data/E8b_multi_form_results.json`, `data/E8b_r2_vs_K.png`

**What it tests:** Whether a FAMILY of twisted L-functions L(s, f_i x chi_N) for 19 level-1 Hecke eigenforms (weights 12–36) carries factor information that scales with the number of forms K. Treats factorization as a noisy-channel decoding problem.

**Setup:** 19 eigenforms, 10 s-values per form (7 critical line + 3 slightly right), 60 semiprimes (N up to 33043). Features: z-scored log|L|, PCA-reduced. Target: log(min(p,q)). Metric: LOOCV R² via ridge regression.

**Results:**

| K (forms) | R² | Null 95th | Significant? |
|:-:|:-:|:-:|:-:|
| 1 (k=12) | -0.005 | +0.060 | No |
| 5 | -0.020 | +0.053 | No |
| 10 | -0.028 | +0.057 | No |
| 14 (best) | +0.093 | +0.072 | Marginal |
| 19 (all) | +0.058 | +0.070 | No |

- R² trend: marginally increasing (first half mean -0.019, second half +0.038) but within noise
- Critical line more sensitive (R² = 0.21) than convergent region (R² = -0.03), confirming user's prediction
- Best individual correlation: |r| = 0.49 for k=36 forms at s = k/2 + 0.5i (p = 7×10⁻⁵), but does not generalize in LOOCV
- Root number ε = (-1)^{k/2} is independent of N for ALL level-1 forms twisted by quadratic χ_N (proven algebraically, not just empirical)

**Verdict:** No amplification. Adding forms does not increase factor information. The "noisy-channel decoding" model fails: each form carries the same ~0 bits of factor information, not independent noisy measurements. Closes the level-1 GL(2) × quadratic-twist amplification corridor.

---

### E4: Hirano Dijkgraaf-Witten Invariants (NOT YET EXECUTED)

**File:** `E4_hirano_dw_invariants/E4_hirano_mod2_dw.ipynb`

Notebook created but not executed. Consensus: likely hits the same CRT obstruction. Only worth running if it tests a genuinely different access model.

---

## 3. The Central Obstruction

All experiments converge to a single obstruction pattern:

### The CRT Product Structure

For N = pq, the tested automorphic quantities all factor into local components at finite places:

- **Orbital integral:** O(t, N) = O_p(t) · O_q(t) where O_p(t) = 1 + kronecker(t²−4, p)
- **BK transform:** f̂(t, N) = f̂_p(t) · f̂_q(t) (Euler product)
- **Theta function:** h(m) = (1 + L_p(m)) · (1 + L_q(m))
- **Kloosterman sums:** S(m, 1; pq) = S(m mod p, 1; p) · S(m mod q, 1; q)

The local components L_p(·), L_q(·) are individually computable if you know p, q — but from N alone, only the Jacobi symbol J(·) = L_p(·) · L_q(·) is accessible. This product collapses the factor-separated information.

### The Quadratic Residuosity Bottleneck

The exact information lost in the Jacobi collapse is: for t with J(t) = +1 and gcd(t²−4, N) = 1, whether L_p = L_q = +1 (both QR) or L_p = L_q = −1 (both QNR). Distinguishing these is the quadratic residuosity problem, believed equivalent to factoring under standard complexity assumptions.

### Tested Observable Class

The obstruction has been verified for:
1. Orbital integrals (E7, E7b)
2. BK/nonabelian Fourier transforms at Iwahori level (E6)
3. GL₂ conjugacy class counts (E6 Part B)
4. 2D (trace, det) matrix observables (E6 Part C)
5. Theta functions / quadratic Gauss sums (E7d Part A)
6. Kloosterman sums (E7d Parts B-C)
7. All pointwise Jacobi symbol functions and their nonlinear combinations (E7c: 8 observables)
8. Arithmetic weightings: J*Lambda, J*mu (E7e)
9. Dirichlet character twists: J*chi_m for m in {3,4,5,7,8} (E7e)
10. Multi-discriminant elimination: 12-discriminant families with equal, alternating, SVD, and kurtosis-optimized weights (E7e)
11. Discriminant coherence analysis across families (E7e)
12. Functional equation reflection symmetry (E7e: trivial because J is even)
13. Twisted GL(2) L-function values L(s, Delta x chi_N) at critical and convergent s-values (E8a)
14. Euler factor removal scoring for factor candidate primes (E8a: reduces to gcd)
15. Root number epsilon for L(s, Delta x chi_N) (E8a: determined by N mod 4, no factor signal)
16. L-value profiles for confusable semiprimes of similar size (E8a: differences not systematically exploitable)
17. Multi-form L-function amplification: 19 level-1 eigenforms (weights 12-36) twisted by chi_N (E8b: R^2 ~ 0, no amplification)
18. Critical line vs convergent region sensitivity comparison (E8b: critical line R^2=0.21 > convergent -0.03)
19. Root number independence from N for level-1 quadratic twists: epsilon = (-1)^{k/2} (E8b: algebraic proof)

### What Has NOT Been Tested

1. **L-function zero distribution** — E8a tested L-values at sampled s-points but NOT the detailed zero pattern. Resolving zeros to the precision needed for factor extraction requires O(N²) computation.
2. **Spectral isolation via the trace formula** — projecting onto individual automorphic representations (known to require O(N²) in general)
3. **Non-abelian L-functions** — L-functions of higher-rank groups (GL(3), etc.) where the Langlands correspondence could provide qualitatively different coupling between primes
4. **Multi-moment inverse problems** — using families of L-values across multiple forms/twists simultaneously to constrain factor locations through joint analytic structure

---

## 4. Quantitative Summary Table

| Experiment | Key Observable | Peak/Bulk | E(gcd>1) | Top-1 | Cost | Status |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| E7 | O(t,N) (oracle) | N^{0.25} | 0.667 | 100% | O(N log N) | Oracle-only |
| E7b | Spectral sparsity | partR ~ N^{0.507} | 0.667 exact | 100% | O(N log N) | Oracle-only |
| E7c | J(t), T_±, etc. | ~1 (flat) | ~0 | 0% | O(N log N) | No signal |
| E7c | gcd>1 indicator | N^{0.25} | ~0.99 | 100% | ≡ trial div. | Trivial access |
| E6 | BK transform | = E7 | = E7 | = E7 | = E7 | Collapses to E7 |
| E6 | GL₂ count | ~1 | → 0 | 20% | O(N) | Signal dilutes |
| E7d | Theta function | = E7 (R=0.999) | = E7 | = E7 | = E7 | = E7 |
| E7d | Kloosterman |  anomaly real | — | — | O(N^{3/2}) | Too expensive |
| E7e | J*Lambda, J*mu | ~1 (flat) | ~0.016 | 0-7% | O(N log N) | No signal |
| E7e | J*chi_m twists | ~1 (flat) | ~0.015 | 0-7% | O(N log N) | No signal |
| E7e | Multi-disc (12 d's) | N^{-0.25} | ~0.0002 | 0% | O(N log N) | Anti-peaks |
| E7e | SVD/max-kurt optim | N^{-0.25} | ~0.0002 | 0% | O(N log N) | Anti-peaks |
| E8a | L(s,Δ⊗χ_N) profiles | varies | — | — | O(√N per s) | = gcd test |
| E8a | Euler factor scoring | factors #1,#2 | — | 100% | O(√N) | = trial division |
| E8a | Root number epsilon | +1 for all | — | — | O(√N) | No factor signal |
| E8a | Confusable L-values | diff 0.28-0.37 | — | — | O(√N per s) | Not exploitable |
| E8b | 19-form amplification | R^2 ~ 0 | — | — | O(K√N) | No signal |
| E8b | Critical vs right | R^2=0.21 vs -0.03 | — | — | — | Critical more sensitive |
| E5 | Dimension formula | — | — | — | O(N^{1.91}) | Killed |

---

## 5. Key Analytical Insights

### The 2/3 Energy Split (Exact)

For balanced semiprimes (p ≈ q), exactly 2/3 of the spectral energy of O(t,N) sits in gcd>1 modes:

    E(gcd>1) / E(total) = (p−1 + q−1) / (N−1) → 2/3

This is verified to 6+ decimal places across all tested semiprimes (E7b data shows values converging to 0.6667 from above).

### DFT Convention

All experiments use the +2πi convention:

    f̂(ξ) = (1/N) Σ_t f(t) e^{+2πi tξ/N}

Implemented as `np.conj(np.fft.fft(sig)) / N` (conjugating numpy's −2πi convention).

### The Sum vs Product Distinction

The gcd indicator G(t) = 1_{gcd(t²−4,N)>1} has CRT decomposition u(a)+v(b)−u(a)v(b) (inclusion-exclusion of "p divides" and "q divides"). This is a *sum* of CRT terms, not a pure product, which is why it has real peaks (N^{0.25} scaling, R² = 0.9999). But accessing this requires computing gcd(t²−4, N), which is trial-division-equivalent.

---

## 6. Status Qualifications (Per User Feedback)

Several initially strong claims were refined through iterative discussion:

1. **"CRT product is fundamental obstruction"** — Working hypothesis, not theorem. No formal proof rules out that a cleverly chosen global operation could bypass it.

2. **"Equivalent to factoring"** — Requires regime and assumption qualifiers. The QR ≡ factoring reduction holds under standard assumptions but isn't unconditional.

3. **"Perfectly abelian w.r.t. prime factorization"** — Too strong. The tested class of finite-place observables behaves as CRT products, but global constraints could couple primes.

4. **"Every finite-place observable factors"** — Overreach. Formally, the tested *kernels* (Iwahori, spherical Hecke) factor. Other test functions at finite places might not.

5. **"Kloosterman useless"** — Depends on whether faster evaluation strategies exist (e.g., via p-adic methods, average-value formulas, or algebraic-geometric shortcuts).

6. **"Theta is only oracle"** — Theta IS computable as a phase sum Σ_n e^{2πi a n²/N}. But extracting factor information from its values requires distinguishing the (1+L_p)(1+L_q) structure from the collapsed (1+J) version, which circles back to QR.

---

## 7. Remaining Theoretical Corridors

Three categories of approaches have NOT been ruled out:

### A. Pole Subtraction / Analytic Continuation
The trace formula relates spectral data to geometric/arithmetic data. The "beyond endoscopy" program (Langlands, Altug) works by subtracting known poles from L-functions, leaving a residual that is sensitive to whether an automorphic representation exists. A proxy experiment — applying smoothing or subtractive operations to test whether Jacobi-flatness changes — has been proposed but not executed.

### B. Spectral Projection
Projecting onto individual automorphic representations (via Hecke eigenvalues or spectral decomposition) could isolate factor information. Known implementations require O(N²) or worse, but there may be unknown shortcuts.

### C. Global Coherence Constraints
Epsilon factors, root numbers, and L-value special relationships enforce global consistency conditions that couple local data across primes. These constraints are inherently non-factorizable over primes and could theoretically distinguish ++ from −− without knowing individual factors. No efficient computational test of this is known.

---

## 8. File Inventory

### Experiment Scripts
| File | Experiment | Lines |
|------|-----------|-------|
| `E5_martin_dimension/run_E5.sage` | E5 dimension formula | ~200 |
| `E7_altug_beyond_endoscopy/run_E7.sage` | E7 orbital DFT | ~300 |
| `E7_altug_beyond_endoscopy/run_E7b_spectral_sparsity.sage` | E7b sparsity | ~350 |
| `E7_altug_beyond_endoscopy/run_E7c_jacobi_probes.sage` | E7c Jacobi probes | ~500 |
| `E6_luo_ngo_kernel/run_E6.sage` | E6 BK transform | ~400 |
| `E7_altug_beyond_endoscopy/run_E7d_global_separators.sage` | E7d separators | ~319 |
| `E7_altug_beyond_endoscopy/run_E7e_analytic_proxies.sage` | E7e analytic proxies | ~490 |
| `E8_global_projector/run_E8a_Lfunction_tomography.sage` | E8a L-function tomography | ~565 |
| `E4_hirano_dw_invariants/E4_hirano_mod2_dw.ipynb` | E4 DW (not executed) | — |
| `E8_global_projector/run_E8b_multiL.sage` | E8b multi-form amplification | ~600 |
| `E10_integer_carry/run_E10_carry_signals.sage` | E10 carry signals | ~540 |
| `E11_feature_extraction/run_E11_feature_extraction.sage` | E11 ML feature extraction | ~870 |

### Data Files
| File | Contents |
|------|----------|
| `data/E7_cancellation_results.json` | E7 orbital DFT (50 semiprimes to N=5000) |
| `data/E7b_spectral_sparsity_results.json` | E7b sparsity analysis |
| `data/E7c_jacobi_probes_results.json` | E7c 10-observable probe (25 semiprimes to N=497k) |
| `data/E6_bk_transform_results.json` | E6 BK analysis (20 semiprimes to N=47k) |
| `data/E7d_global_separators_results.json` | E7d theta + Kloosterman (20 semiprimes to N=256k) |
| `data/E7e_analytic_proxies_results.json` | E7e proxy tests (15 semiprimes, 13 signals) |
| `data/E8a_Lfunction_tomography_results.json` | E8a L-function values (12 semiprimes, 5 confusable pairs) |
| `data/E10_carry_signals_results.json` | E10 carry signal metrics |
| `data/E11_feature_extraction_results.json` | E11 ML results (111 features x 600 semiprimes) |
| `data/*_plots.png` | Visualization plots for E7, E7b, E7c, E6, E7e, E10, E11 |

### Configuration
| File | Purpose |
|------|---------|
| `.gitignore` | Ignores `*.sage.py` cache files |
| `utils/` | Shared semiprime generation utilities |

---

## 9. Git History

```
df26a30 E8b: Multi-form L-function amplification shows no factor signal
4e67d8f Update SUMMARY.md with E8a L-function tomography results
5d314ce E8a: L-function tomography shows twisted L-values reduce to gcd testing
5c63423 Update SUMMARY.md with E7e analytic proxy results
0ff9c7a E7e: Analytic proxy tests confirm Jacobi-flat data resists all cheap transforms
f789d92 Add comprehensive research summary covering E5-E7d experiments
d15a82c E7d: Global analytic separators (theta, Kloosterman) hit same CRT obstruction
6a6fd6e Add .sage.py cache files to gitignore
dcc676b E6: BK transform reduces to orbital integral, adds nothing to E7
74aa802 E7c: Jacobi-only spectral probes confirm refined no-go taxonomy
d19052c E7b: Confirm spectral sparsity, top-mode factor extraction, Jacobi no-go
8f89c05 E7b: Spectral sparsity analysis confirms 3-tier Fourier structure
15d4679 E7: Run Beyond Endoscopy cancellation analysis (50 semiprimes to N=5000)
f5784eb E5: Fix dimension formula (mu*mu inversion) and add runnable sage script
e0783a7 E4: Add Hirano mod-2 Dijkgraaf-Witten invariants probe
d519a85 Add shared semiprime generation utilities
```

---

## 10. Technical Notes / Gotchas

1. **Sage `.sage.py` caching:** Sage compiles `.sage` files to `.sage.py` and caches them. After modifying function signatures, `rm -f *.sage.py` is required before rerunning, otherwise the old cached version is loaded. Added `*.sage.py` to `.gitignore`.

2. **Semiprime generation:** Using `randint`-based generation with tight constraints can hang. The working approach uses `np.logspace` for target primes and `next_prime(p); q = next_prime(p)` for guaranteed balanced pairs.

3. **Large N performance:** With logarithmic spacing from p=50, 25 semiprimes can reach N=1.8B if max_p is uncapped, causing 100s+ per DFT iteration. Cap max_p ≈ 700 for interactive work (N up to ~500k).

4. **Duplicate semiprimes:** Logarithmic spacing of target primes can map nearby targets to the same prime pair. The E6 results contain duplicate entries for N=143, 323, 899 — this doesn't affect conclusions.

---

## 11. The Dimension Barrier (E9 Analysis)

### The complexity constraint

Any factoring algorithm must run in poly(log N) time. Since dim S_k(Γ_0(N)) = O(N), any computation that enumerates basis elements, builds Hecke matrices, or iterates over eigenforms costs at least O(N) = O(2^n) where n = log₂N. This kills all "work inside the level-N space" approaches, including the proposed E9 experiment (Hecke moment sketches at level N), which was abandoned before completion on complexity grounds.

### The succinct representation question

The Eichler-Selberg trace formula provides a "succinct" representation of Tr(T_ℓ | S_k(Γ_0(N))): the formula has O(√ℓ) terms involving class numbers and local embedding numbers. But evaluating the local factors at p|N requires:

    ∏_{p|N} (1 + χ_D(p)) = 1 + (χ_D(p) + χ_D(q)) + χ_D(N)

The Jacobi symbol χ_D(N) is poly(log N)-computable. The "missing piece" χ_D(p) + χ_D(q) = r_D(N) - 1 - χ_D(N), where r_D(N) counts representations of N by the principal quadratic form of discriminant D.

**Therefore:** A poly(log N) algorithm for r_D(N) would break the missing-bit barrier. No such algorithm is known; all methods require O(√N).

### What remains

See `ACCESS_MODEL_REQUIREMENTS.md` for the full access-model audit, formal no-go template, and open questions.

---

## 12. Current Status and Remaining Corridors

The experimental program has systematically closed all "accessible" corridors:

- **E7-E7e:** Finite-place Jacobi observables collapse to CRT products (structural proof + experiments)
- **E8a-E8b:** Twisted GL(2) L-functions carry ~0 extractable bits (χ(p)=0 structural argument + experiments)
- **E9 (abandoned):** Level-N space computations are exponential in input size (dimension barrier)

**Remaining corridors (speculative, all require new poly(log N) primitives):**

1. **Succinct spectral projectors:** Evaluate trace formula terms without enumerating divisors of N. No known method.
2. **Analytic continuation shortcuts:** Evaluate automorphic kernels (BK/Ngô) at specific points in poly(log N). No known fast evaluation.
3. **p-adic methods:** p-adic L-functions or modular forms with special-value computations. Unexplored.
4. **Arithmetic topology / QFT invariants:** Kim's arithmetic Chern-Simons, partition functions. No algorithmic content.

The fundamental bottleneck across all remaining corridors: computing r_D(N) — or an equivalent quantity that resolves χ_D(p) + χ_D(q) from N alone — in poly(log N) time.

---

## 13. E10: Integer-Carry Signals (In Progress)

### Motivation

All previous experiments tested functions on Z/NZ that decompose through CRT as products or low-rank sums of local components. The DFT of such functions is spectrally flat (peak ~ N^{-0.25}).

However, the INTEGER representation t ∈ [0, N) introduces non-separable coupling via "carry bits." The CRT reconstruction t = a·q·q̃ + b·p·p̃ (mod N) involves a carry c = ⌊(a·q·q̃ + b·p·p̃)/N⌋ ∈ {0,1}. Functions that depend on this carry (e.g., ⌊t²/N⌋) are NOT rank-1 CRT-separable, even though they are computable in poly(log N) time.

### Signals tested

| Signal | Definition | CRT-separable? | Carry-based? |
|--------|-----------|:-:|:-:|
| f_jacobi (CONTROL) | J(t²-4, N) | Yes (rank 1) | No |
| f_carry_jacobi | J(⌊t²/N⌋, N) | No | Yes |
| f_carry_parity | (-1)^⌊t²/N⌋ | No | Yes |
| f_mixed | J(t,N) · (-1)^⌊2t/√N⌋ | No | Yes |
| f_carry_sum | Σ_{k=1}^{10} (-1)^⌊kt²/N⌋ | No | Yes |
| f_lattice | J(t² mod ⌊√N⌋, N) | No | Yes |

### Key question

Do carry-based signals break spectral flatness? If so, integer-carry operations provide a new primitive beyond the Jacobi barrier. If not, the barrier extends to the full integer-arithmetic oracle model.

### Results (quick probe: N ~ 500 to 12000, ~90 semiprimes)

| Signal | Peak scaling α | Factor excess | CRT rank (90%) | Verdict |
|--------|:-:|:-:|:-:|---------|
| jacobi_control | N^{-0.356} | 0.539 | 1.0 | Flat (E7c confirmed) |
| carry_jacobi | N^{-0.355} | 0.980 | 10.3 | Flat — same as control |
| carry_parity | N^{-0.447} | 0.987 | 9.0 | WORSE than control |
| mixed_jacobi_carry | N^{-0.413} | 0.957 | 8.2 | Worse than control |
| carry_sum | N^{-0.473} | 1.036 | 9.1 | Worst decay of all |
| lattice_jacobi | N^{-0.014} | 0.961 | 9.4 | Numerical artifact for N=2p |

**Key observations:**
1. **CRT rank increases:** Carry operations raise rank from 1 (Jacobi) to ~10, scaling as N^{0.3}. This confirms that integer floor/mod DOES break rank-1 CRT separability.
2. **Spectra remain flat:** Despite higher rank, all carry signals have decaying peaks (α ≤ -0.35). No factor-localized energy.
3. **Factor energy excess ≈ 1.0:** Carry signals have exactly the expected factor energy for a random function — no elevation above baseline.
4. **Conclusion:** Integer-carry operations increase CRT rank but do NOT align the excess rank with factor frequencies. The barrier extends to the full Ring+Jacobi+Integer oracle model.

---

## 14. Barrier Theorem Formalization

See `BARRIER_THEOREM.md` for the formal treatment:

1. **Oracle model (RJI):** Ring(N) + Jacobi + Integer arithmetic (floor, comparison, bits)
2. **CRT factorization lemma:** Rank-r functions have DFT peak ≤ r/√N
3. **Ring+Jacobi circuits:** Depth-d circuits produce rank ≤ 3^d functions
4. **The carry question:** Do integer operations increase rank beyond poly(log N)?
5. **Connection to QRP:** Spectral flatness → S_D(N) hardness → factoring hardness

See `ACCESS_MODEL_REQUIREMENTS.md` for the full access-model audit and hinge scalar catalog.

---

## 15. E11: Comprehensive Poly(log N) Feature Extraction

### Motivation

E10 closed carry-based signals individually. E11 asks: can ANY combination of ~111 poly(log N)-computable features predict the hinge scalar S_D(N) = chi_D(p) + chi_D(q)?

This is the "universal test" — ridge regression with LOOCV searches the full linear span of all features for predictive combinations. Permutation tests provide statistical significance.

### Feature groups (111 features total)

| Group | Count | Examples |
|-------|:-----:|---------|
| Jacobi symbols | 20 | J(k, N) for first 20 primes |
| Modular exponentiations | 25 | g^N mod N, g^{(N-1)/2} mod N, J(g^N, N) |
| Euler residuals | 5 | g^{N-2*isqrt(N)+1} mod N (encodes p+q error) |
| Integer carry | 15 | floor(t^2/N), carry parity, J of carry |
| CF convergents | 20 | Partial quotients, CF remainders, J of remainders |
| Pollard p-1 | 6 | g^{20!} mod N indicators + values |
| Mixed interactions | 10 | Cross-group products (J x modexp, etc.) |
| N-arithmetic controls | 5 | N mod k for small k |
| Random controls | 5 | Null hypothesis baseline |

### Key design choice: balanced semiprimes only

Only semiprimes with p/q >= 0.3 were generated, ensuring trial-division-equivalent features cannot succeed. This is the regime relevant to the barrier question (RSA-type semiprimes).

### Results (600 semiprimes, 16-22 bit, balanced)

**Hinge scalars (the barrier test):**

| Target | R^2_CV | p-value | Verdict |
|--------|:------:|:-------:|---------|
| S_{-3}(N) | 0.013 | 0.020 | Flat |
| S_{-4}(N) | 0.002 | 0.130 | Flat |
| S_5(N) | 0.025 | 0.000 | Flat (inconsistent across bit sizes) |
| S_{-7}(N) | -0.008 | 0.995 | Flat |
| S_8(N) | 0.003 | 0.140 | Flat |

**Non-hinge targets (expected predictability):**

| Target | R^2_CV | Top features |
|--------|:------:|-------------|
| p/sqrt(N) | 0.353 | carry_hw, cf_pq_0, cf_rem (balance detection) |
| log(q/p) | 0.350 | Same as above |

**Critical diagnostics:**
- Random control mean |r| = 0.035; Real feature mean |r| = 0.034 (identical — no feature group outperforms random)
- High ANOVA F for J_cf_rem on S_{-4} (F=44577) is the known Jacobi mechanism: J(|r_k|, N) = J(-1, N) when r_k < 0, which detects S_{-4}=0 vs S_{-4}!=0 but CANNOT distinguish +2 from -2 (the QRP)
- Per-bit-size R^2 for S_5: 16-bit=-0.003, 18-bit=0.108, 20-bit=0.074, 22-bit=0.002 — inconsistent scaling confirms noise

### Conclusion

No combination of 111 poly(log N)-computable features achieves meaningful prediction of any hinge scalar. The barrier extends to the full feature space including modular exponentiations, CF convergents, and mixed interactions. All apparent "signals" reduce to known mechanisms (Jacobi S_D=0/!=0 discrimination, CFRAC-equivalent CF remainder gcd hits, Pollard p-1 smoothness).

---

## 16. Literature Survey (2023-2026)

A systematic search of recent literature found no new poly(log N) classical primitive:

| Direction | Best known | New results (2023-2026) |
|-----------|-----------|------------------------|
| Classical factoring | GNFS at L(1/3) | Schnorr lattice tested but unproven |
| Dequantize Shor | Quantum only | Regev (2023) still quantum; dequantization doesn't reach period-finding |
| Succinct Hecke traces | O(N) via modular symbols | All methods require factoring N |
| r_D(N) in poly time | O(sqrt(N)) | No progress |
| Langlands -> factoring | No algorithmic content | Sakellaridis (2023) pure theory |
| Class groups | L(1/2) subexponential | Incremental only |

Notable: The Jacobi Factoring Circuit (STOC 2025) factors N=P^2*Q using Jacobi in quantum superposition. For semiprimes PQ, J(a,PQ) is rank-1 CRT — confirming our barrier is overcome by quantum parallelism, not classical tricks.

---

## 17. Langlands Ecosystem Survey: Exhaustive Poly(log N) Audit

A comprehensive web search across the full Langlands program ecosystem found **no new poly(log N) primitive**. Every tool hits one of three obstructions:

### Obstruction 1: Local Langlands Decomposition (CRT at finite places)

All functorial constructions decompose through local Langlands at each prime. Computations at p|N require knowing p.

| Method | Status | Why it fails |
|--------|--------|-------------|
| Base change GL(2) (Langlands, Arthur-Clozel) | Closed | Splitting of p,q in extension K requires factoring |
| Symmetric powers (Newton-Thorne 2021-2026) | Closed | Bad Euler factors at p\|N require knowing p, q |
| Rankin-Selberg convolutions | Closed | Level-1 case = E8b (R^2~0); level-N = Obstruction 2 |
| Endoscopic transfer (Arthur classification) | Closed | Local endoscopy at p\|N needs the prime |
| Theta correspondence / Weil representation | Closed | Factors through CRT on Z/NZ (E7d: R=0.999) |
| Braverman-Kazhdan generalized Fourier transform | Closed | Local BK = Euler product (E6 proved); no algorithmic content |

### Obstruction 2: Dimension Barrier (dim O(N) = exponential)

| Method | Status | Bottleneck |
|--------|--------|-----------|
| Waldspurger formula (central L-values) | Closed | c(N) lives in level-4N space, dim O(N) |
| Arakelov theory on X_0(N) | Closed | Heights/Green's functions need automorphic forms at level N |
| Spectral projection (individual eigenforms) | Closed | O(N^2) Hecke matrices |
| Edixhoven-Couveignes at level N | Closed | Poly(log p) for FIXED level; level N gives dim O(N) |

### Obstruction 3: O(sqrt(N)) — not poly(log N)

| Method | Complexity | Bottleneck |
|--------|-----------|-----------|
| Hilbert class polynomial H_D | O(\|D\|^{1+eps}) | Root-finding mod composites = factoring |
| Heegner points at level N | O(N^{1/2}) | Heegner hypothesis needs chi_D(p) per prime |
| Class field of Q(sqrt(-N)) | O(N^{1/2+eps}) | Class number h(-N) |
| r_D(N) representation numbers | O(sqrt(N)) | THE hinge scalar bottleneck |
| Harvey/Kedlaya point counting | Poly(log p) over F_p | Requires knowing the prime p |
| Lauder's p-adic Rankin L-functions | Poly in p-adic precision | Requires knowing which prime p to localize at |

### Additional areas checked (all negative)

| Area | Assessment |
|------|-----------|
| Etale cohomology of Spec(Z/NZ) | Decomposes via CRT: H^i(Z/NZ) = H^i(Z/pZ) x H^i(Z/qZ) |
| K-theory of Z/NZ | K_1 = (Z/NZ)* encodes phi(N), but computing it requires phi(N) |
| Brauer-Manin obstruction | CRT decomposition at finite places |
| Isogenies over Z/NZ (ECM) | Sub-exponential, not poly(log N) |
| Castryck-Decru SIDH attack (2022) | Poly-time but requires auxiliary torsion data with no factoring analogue |
| Polynomial splitting mod N (Berlekamp) | Known technique; probability -> 0 for generic semiprimes |

### The structural argument

The Langlands program is built on the **local-global principle**: pi = tensor product of pi_v over all places v. Every functorial construction preserves this tensor product structure. Any computation that "sees" the bad primes p|N requires local data at those primes, which requires knowing them. This is not a limitation of current algorithms — it is structural to the theory.

### The one genuinely poly(log N) tool

**Edixhoven-Couveignes (Princeton 2011)**: Computes a_p(f) for fixed-level forms at a prime p in poly(log p, k) via etale cohomology. For our problem: a_N(f) = a_p(f) * a_q(f) (multiplicativity) — requires factoring. And extending to level N faces dim O(N).

---

## 18. Final Status

### Closed corridors (with evidence type)

| Corridor | Closed by | Evidence |
|----------|-----------|----------|
| Jacobi observables | E7c | Structural proof + experiments |
| Analytic proxies (13 types) | E7e | Anti-peaks N^{-0.25} |
| Twisted GL(2) L-functions | E8a-b | chi(p)=0 + multi-form R^2=0 |
| Level-N computations | E9 | Dimension barrier O(N) |
| Integer-carry signals | E10 | Flat spectra despite rank increase |
| 111-feature ML sweep | E11 | All R^2_CV <= 0.025, random=real |
| Literature (6 directions) | Survey | No new primitives 2023-2026 |
| Langlands ecosystem (20+ tools) | Web survey | Three universal obstructions |

### The barrier in one sentence

Every poly(log N)-computable observable on Z/NZ that we can construct or find in the Langlands program literature has spectrally flat DFT at factor frequencies, consistent with (and partially implied by) the hardness of the Quadratic Residuosity Problem.
