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

### What Has NOT Been Tested

1. **Genuinely global analytic objects** — L-function poles/residues, spectral projectors onto automorphic representations, functional equations of non-abelian L-functions. These cannot be implemented as cheap transforms on Jacobi sequences.
2. **Spectral isolation via the trace formula** — projecting onto individual automorphic representations (known to require O(N²) in general)
3. **Global coherence constraints** — epsilon factors, root numbers, L-value special relationships that couple the local data across primes in ways that individual Euler factors don't

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
| `E4_hirano_dw_invariants/E4_hirano_mod2_dw.ipynb` | E4 DW (not executed) | — |

### Data Files
| File | Contents |
|------|----------|
| `data/E7_cancellation_results.json` | E7 orbital DFT (50 semiprimes to N=5000) |
| `data/E7b_spectral_sparsity_results.json` | E7b sparsity analysis |
| `data/E7c_jacobi_probes_results.json` | E7c 10-observable probe (25 semiprimes to N=497k) |
| `data/E6_bk_transform_results.json` | E6 BK analysis (20 semiprimes to N=47k) |
| `data/E7d_global_separators_results.json` | E7d theta + Kloosterman (20 semiprimes to N=256k) |
| `data/*_plots.png` | Visualization plots for E7, E7b, E7c, E6 |

### Configuration
| File | Purpose |
|------|---------|
| `.gitignore` | Ignores `*.sage.py` cache files |
| `utils/` | Shared semiprime generation utilities |

---

## 9. Git History

```
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

## 11. Proposed Next Steps

The experimental program has reached a natural decision point. Three directions have been proposed:

1. **Research decision memo:** Formally state the tested obstruction class and remaining corridors as a concise document for scoping the next research cycle.

2. **Pole-subtraction proxy experiment:** Apply smoothing/subtractive operations imitating beyond-endoscopy residue removal to test whether Jacobi-flatness changes under such operations.

3. **Global analytic separator toy problems:** Design 2-3 concrete Langlands-native test problems where a global operation (spectral projection, epsilon factor computation) could theoretically distinguish ++ from −− configurations.
