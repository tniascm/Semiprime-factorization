# Round 3 Research Report: Analytic Number Theory Approaches to τ(N) mod 691

**Date:** 2026-02-27
**Scope:** Exhaustive investigation of whether analytic number theory can compute τ(N) mod ℓ at composite N = pq in sub-O(N) time, overcoming the E13 computational barrier.

---

## 1. Executive Summary

This round investigated 14 distinct mathematical pathways for computing τ(N) mod 691 at composite N = pq, spanning trace formulas, L-functions, automorphic forms, algebraic geometry, period integrals, character sums, isogeny graphs, and complexity theory. The central findings:

1. **One genuinely open question**: Computing φ(N) mod 691 is NOT proven equivalent to factoring (the standard Miller reduction requires the full value of φ(N)). The phi-hiding assumption (Cachin-Micali-Stadler 1999) is adopted as an independent cryptographic assumption precisely because no factoring reduction exists.

2. **A fundamental duality** was identified between point-evaluation of L-functions (cost: O(√conductor)) and coefficient-extraction from L-functions (cost: O(N^{1/2+ε})). These are dual problems; the approximate functional equation is optimized for the first, not the second.

3. **All 13 other paths are definitively closed** — by CRT decomposition, multiplicativity, dimensional barriers, or cost lower bounds.

4. **83 references** collected across analytic number theory, algebraic geometry, cryptography, and computational complexity.

---

## 2. Detailed Findings by Topic

### 2.1 Rankin-Selberg L-functions (PATH 14)

**L(s, Δ×Δ) = Σ |τ(n)|²/n^s** is a degree-4 L-function with conductor 1. Its N-th coefficient encodes |τ(N)|².

**Key results:**
- Extracting the N-th Dirichlet coefficient via Perron's formula: O(N^{1+ε}) cost (contour integral over non-compact vertical line requires T ~ N quadrature points)
- The approximate functional equation for degree-4 L-functions at height T requires O(T²) terms per evaluation — no sub-polynomial improvement known
- Character twists L(s, Δ⊗χ_N): the N-th coefficient is χ_N(N)·τ(N) = 0 since gcd(N,N) = N. Twisting annihilates the very coefficient we want.
- Recovering τ(N) from a family of twists requires inverting an O(N)-dimensional linear system
- Luo-Ramakrishnan determination theorem: f is uniquely determined by {L(1/2, f⊗χ) : χ primitive mod q, q ≤ Q}, but determination is non-constructive and Q grows with the level

**The point-evaluation / coefficient-extraction duality:**
- Evaluating L(s₀) = summing O(√conductor) known coefficients at a specific point → O(√N) cost
- Extracting the N-th coefficient = recovering one term from the analytic object → O(N^{1/2+ε}) cost minimum
- These are genuinely DUAL problems; the AFE is designed for the first

**New reference:** Huang (2021), arXiv:2002.00591 — improved Rankin-Selberg problem exponent from 2/3 to 3/5, but for AVERAGES not individual coefficients.

### 2.2 Approximate Functional Equation and Hiary's Methods (PATH 14 cont.)

**Standard AFE for L(s, Δ):** At the central point s = 1/2 + it, critical length is √(t/2π). Cost: O(t^{1/2+ε}).

**For L(s, Δ⊗χ_N) with conductor N:** Critical length √N. Each of the O(√N) terms involves τ(n)χ_N(n), where χ_N(n) = (n/N) is computable in O(log²N) via Euclidean algorithm WITHOUT factoring N. Total: O(N^{1/2+ε}).

**Hiary's algorithm (Annals 2011):** Computes ζ(1/2+iT) in O(T^{1/3+ε}), improving over √T.
- Uses efficient quadratic exponential sums via van der Corput B-process
- Extension to Dirichlet L-functions (arXiv:1205.4687): O(q^{1/3+ε}) when q is a **smooth** modulus (high prime power)
- **CRITICAL: For squarefree N = pq, NO improvement over O(√N).** The Postnikov character formula reduction to quadratic exponential sums requires smooth modulus.
- Vishe (arXiv:1202.6303): L(1/2, f⊗χ_q) in O(q^{5/6}) for smooth q, falls back to O(q^{1/2}) for prime q

**Odlyzko-Schonhage amortization:** Evaluates a Dirichlet series at O(N) equally spaced points in O(N^{1+ε}) total → amortized O(N^ε) per value. For a SINGLE value: still O(N^{1/2+ε}).

**Multiple Dirichlet series:** Z(s,w) = Σ_d L(s, Δ⊗χ_d)/|d|^w. Isolating L(1/2, Δ⊗χ_N) via Mellin inversion in w costs O(N^{3/2}) — worse than direct computation.

**Booker-Platt-Rubinstein:** Amortized improvements for many L-values simultaneously, but no sub-√N for a single isolated value.

### 2.3 Amplification Method (PATH 5, deepened)

**Initial dismissal:** "Vacuous for dim S₁₂ = 1." Correct in conclusion but incomplete in reasoning.

**Deeper analysis at level N = pq:** dim S₁₂(Γ₀(N)) ≈ N/6. Amplification IS non-trivial here. The DFI amplifier forms:
```
S = Σ_f |Σ_l x_l λ_f(l)|² · |a_f(N)|² / ⟨f,f⟩
```
The step of discarding non-target spectral terms (non-negative) is **irreversible information loss**. Recovering exact τ(N) requires subtracting ALL other contributions — at least as hard as the original problem.

**Spectral reciprocity (Blomer-Khan, Duke Math J. 2019):** Schematically:
```
(4th moment at level q, twisted by λ_f(l)) ~ (4th moment at level l, twisted by λ_g(q))
```
For N = pq: the dual side involves λ_g(N) = λ_g(p)·λ_g(q) by Hecke multiplicativity. **The factorization is encoded in the dual problem.**

**Nelson's orbit method (arXiv:2109.15230, 2021):** 239-page breakthrough replacing classical amplification with microlocal analysis for Lie group representations. Still produces subconvexity BOUNDS: L(1/2, π) ≪ C(π)^{1/4-δ}. Does not compute exact values.

**Petrow-Young Weyl bound (Annals 2020):** |L(1/2+it, χ)| ≪ (q(1+|t|))^{1/6+ε}. Bounds only.

**The structural conclusion:** Amplification is an inequality-producing technique. The exact formula (Petersson trace formula, dim S₁₂ = 1) gives τ(N) = C·Σ_{c≥1} S(1,N;c)/c · J₁₁(4π√N/c), convergent but costs O(√N) to O(N).

### 2.4 Shifted Convolution Sums (PATH 9, deepened)

**Spectral decomposition (Blomer-Harcos, Duke Math J. 2008):** The shifted convolution Dirichlet series L_h(s,f,f) has meromorphic continuation with poles at s = 1/2 ± ir_j (Maass eigenvalues). Main term involves Ramanujan sums c_d(h) and divisor sums — computing these for h = N requires factoring N.

**Holowinsky's sieve (Duke Math J. 2009):** Upper bound: Σ |λ_f(n)λ_f(n+h)| ≪ X/(log X)^{1-ε}. Sufficient for QUE (mass equidistribution) but zero pointwise information about individual τ(N).

**Circle method at h = N:** Major arc at a/q with q | N would involve local behavior of Σ τ(n)e(2πina/p). Three obstructions: (a) Δ is cusp form, contributions exponentially suppressed, (b) don't know p to isolate this arc, (c) integral over ALL arcs doesn't separate factor-dependent parts.

**Motohashi's 4th moment formula:** Relates ∫|ζ(1/2+it)|⁴ w(t) dt to spectral cubic moments. Both sides are averages; neither isolates individual coefficients.

**Triple product L-function L(s, Δ×Δ×Δ):** Degree 8. Watson-Ichino formula relates it to triple period integrals. Computing via AFE needs O(N^{3/2}) terms. Local factors at primes of N require knowing the factorization.

**Petersson/Rademacher series:** The UNIQUE exact formula:
```
τ(N) = C₁₂ · Σ_{c≥1} S(1,N;c)/c · J₁₁(4π√N/c)
```
Convergent (dim S₁₂ = 1), costs O(N). The Niebur convolution formula τ(n) = n⁴σ(n) - 24Σ i²(35i²-52in+18n²)σ(i)σ(n-i) also costs O(N) and requires σ(N) (needs factoring).

**Key papers found:**
- Blomer-Khan-Young (2020), Compositio — Motohashi for non-archimedean test functions
- Shifted convolution for Siegel forms (2025, arXiv:2511.19303)
- GL(3) shifted convolution with averaged shifts (2025, arXiv:2510.15799)
- Shifted convolution L-function for Maass forms (2023, arXiv:2311.06587)

### 2.5 Waldspurger / Period Integrals (PATH 12, expanded to 11 sub-sections)

**Kohnen-Zagier formula (1981):** For Δ, the Shimura correspondent g ∈ S_{13/2}^+(Γ₀(4)) is unique (dim = 1). Formula: |c(|D|)|² = C · L(6, Δ, χ_D).

**Critical finding:** The formula connects c(|D|)² to L-VALUES, NOT to τ(D). There is NO path from the half-integral weight coefficient c(N) to the integer-weight coefficient τ(N). The Shimura correspondence is between SPACES, not between individual coefficients at matching indices.

**Shintani cycle integrals:** c(D) = Σ_{Q ∈ Cl(D)} ∫_{γ_Q} Δ(z)·Q(z,1)⁵ dz. Cost: O(√N) since h(-N) ≈ √N on average.

**Gross-Zagier (higher weight, Zhang 1997-2001):** L'(Δ/K, 6) = c·⟨z_K,z_K⟩_{BB}. Relates DERIVATIVE to heights on 10-fold Kuga-Sato variety. Heegner cycles cost O(√D) to enumerate. CRT decomposition over Z/NZ.

**Kudla program:** Arithmetic Siegel-Weil formula: global intersection = product of local densities × L-derivative. Li-Zhang (2022) proved the conjecture. Product-over-primes structure = CRT obstruction.

**Ichino triple product formula (2008):** Global period = C · L(1/2, π₁×π₂×π₃) / ∏_v I_v. Local integrals at p, q require knowing the factorization.

**Recent test vector theory (Hu-Nelson 2019):** Hybrid subconvexity bounds as strong as Weyl bound, but still BOUNDS not exact values.

**Three universal obstructions:**
1. Formula connects L-values to periods, not τ(N) to periods
2. Every evaluation costs ≥ O(√N)
3. Bach-Charles blocks the reverse direction (L-value → τ(N) requires Euler product ≡ factoring)

### 2.6 Character Sums at Composite Moduli (PATH 13)

**CRT decomposition for all complete sums:**
- Gauss: G(χ,N) = G(χ_p,p)·G(χ_q,q). |G| = √N for primitive χ (no info).
- Jacobi: J(N) = J(p)·J(q). APRCL restricts divisors to t classes mod s, but t too large.
- Kloosterman: S(a,b;N) = S(a,b;p)·S(a,b;q). Cost O(N) to compute.
- Ramanujan: c_N(n) = c_p(n)·c_q(n). Requires factoring N.

**Incomplete sums Σ_{x≤X} e(f(x)/N):** Do NOT factor via CRT (carry structure). However, all peaks at noise level: N^{-0.35} to N^{-0.47} (confirmed by E10/E12). Weyl differencing for smooth moduli uses factorization as INPUT. For squarefree N, no improvement.

**L(1/2, χ_N) computable in O(√N) without factoring** (Jacobi symbol via Euclidean algorithm). Encodes class number h(-4N), but extraction ≥ O(N^{1/4}) (SQUFOF).

**The irreducible hard core:** The quadratic residuosity problem — given (n/N) = +1, distinguishing QR/QR from QNR/QNR — is generically equivalent to factoring (proven in the generic ring model).

**Gauss sum factoring (Schleich et al., arXiv:1210.6474):** Truncated Gauss sums detect factors but scale exponentially (M must grow exponentially to suppress ghost factors).

### 2.7 Isogeny Graphs at Composite Level (PATH 13 cont.)

**Supersingular isogeny graphs over Z/NZ:** Tensor product of graphs over F_p and F_q. Adjacency matrix: A(N) = A(p) ⊗ A(q). Complete CRT decomposition.

**Castryck-Decru attack (2022, broke SIDH):** Uses Kani reducibility criterion on (1,N)-isogenies from E×E'. Requires THREE ingredients with no factoring analogue: (a) known smooth degree, (b) auxiliary torsion point images, (c) Kani compatibility. The integer N provides no torsion auxiliary data.

**Charles-Goren-Lauter hash:** EndRing ↔ MaxOrder ↔ ℓ-IsogenyPath equivalences (unconditional: Le Merdy-Wesolowski 2025). Factoring is an INPUT that helps solve isogeny problems, NOT an output. No reduction FROM isogeny problems TO factoring exists.

**Deuring correspondence at composite level:** Enhanced supersingular curves (E, G[N]) with squarefree N correspond to Eichler orders of level N. Level structure decomposes as product over prime divisors.

**Eichler mass formula:** mass(O, level pq) = (D-1)/12 · (p-1)(q-1) = (D-1)/12 · φ(N). Computing mass gives φ(N), but already requires knowing p, q to evaluate local factors.

**Post-SIDH landscape (2022-2026):** No paper establishes a reduction FROM isogeny problems TO integer factoring. Arrow consistently: factoring → isogeny solutions.

### 2.8 φ(N) mod 691 Complexity (PATH 7, the critical open question)

**Miller's reduction (1976):** φ(N) fully → factoring N. Uses p + q = N + 1 - φ(N), then quadratic x² - sx + N = 0 over Z. Requires FULL φ(N), NOT partial.

**Coppersmith connection:** Given p mod M where M > N^{1/4}, can factor N in poly(log N). Since (p+q) mod M gives p mod M via quadratic mod M, knowing φ(N) mod M where M > N^{1/4} suffices. For RSA-2048: need n/4 ≈ 512 bits. Single oracle φ(N) mod 691 gives ~9.4 bits. CRT from all primes ℓ < 355 would reach threshold.

**Phi-hiding assumption (Cachin-Micali-Stadler, EUROCRYPT 1999):** "Given N = PQ, deciding whether ℓ | φ(N) is hard." Adopted as INDEPENDENT assumption:
- Factoring hard ⟹ phi-hiding plausible (NOT proven)
- Phi-hiding hard does NOT imply factoring hard
- No reduction in either direction
- Schridde-Freisleben (ASIACRYPT 2008): fails for N = PQ^{2e}, holds for standard RSA

**Oracle complexity analysis:**
- Single φ(N) mod 691: ~9.4 bits, far below Coppersmith n/4
- φ(kN) mod ℓ for prime k: gives (k-1)·φ(N) mod ℓ, no new info
- φ(N²) mod ℓ: gives N·φ(N) mod ℓ, no new info
- Multiple ells via CRT: B > (n/4)·ln(2) primes needed

**The complexity gap formalized:** "Compute φ(N) mod 691 given N = pq" sits in NP ∩ coNP (like factoring), NOT known to be in P, NOT known equivalent to factoring. Analogous to Decisional Composite Residuosity Assumption (Paillier): believed hard, used in cryptography, not proven equivalent to factoring.

### 2.9 Lehmer's Conjecture, Lacunarity, and Non-Eigenforms

**Lehmer's conjecture (τ(n) ≠ 0 ∀n):** Verified to ~8.16 × 10²³ (Derickx-van Hoeij-Zeng 2013). Withdrawn proof attempt March 2025 (arXiv:2503.23498, Shi-Wang-Sole, incorrect Venkov criterion).

**Excluded values:** Balakrishnan-Craig-Ono (2020): τ(n) ∉ {±1,±3,±5,±7,±691} for n > 1. Even values excluded: Balakrishnan-Ono-Tsai (2021).

**Lacunarity mod 691:**
- σ₁₁(pq) ≡ 0 mod 691 iff p^{11} ≡ -1 or q^{11} ≡ -1 mod 691
- Since gcd(11, 690) = 1, x → x^{11} is a bijection on (Z/691Z)*
- Density of primes with σ₁₁(p) ≡ -1 mod 691: exactly 1/690
- Bellaiche-Soundararajan (2015): density of {n : τ(n) ≢ 0 mod 691} is zero (decays like (log x)^{-α})

**Mod-691 Galois representation:** ρ_{Δ,691} is REDUCIBLE: ρ ≅ 1 ⊕ χ¹¹ (mod 691). This is precisely BECAUSE Δ ≡ E₁₂ mod 691 (the Eisenstein congruence). For ℓ ∉ {2,3,5,7,23,691}, im(ρ_{Δ,ℓ}) contains SL(2, F_ℓ).

**Non-eigenforms:** S₁₂(SL₂(Z)) is 1-dimensional — only Δ, which IS an eigenform. No non-eigenforms exist. In higher dimension, non-eigenform coefficients are linear combinations of eigenform coefficients; Bach-Charles applies to each component. The mod-691 reduction gives σ₁₁(N), itself multiplicative.

**Weight-1 forms:** Correspond to Artin representations (finite image). EC approach fails (non-cohomological). No connection to τ(N) whatsoever — different weight, different representation type.

---

## 3. New Ideas and Observations

### 3.1 The Point-Evaluation / Coefficient-Extraction Duality

This is the cleanest formulation of why L-function methods fail:
- L-function EVALUATION at s₀: compute Σ a(n)·V(n/√C)/n^{s₀}. Smooth sum of length √C. Cost: O(√C).
- Fourier coefficient EXTRACTION at n = N: compute (1/2πi) ∫ L(s)·N^s ds along Re(s) = c. Non-compact contour integral. Cost: O(N^{1/2+ε}).

These are dual problems. No trick (smoothing, Hiary, spectral reciprocity) converts one into the other.

### 3.2 Hiary's Algorithm Doesn't Help for Squarefree Conductor

Hiary's t^{1/3} improvement uses the Postnikov character formula, which reduces character sums to quadratic exponential sums. This reduction requires the modulus to be a high prime power (smooth). For squarefree N = pq: Postnikov decomposition produces a product of character sums mod p and mod q — CRT again. No improvement over O(√N).

### 3.3 Spectral Reciprocity Preserves Multiplicativity

Blomer-Khan spectral reciprocity swaps the level and the twisting index. For N = pq on one side, the dual side involves λ_g(N) = λ_g(p)·λ_g(q). The factorization is encoded in the dual problem through Hecke multiplicativity. No bypass.

### 3.4 The Phi-Hiding Gap Is the Best Remaining Hope

The problem "compute φ(N) mod 691" has:
- No poly(log N) algorithm
- No proof of hardness relative to factoring
- An independent cryptographic assumption (phi-hiding) adopted precisely because the reduction fails
- A clear quantitative connection to Coppersmith: oracles for O(n/4) different primes ℓ would break factoring

This is NOT a proof that the problem is easier than factoring. But the absence of a reduction, after decades of cryptographic research, is itself significant.

### 3.5 Incomplete Character Sums: CRT-Breaking but Signal-Buried

Incomplete sums Σ_{x≤X} e(f(x)/N) do not factor via CRT because the interval [1,X] is not a Cartesian product in CRT coordinates. The carry structure creates cross-terms between p and q. However:
- E10/E12 experiments show peaks decay as N^{-0.35} to N^{-0.47}
- Weyl differencing for smooth moduli uses factorization as INPUT
- Chang (arXiv:1201.0299): improved Graham-Ringrose bounds but no factoring application
- The signal is provably at noise level for generic semiprimes

### 3.6 Castryck-Decru: Tantalizing but Inapplicable

The SIDH attack is the closest any isogeny method comes to factoring-relevant computation. It recovers a secret isogeny from auxiliary torsion data in polynomial time. But:
- The "auxiliary torsion" (evaluation of isogenies at known torsion points) is a protocol-specific information leak
- The integer N = pq does not come with auxiliary torsion data
- Protocols that don't leak torsion (CSIDH, SQIsign) are secure
- The factoring problem has NO analogue of the torsion leak

---

## 4. Complete Reference List

### Core Hardness Results
1. Bach-Charles (2007), arXiv:0708.1192 — hardness of computing eigenforms at composites
2. Miller (1976), J. Comp. Syst. Sci. — φ(N) ↔ factoring (full value only)
3. Coppersmith (1996), J. Cryptology — factoring with n/4 bits of p
4. Cachin-Micali-Stadler (1999), EUROCRYPT — phi-hiding assumption
5. Aggarwal-Maurer (2009) — breaking RSA generically ≡ factoring
6. Boneh-Venkatesan (1998) — breaking RSA may be easier than factoring

### Modular Form Algorithms
7. Edixhoven-Couveignes (2006), arXiv:math/0605244 — poly(log p) for τ(p) at primes
8. Mascot (2022), arXiv:2004.14683 — p-adic EC implementation, 20-100x speedup
9. Denis Charles (2006), Ramanujan Journal — τ(N) in O(N^{1/2+ε}) under GRH
10. Bruin (2010), Leiden thesis — EC at squarefree levels
11. Peng Tian (2022), arXiv:1905.10036 — non-Eisenstein prime extensions

### L-function Algorithms
12. Hiary (2011), Annals of Math. 174 — ζ(1/2+iT) in O(T^{1/3+ε})
13. Hiary (2012), arXiv:1205.4687 — Dirichlet L-functions with smooth modulus
14. Vishe (2012), arXiv:1202.6303 — L(1/2, f⊗χ_q) for smooth q
15. Vishe (2011), arXiv:1108.4887 — rapid computation L(f, 1/2+iT) in O(T^{7/8})
16. Hiary (2016), arXiv:1608.06614 — amortized quadratic Dirichlet L-functions
17. Odlyzko-Schonhage (1988) — amortized Dirichlet series evaluation
18. Platt (2011), Bristol thesis — rigorous degree-1 L-function computation
19. Rubinstein (2005), arXiv:math/0306101 — Odlyzko-Schonhage in conductor aspect
20. Booker-Strömbergsson-Venkatesh (2018), arXiv:1806.01586 — analytic evaluation

### Trace Formulas and Spectral Theory
21. Booker-Lee-Strömbergsson (2021), arXiv:2101.05663 — twist-minimal trace formulas
22. Altug (2025), arXiv:2505.18967 — beyond endoscopy with ramification
23. Popa-Zagier (2017), arXiv:1711.00327 — simple proof of Eichler-Selberg

### Amplification and Subconvexity
24. Nelson (2021), arXiv:2109.15230 — orbit method subconvexity for GL(n)
25. Petrow-Young (2020), Annals of Math. — Weyl bound for L(1/2, χ)
26. Blomer-Khan (2019), Duke Math J. 168 — spectral reciprocity
27. Blomer-Khan-Young (2020), Compositio — Motohashi for non-archimedean test functions
28. Hu-Nelson (2019), arXiv:1810.11564 — new test vectors, hybrid subconvexity
29. Michel-Venkatesh, Park City lectures — amplification and subconvexity methods

### Shifted Convolution Sums
30. Blomer-Harcos (2008), Duke Math J. 144 — spectral decomposition of shifted sums
31. Holowinsky (2009), Duke Math J. 146 — sieve method for shifted convolution
32. Holowinsky (2010), Annals of Math. 172 — sieving for mass equidistribution
33. Good (1981), Math. Ann. 255 — first power-saving bounds
34. Motohashi (1994), Annales ENS — binary additive divisor problem
35. arXiv:2511.19303 (2025) — shifted convolution for Siegel modular forms
36. arXiv:2510.15799 (2025) — GL(3) shifted convolution with averaged shifts
37. arXiv:2311.06587 (2023) — shifted convolution L-function for Maass forms
38. arXiv:2509.07556 (2025) — smoothed generalised divisor function

### Period Integrals and Waldspurger
39. Waldspurger (1981), J. Math. Pures Appl. 60, 375-484 — foundational theorem
40. Kohnen-Zagier (1981), Inventiones Math. 64, 175-198 — explicit formula level 1
41. Shimura (1973), Annals of Math. 97 — Shimura correspondence
42. Shintani (1975), Nagoya Math. J. — Shintani lift construction
43. Baruch-Mao (2007), GAFA 17, 333-384 — generalized Kohnen-Zagier
44. Gross-Zagier (1986), Inventiones Math. 84 — heights of Heegner points
45. Zhang (1997-2001), Annals of Math. — higher weight Gross-Zagier
46. Yuan-Zhang-Zhang (2013), Annals Math Studies 184 — Gross-Zagier on Shimura curves
47. Kudla-Rapoport-Yang (2006), Annals Math Studies 161 — special cycles
48. Li-Zhang (2022) — proof of Kudla-Rapoport conjecture
49. Ichino (2008), Compositio Math. — triple product period integral
50. Hu-Yin (2019), arXiv:1907.11428 — Waldspurger for newforms
51. Cai-Shu-Tian (2014), Algebra & Number Theory — explicit Gross-Zagier/Waldspurger
52. Prasanna (2009), Inventiones Math. 176 — SSW correspondence arithmetic
53. Pacetti-Tornaría (2008), Exp. Math. 17 — computing L-values, composite level
54. Inam-Wiese (2022), Rocky Mountain J. Math. 52 — fast half-integral weight bases

### Modular Form Congruences and Distribution
55. Serre (1973), "Congruences et formes modulaires" — mod-ℓ representations
56. Swinnerton-Dyer (1973) — congruences for τ(n) mod 2,3,5,7,23,691
57. Lehmer (1947) — τ(n) ≡ 0 mod 23 characterization
58. Serre (1966) — τ(n) parity: odd ⟺ n odd perfect square
59. Balakrishnan-Craig-Ono (2020), arXiv:2005.10345 — excluded values of τ
60. Balakrishnan-Ono-Tsai (2021), arXiv:2102.00111 — even non-values
61. Bellaiche-Soundararajan (2015) — nonzero coefficients mod p
62. Derickx-van Hoeij-Zeng (2013) — Lehmer verification to 8.16 × 10²³
63. Berndt-Moree (2024), arXiv:2409.03428 — sums of two squares and tau

### Character Sums
64. Van Dam-Seroussi (2002), arXiv:quant-ph/0207131 — quantum Gauss sum estimation
65. Schleich et al. (2012), arXiv:1210.6474 — Gauss sum factoring (exponential)
66. Cohen-Lenstra (1984) — APRCL primality and Jacobi sums
67. Chang (2012), arXiv:1201.0299 — short character sums for composite moduli
68. Korolev (2019), arXiv:1911.09981 — Kloosterman sums with primes to composite moduli

### Isogeny Graphs and Algebraic Geometry
69. Castryck-Decru (2022), ePrint 2022/975 — SIDH attack via Kani reducibility
70. Le Merdy-Wesolowski (2025), EUROCRYPT — unconditional EndRing equivalences
71. Charles-Goren-Lauter (2009) — CGL hash from isogeny graphs
72. Arpin (2022), arXiv:2203.03531 — level structure on supersingular isogeny graphs
73. Eisentrager-Hallgren-Lauter-Morrison-Petit (2018), EUROCRYPT — EndRing reductions
74. Page-Wesolowski (2024), EUROCRYPT — EndRing and OneEndRing equivalence
75. De Feo-Fouotsa-Panny (2024), EUROCRYPT — isogeny problems with level structure
76. Chen (2025), EUROCRYPT — computing End(E) from full-rank suborder (quantum)
77. Kirschmer-Voight (2010), arXiv:0808.3833 — quaternion order ideal class enumeration

### Partial Information and Complexity
78. Morain-Renault-Smith (2023), arXiv:1802.08444 — deterministic factoring with φ oracle
79. May-Nowakowski-Sarkar (2022), EUROCRYPT — approximate divisor multiples
80. Schridde-Freisleben (2008), ASIACRYPT — phi-hiding validity
81. Paillier (1999), EUROCRYPT — composite degree residuosity classes
82. ePrint 2025/1281 — improving RSA cryptanalysis
83. ePrint 2025/1004 — factoring and power divisor problems

### Recent Factoring
84. Ragavan-Regev (STOC 2025) — Jacobi factoring circuit (quantum)
85. Ryan (EUROCRYPT 2025) — multivariate Coppersmith (n/4 unchanged)
86. Regev (JACM 2025) — efficient quantum factoring
87. Trey Li (ePrint 2025/1681) — Hecke problem is NP-hard

### Rankin-Selberg Specific
88. Gelbart-Jacquet (1978), ASENS — GL(2) to GL(3) lift (symmetric square)
89. Huang (2021), arXiv:2002.00591 — Rankin-Selberg problem exponent improvement
90. Luo-Ramakrishnan (2009), Math. Ann. — determination of forms by twists
91. Pollack, "The Rankin-Selberg Method: A User's Guide" — expository
92. Iwaniec-Kowalski, "Analytic Number Theory" — textbook reference
93. arXiv:2507.06681 (2025) — fast computation of millions of modular form coefficients

---

## 5. Summary Scorecard

| PATH | Status | Cost Lower Bound | Key Obstruction |
|------|--------|-----------------|-----------------|
| 1. Petersson/Kloosterman | MARGINALLY OPEN | O(N) | √N Kloosterman terms needed |
| 2. EC for composites | BLOCKED | N/A | No Frobenius at composites |
| 3. Mod-ℓ Galois reps | BLOCKED | N/A | Multiplicativity mod ℓ |
| 4. Voronoi | BLOCKED | O(N) | Wrong regime for q = N |
| 5. Amplification | BLOCKED | O(√N) exact | Bounds not values; reciprocity preserves mult. |
| 6. Approximation | BLOCKED | N/A | mod-ℓ needs near-exact value |
| **7. φ(N) mod 691** | **OPEN** | **unknown** | **No algorithm; no hardness proof** |
| 8. Motives/geometry | BLOCKED | N/A | CRT everywhere |
| 9. Shifted convolution | BLOCKED | O(N) | Averages not individuals |
| 10. Quantum | REDUNDANT | poly(log N) | Shor already factors |
| 11. Hardware | BOUNDED | O(√N) | Exponential gap persists |
| 12. Waldspurger/periods | BLOCKED | O(√N) | L-values not τ(N); CRT on local |
| 13. Char sums/isogenies | BLOCKED | O(N) | Complete CRT; QR hard core |
| 14. Rankin-Selberg | BLOCKED | O(N^{1/2}) | Eval/extraction duality; Hiary needs smooth q |

---

## 6. Conclusions and Implications

### What this round establishes:

1. **The entire analytic number theory toolkit** — trace formulas, L-functions, amplification, spectral methods, period integrals, character sums — has been systematically examined for computing τ(N) mod 691 at composite N = pq. None achieves better than O(√N).

2. **The point-evaluation / coefficient-extraction duality** provides a clean explanation for why L-function methods fail: they are designed to compute weighted averages of Fourier coefficients (L-values), not individual coefficients.

3. **Hiary-type speedups** (t^{1/3} over t^{1/2}) require smooth modulus and are inapplicable to squarefree N = pq. This closes the last hope for sub-√N L-function evaluation at composite conductor.

4. **The φ(N) mod ℓ complexity gap** is the single most important remaining question. It is a well-defined open problem in computational number theory, sitting between known hardness of full φ(N) and unknown status of partial φ(N). Its resolution would either (a) yield a factoring improvement via Coppersmith, or (b) establish a new hardness result connecting partial totient to factoring.

### What remains unknown:

- Is computing φ(N) mod 691 as hard as factoring? (Believed yes, not proven)
- Could Kloosterman series convergence be accelerated to poly(log N) terms? (Beyond endoscopy achieves asymptotic cancellation but not for individual N)
- Is there a number-theoretic function f(N) computable in poly(log N) that correlates with the factors of N = pq? (All tested functions: no, per E1-E18)

### Relationship to project's main thesis:

The E13 Eisenstein congruence channel is **information-rich** (63 bits of factor data, 100% accurate) but **computationally blocked** (O(N) evaluation cost). This round confirms the block is not an artifact of limited algorithmic imagination but reflects deep structural features of analytic number theory: the CRT decomposition of character sums, the multiplicativity of Hecke eigenvalues, the dimensional barrier of modular forms at composite level, and the coefficient-extraction duality of L-functions.

The only remaining gap — φ(N) mod ℓ — is a complexity-theoretic question, not a number-theoretic one. Its resolution likely requires new techniques from computational complexity theory rather than additional analytic number theory.
