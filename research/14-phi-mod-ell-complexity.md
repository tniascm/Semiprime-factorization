# Round 4 Research Report: The φ(N) mod ℓ Complexity Gap

**Date:** 2026-02-27
**Scope:** Exhaustive investigation of the single remaining open question from Round 3: can φ(N) mod ℓ be computed without factoring N = pq? Covers cryptographic assumptions, Coppersmith thresholds, complexity theory, algebraic methods, elliptic curve approaches, and recent literature (2020-2026).

---

## 1. Executive Summary

This round investigated every known avenue for computing φ(N) mod ℓ (equivalently, (p+q) mod ℓ, or σ_k(N) mod ℓ) without factoring N = pq. Eight parallel research threads covered:

1. **Phi-Hiding Assumption** — full history, status, attacks
2. **Coppersmith thresholds** — exact bit requirements, CRT accumulation
3. **Complexity-theoretic status** — NP ∩ coNP placement, oracle separations
4. **Divisor sums mod ℓ** — Newton's identity recovery, algebraic complexity
5. **NP ∩ coNP landscape** — hierarchy of factoring-adjacent problems
6. **Algebraic methods** — 10 avenues (group theory, Carmichael, ECM, pairings, etc.)
7. **Elliptic curves over Z/NZ** — Schoof, SEA, Satoh, Kedlaya, ECPP
8. **Recent papers (2020-2026)** — 30+ papers surveyed

### Central Conclusion

**The gap is real but almost certainly not exploitable.** Every algebraic, analytic, and geometric method for computing φ(N) mod ℓ reduces to factoring or hits the CRT decomposition barrier. The problem sits in NP ∩ coNP with no known algorithm and no hardness proof, but the convergence of evidence from multiple independent directions strongly suggests it is as hard as factoring.

### Key Numbers

| Parameter | Value |
|-----------|-------|
| Bits from single φ(N) mod 691 oracle | ~9.4 bits |
| Coppersmith threshold for RSA-2048 | 512 bits |
| Primes needed to reach threshold | ~76 (all primes up to 383) |
| E13 channels accumulated | 63.3 bits (12.4% of threshold) |
| Coppersmith n/4 threshold changed since 1996? | **No** |
| Any poly(log N) algorithm for φ(N) mod ℓ? | **No** |
| Any proof φ(N) mod ℓ is as hard as factoring? | **No** |

---

## 2. The Phi-Hiding Assumption (PHA)

### 2.1 Definition and Origin

**Paper:** Cachin-Micali-Stadler, "Computationally Private Information Retrieval with Polylogarithmic Communication," EUROCRYPT 1999, LNCS 1592, pp. 402-414.

**PHA states:** Given N = PQ (RSA modulus) and two small primes e₁, e₂, exactly one dividing φ(N), no polynomial-time algorithm can distinguish which one divides φ(N) with probability significantly greater than 1/2.

They **assumed** PHA (did not prove it) and **proved** their PIR protocol achieves computational privacy under PHA. PHA was explicitly introduced as a new, standalone assumption — not derived from factoring hardness.

### 2.2 Relationship to Factoring

**Factoring → PHA?** No known reduction. PHA is considered **strictly stronger** than the factoring assumption. The hierarchy:
```
Factoring (weakest) < RSA assumption < Phi-Hiding (strongest)
```

**PHA → Factoring?** Yes, trivially: if you can factor N, you can check ℓ | φ(N). Contrapositive: PHA hard ⟹ factoring hard.

**Multi-query PHA → Factoring:** A PHA oracle queried on many small primes ℓ reveals which divide φ(N), recovering φ(N) mod M for various M. With enough queries, Coppersmith recovers the factorization. So a PHA oracle does imply factoring, through multi-query reduction.

### 2.3 Decision vs. Computation

The **decision** problem (does ℓ | φ(N)?) and the **computation** problem (compute φ(N) mod ℓ) are related but distinct:

- Computation trivially solves decision (check if output is 0)
- Decision does NOT obviously solve computation for ℓ > 2
- For prime ℓ: decision asks "is φ(N) mod ℓ = 0?" — strictly weaker than full residue
- Search-to-decision reduction possible with auxiliary modulus construction, but non-trivial

### 2.4 Attacks and Validity

**Schridde-Freisleben (ASIACRYPT 2008):** PHA is invalid for N = P·Q^{2e}. The repeated prime power structure leaks information. Does NOT break PHA for standard RSA moduli N = PQ.

**Coppersmith attack bound:** PHA holds when log(ℓ) ≤ log(P)/2 - λ (security parameter). For 1024-bit P with λ = 80: ℓ must be smaller than ~432 bits.

**Multi-prime PHA (Herrmann, AFRICACRYPT 2011; Xu et al., CT-RSA 2016):** For N = p₁·p₂·...·pₘ, attacks work when ℓ > N^{2/(3m) - 1/(4m²)}. For m=3: ℓ > N^{7/36}.

### 2.5 Status (2020-2026)

**No new advances.** The assumption is in a mature state:
- Basic PHA for N = PQ remains unbroken
- Multi-prime variants well-characterized
- No new reductions to/from standard assumptions
- No quantum attacks beyond generic Shor
- Continues to be used through the lossy RSA framework

### 2.6 Protocols Relying on PHA

1. **Cachin-Micali-Stadler PIR (1999)** — polylog communication private information retrieval
2. **Gentry-Ramzan PIR (ICALP 2005)** — constant communication rate
3. **Gentry-MacKenzie-Ramzan PAKE (CCS 2005)** — password-authenticated key exchange
4. **Kiltz-O'Neill-Smith (CRYPTO 2010)** — RSA-OAEP instantiability (lossy RSA)
5. **Kakvi-Kiltz (EUROCRYPT 2012)** — tight RSA-FDH security
6. **Lewko-O'Neill-Smith (EUROCRYPT 2013)** — PKCS #1 v1.5 semantic security
7. **Hemenway-Ostrovsky-Rosen (TCC 2015)** — non-committing encryption

---

## 3. Coppersmith Thresholds and CRT Accumulation

### 3.1 The Core Theorem

**Coppersmith (1996):** Let N = pq balanced. Given p₀ = p mod M for known M > N^{1/4}, then N can be factored in polynomial time.

**Proof sketch:** Set f(x) = Mx + p₀, which has root x₀ = (p - p₀)/M modulo p. Since |x₀| = p/M < N^{1/2}/M < N^{1/4} when M > N^{1/4}, Coppersmith's lattice method recovers x₀ in poly(log N) time.

### 3.2 The Reduction Chain: φ(N) mod M → Factoring

```
φ(N) mod M  ──(trivial arithmetic)──>  (p+q) mod M
            ──(quadratic formula mod M)──>  p mod M  (2 candidates per prime factor of M)
            ──(Coppersmith lattice)──>  p  (exact)
            ──(division)──>  q = N/p
```

**Step 1:** (p+q) mod M = (N + 1 - φ(N) mod M) mod M. Trivial.

**Step 2:** p, q are roots of t² - s·t + N ≡ 0 (mod M) where s = (p+q) mod M. Discriminant (p-q)² is always a perfect square. Solve mod each prime factor of M, combine via CRT. Yields 2^k candidates for k prime factors — Coppersmith handles ambiguity.

**Step 3:** Coppersmith's theorem with M > N^{1/4}.

### 3.3 Exact Bit Requirements

| RSA Size | n bits | p bits | Coppersmith threshold | Primes needed (up to B) | # primes |
|----------|--------|--------|----------------------|------------------------|----------|
| RSA-1024 | 1024 | 512 | 256 bits | B ≈ 193 | ~44 |
| RSA-2048 | 2048 | 1024 | 512 bits | B ≈ 383 | ~76 |
| RSA-4096 | 4096 | 2048 | 1024 bits | B ≈ 761 | ~135 |

**Derivation:** Product of all primes ≤ B (the primorial) has log₂ ≈ B/ln(2) ≈ 1.4427·B bits. Setting this ≥ n/4 gives B ≈ n·ln(2)/4.

### 3.4 E13 Connection

E13's 7 Eisenstein congruence channels use primes {691, 3617, 43867, 283, 617, 131, 593}, yielding ~63.3 bits. This is **12.4% of the RSA-2048 Coppersmith threshold** (512 bits). Closing the gap requires many more channels (higher-weight Bernoulli primes grow super-exponentially), and the O(N) computational barrier makes this moot.

### 3.5 Threshold Unchanged Since 1996

**Ryan (EUROCRYPT 2025):** Automated multivariate Coppersmith via Groebner bases and graph optimization. Confirms the same n/4 threshold — improvements are in lattice efficiency, not the fundamental bound.

**May-Nowakowski-Sarkar (EUROCRYPT 2022):** 1/3 of CRT-exponents d_p, d_q suffice for CRT-RSA. This is a different attack surface (partial key exposure, not partial factor knowledge). Does NOT change n/4 for φ(N) mod M.

**Herrmann-May (ASIACRYPT 2008):** ~70% random bits of p suffice, but runtime exponential in number of blocks. Polynomial only for O(log log N) blocks. The modular case (p mod M) already handles this as a single block.

---

## 4. Complexity-Theoretic Status

### 4.1 Miller's Reduction and Its Limitation

**Miller (1976):** Given full φ(N), factor N deterministically:
- Compute p + q = N + 1 - φ(N)
- Solve x² - (p+q)x + N = 0 via quadratic formula

**Why partial φ(N) fails:** The quadratic formula requires the EXACT integer p + q. Knowing (p+q) mod ℓ gives p mod ℓ (via quadratic over F_ℓ), but recovering p from p mod ℓ requires ℓ > √N or CRT with enough primes. The reduction is fundamentally non-modular.

**Morain-Renault-Smith (arXiv:1802.08444, AAECC 2023):** Deterministic poly-time factoring from full φ(N) for squarefree N with p > √N, using LLL. Strengthens Miller by removing randomness. Still requires FULL φ(N).

### 4.2 Complexity Class Placement

"Compute φ(N) mod ℓ" as decision problem "Is φ(N) mod ℓ = r?":
- **In NP:** witness = factorization of N → compute φ(N) mod ℓ → verify
- **In coNP:** same witness proves the answer is NOT r
- **In NP ∩ coNP:** yes, unconditionally
- **In UP ∩ coUP:** yes (unique factorization)
- **In P:** unknown
- **Equivalent to factoring:** unknown — no reduction in either direction for partial values

### 4.3 The Sharp Dichotomy

| Problem | Complexity | Equivalent to Factoring? |
|---------|-----------|-------------------------|
| Full φ(N) | Equivalent to factoring | **Yes** (Miller 1976) |
| Full σ_k(N) | Equivalent to factoring | **Yes** (Bach-Miller-Shallit 1986) |
| Full λ(N) | Equivalent to factoring | **Yes** |
| Full ord_N(g) | Equivalent to factoring | **Yes** |
| RSA private key d | Equivalent to factoring | **Yes** (May 2004) |
| Arbitrary square roots mod N | Equivalent to factoring | **Yes** (Rabin 1979) |
| φ(N) mod ℓ (small ℓ) | NP ∩ coNP, status unknown | **Unknown** |
| σ_k(N) mod ℓ (small ℓ) | NP ∩ coNP, status unknown | **Unknown** |
| QR decision | NP ∩ coNP, assumed hard | **Unknown** |
| DCR (Paillier) | Assumed hard | **Unknown** |
| RSA problem (e-th roots) | Assumed hard | **Unknown** |

**The gap:** Full multiplicative function values at composites are all equivalent to factoring. Partial/modular values float in the NP-intermediate zone with no hardness reduction in either direction.

### 4.4 No Oracle Separation Known

No oracle is known relative to which φ(N) mod ℓ is easy but factoring is hard (or vice versa). Constructing such a separation would require showing a structural gap between the two problems — an open question in computational complexity.

### 4.5 Generic Model Results

**Aggarwal-Maurer (J. Cryptology 2013):** In the generic ring model (ring operations on Z/NZ only):
- Breaking RSA ≡ factoring
- QR ≡ factoring
- Jacobi symbol ≡ factoring

**But:** The Jacobi symbol is efficiently computable in reality (via quadratic reciprocity). This proves the generic ring model gives **spurious hardness** — it cannot distinguish partial totient from factoring. Generic model results are uninformative for our question.

### 4.6 Related Independent Assumptions

Three assumptions sit in the same limbo as φ(N) mod ℓ — implied by factoring hardness, not proven equivalent:

1. **Quadratic Residuosity Assumption (QRA)** — Goldwasser-Micali 1984
2. **Decisional Composite Residuosity (DCR)** — Paillier 1999
3. **Phi-Hiding Assumption (PHA)** — Cachin-Micali-Stadler 1999

All three are adopted as independent cryptographic assumptions precisely because no factoring equivalence exists.

---

## 5. Divisor Sums mod ℓ: Newton's Identity Analysis

### 5.1 The Reduction σ_k(N) mod ℓ → φ(N) mod ℓ

For N = pq:
- σ₁(N) mod ℓ = (1 + p + q + N) mod ℓ → gives (p+q) mod ℓ directly
- σ_k(N) mod ℓ = (1 + p^k + q^k + N^k) mod ℓ → gives (p^k + q^k) mod ℓ

Since τ(N) ≡ σ₁₁(N) mod 691 (Ramanujan's congruence), computing τ(N) mod 691 gives (p¹¹ + q¹¹) mod 691.

### 5.2 Recovering p mod ℓ from Power Sums

Given e₂ = pq mod ℓ (known) and s_k = p^k + q^k mod ℓ, Newton's recurrence gives:

```
s_m = e₁·s_{m-1} - e₂·s_{m-2},  s₀ = 2, s₁ = e₁
```

Unrolling: s₁₁(e₁) is a **degree-11 polynomial** in e₁ over F_691:

```
s₁₁ = e₁¹¹ - 11·N·e₁⁹ + 44·N²·e₁⁷ - 77·N³·e₁⁵ + 55·N⁴·e₁³ - 11·N⁵·e₁
```

Over F_691 this has **at most 11 roots**. E13 experiments found exactly 1 root in all cases → 2 p-candidates = {p mod ℓ, q mod ℓ}.

### 5.3 The Bijection Property

Since gcd(11, 690) = gcd(11, 691-1) = 1, the map x → x¹¹ is a **bijection** on (Z/691Z)*. This means:
- Every nonzero element has a unique 11th root
- The power sum s₁₁ carries **maximal information** (no 2-to-1 collapse)
- But this does NOT simplify the Newton polynomial (lower-order terms involving N mod 691 prevent using the bijection directly)

### 5.4 Algebraic Complexity of Recovery

Once s_k mod ℓ is known, extracting p mod ℓ costs O(k² log ℓ) via standard root-finding over finite fields (Berlekamp or Cantor-Zassenhaus). **The recovery is computationally trivial.** The hardness is entirely in EVALUATION — obtaining σ_k(N) mod ℓ without factoring.

### 5.5 The Circularity

Four routes to compute σ₁₁(N) mod 691, all blocked:

| Route | Cost | Obstruction |
|-------|------|-------------|
| q-expansion mod 691 | O(N) | Exponential in log N |
| Trace formula | O(√N) + correction | Eisenstein term needs divisors of N |
| σ₁₁ directly | N/A | Multiplicative: needs p, q |
| Galois representation (EC) | poly(log p) primes only | Bach-Charles blocks composites |

---

## 6. Algebraic Methods: 10 Avenues, All Blocked

### 6.1 Group Theory of (Z/NZ)*

**BLOCKED.** The group has order φ(N), but determining |G| mod ℓ from the group structure requires O(√N) via baby-step giant-step. The order problem is equivalent to factoring (Miller 1976).

### 6.2 Carmichael Function λ(N)

**BLOCKED.** Computing λ(N) = lcm(p-1, q-1) is equivalent to factoring. Even λ(N) mod ℓ leaks structural information about the ℓ-adic valuations of (p-1) and (q-1).

### 6.3 Statistical Properties of Element Orders

**BLOCKED.** Computing ord(g) for random g ∈ (Z/NZ)* requires O(√N). A single order already reveals factorization via Miller's algorithm.

### 6.4 Pohlig-Hellman Approximation

**BLOCKED.** The natural approximation uses (N²-1)/ℓ instead of φ(N)/ℓ. The correction term involves (p+q) in the exponent — the very quantity we seek. Testing all ℓ candidates for (p+q) mod ℓ only works for the trivial case ℓ | φ(N).

### 6.5 Index Calculus in (Z/NZ)*

**BLOCKED.** Relations found in (Z/NZ)* hold in both components (Z/pZ)* and (Z/qZ)* simultaneously. Extracting individual component information requires factoring. Sub-exponential runtime already exceeds poly(log N).

### 6.6 Pocklington-Type Certificates

**NOT APPLICABLE.** Designed for primality proving, not composite analysis. Requires partially factoring φ(N), which requires knowing φ(N).

### 6.7 Baby-Step Giant-Step

**BLOCKED.** O(√N) inherent barrier. Intermediate values depend on the discrete log modulo arbitrary step sizes, not on the group order modulo ℓ. No mod-ℓ information leaks.

### 6.8 ECM with CM Curves

**BLOCKED.** E(Z/NZ) = E(F_p) × E(F_q) by CRT. Any computation over Z/NZ decomposes into independent computations mod p and mod q. The CM structure helps with inversion (extracting p from group orders) but not evaluation (computing group orders).

### 6.9 PARI/GP znorder

**No shortcut.** Falls back to O(√N) generic algorithms without known factorization. No specialized method for computing orders mod small primes.

### 6.10 Weil and Tate Pairings

**BLOCKED.** Require field (not ring) arithmetic. Over Z/NZ, CRT decomposition applies. Galbraith-McKee (ANTS 2005) proved: reduced Tate pairing computation on E(Z/NZ) implies factoring.

---

## 7. Elliptic Curves over Z/NZ: Deep Dive

### 7.1 Kunihiro-Koyama Theorem (EUROCRYPT 1998)

**Computing |E(Z/NZ)| is randomly polynomial-time equivalent to factoring N.**

Strengthened by Dieulefait-Urroz (Mathematics 2020): equivalence is **deterministic** for a pair of twisted curves, using Coppersmith in the reduction.

### 7.2 Why Schoof's Algorithm Fails for Composite N

Three fundamental breakdowns:

**(a) No Frobenius endomorphism.** Over F_p, φ_p: (x,y) → (x^p, y^p) is a well-defined inseparable isogeny. Over Z/NZ, the map (x,y) → (x^N, y^N) is NOT an endomorphism — Z/NZ has zero divisors, the notion of degree and separability break down.

**(b) Division polynomials require a field.** Schoof works in F_q[x]/(ψ_ℓ(x)), requiring polynomial GCD, factoring, and inversion. Over Z/NZ (not an integral domain), polynomial GCD is undefined. Attempting it either reveals a factor (= ECM) or gives CRT-separable output.

**(c) No single trace of Frobenius.** The "Frobenius at N" decomposes via CRT: on E(F_p) it acts as φ_q, on E(F_q) it acts as φ_p. There is no single quadratic T² - t·T + N describing the action globally.

### 7.3 Why SEA Algorithm Fails

Every step requires primality: modular polynomial evaluation (needs j(E) ∈ F_q), root finding (needs Frobenius x → x^q), isogeny computation (Vélu formulas need field), eigenvalue recovery (needs Frobenius on kernel).

### 7.4 Why Satoh and Kedlaya Fail

Both are p-adic algorithms requiring prime characteristic:
- **Satoh:** Lifts from F_q to Z_p, computes Frobenius via canonical lift. No Z_N analogue (Z_N = Z_p × Z_q by CRT — any computation decomposes).
- **Kedlaya:** Monsky-Washnitzer cohomology over fields. Not defined for varieties over non-integral rings.

### 7.5 Why ECPP Fails for Composites

Goldwasser-Kilian/Atkin-Morain ECPP:
- CM method needs Cornacchia's algorithm (requires prime modulus)
- Hilbert class polynomial root-finding mod N requires factoring
- Verification theorem: if N composite, verification fails with high probability

ECPP does NOT compute |E(Z/NZ)| mod ℓ as a subroutine — it assumes primality.

### 7.6 ECM Success Mode: No Information Leakage

In Lenstra's ECM, the non-factoring outcome (gcd stays at 1) tells us only that the smooth bound B was insufficient for both component orders. Each trial provides ~1 bit (smooth vs. not-smooth) about |E(F_p)| and |E(F_q)|. No mod-ℓ information about group orders or p+q leaks.

### 7.7 Division Polynomials mod N

ψ_ℓ(x) can be computed mod N. Analyzing it either:
- Fails (non-invertible leading coefficient) → reveals factor (= ECM)
- Succeeds → output lives in (Z/NZ)[x], conflating roots mod p and mod q

Resultants, discriminants, and GCDs of ψ_ℓ mod N are all CRT-separable quantities. No escape from the barrier.

---

## 8. The NP ∩ coNP Landscape

### 8.1 Problems in NP ∩ coNP Not Known in P

Remarkably few natural problems sit here (Gasarch 2024):
- **Factoring** (Pratt certificates)
- **Discrete log mod composite**
- **Parity games** (quasi-polynomial: Calude et al. STOC 2017)
- **Graph isomorphism** (conditionally, under derandomization)
- **Unknotting** (quasi-polynomial: Lackenby 2021)
- **ARRIVAL** (zero-player switching game)

### 8.2 Hierarchy of Factoring-Related Problems

```
EQUIVALENT TO FACTORING (poly-time reductions both ways):
  Full φ(N)  ≡  Full λ(N)  ≡  RSA key d  ≡  Square roots mod N
  ≡  Order finding mod N  ≡  Discrete log mod N  ≡  Full σ_k(N)

NOT KNOWN EQUIVALENT (factoring → these, but not reverse):
  QR decision  |  Higher power residuosity  |  DCR (Paillier)
  PHA  |  RSA problem  |  φ(N) mod ℓ  |  CDH in (Z/NZ)*

IN P (easy despite generic model hardness):
  Jacobi symbol  |  Primality (AKS)  |  GCD  |  DDH in (Z/NZ)*

NP-COMPLETE (if NP ≠ coNP, harder than factoring):
  Bounded square root: ∃x<c : x²≡a (mod N)?  (Manders-Adleman 1978)
  Bounded factoring: ∃ factor in [a,b]?
```

### 8.3 Bit Security

Alexi-Chor-Goldreich-Schnorr (1988) + Hastad-Naslund (2004): predicting ANY single bit of x given E_N(x) = x^e mod N is as hard as inverting RSA entirely. No individual bit of a factor can be efficiently extracted from N under the RSA assumption.

### 8.4 McCurley's Problems (1987, 1994)

McCurley systematically studied reductions among 13 number-theoretic problems. Many remain open after 30+ years, including the QR↔factoring question (open since Goldwasser-Micali 1982).

---

## 9. Recent Literature (2020-2026)

### 9.1 Papers That Shift Partial-Information Thresholds

| Paper | Year | Venue | Result |
|-------|------|-------|--------|
| May-Nowakowski-Sarkar | 2022 | EUROCRYPT | 1/3 of CRT-exponents d_p, d_q suffice |
| Feng-Nitaj-Pan | 2024 | CiC | Saves log₂(e) bits in small-e PKE |
| Zhao-Zhang-Cao et al. | 2024 | CiC | Implicit factoring with shared arbitrary bits |
| Ajani-Bright | 2024 | ISSAC | SAT + Coppersmith hybrid for random bit positions |
| Ryan | 2025 | EUROCRYPT | Automated Coppersmith lattices (same thresholds) |

### 9.2 Papers That Do NOT Change the Landscape

| Paper | Year | Issue |
|-------|------|-------|
| Mehta-Rana | 2026 (ePrint) | Likely O(|p-q|), no complexity analysis |
| Bansimba et al. | 2025 (arXiv) | ML approximation of φ(N) — useless without exact value |
| Schnorr | 2021 (ePrint) | SVP factoring claim — debunked (Ducas et al.: 0/1000 success) |
| D-Wave RSA-2048 | 2024 | Trivially close primes — debunked (Gutmann-Neuhaus 2025) |
| Tesoro et al. | 2024 (arXiv) | Tensor network: 100-bit RSA, unclear if scales |

### 9.3 Quantum (Still Not Classical)

- **Regev (2023) + Ragavan-Vaikuntanathan:** O(n^{3/2}) gates, √n runs. Space O(n log n). Unconditional.
- **Litinski-Nickerson (STOC 2025):** Jacobi factoring circuit. O(n) gates, sublinear qubits. Only for P²Q form.
- **Gutmann-Neuhaus (ePrint 2025/1237):** All quantum factoring "records" used artificially constructed numbers. Replicated on 1981 VIC-20.

### 9.4 Oracle Factoring

**Dabrowski-Pomykala-Shparlinski (J. Complexity 2023):** Deterministic poly-time factoring of almost all N given |E(Z/NZ)| for certain curves. Extends Miller-Shoup to elliptic curve setting. Confirms that EC point counts are oracle-sufficient for factoring.

### 9.5 Dieulefait-Urroz (Mathematics 2020)

Factoring ≡ counting points on twisted elliptic curve pair over Z/NZ, in **deterministic** polynomial time. Strengthens Kunihiro-Koyama from randomized to deterministic.

---

## 10. Synthesis: Why the Gap Is (Almost Certainly) Not Exploitable

### 10.1 Convergence of Evidence

Seven independent lines of evidence all point to φ(N) mod ℓ being as hard as factoring:

1. **Algebraic:** Every algebraic operation over Z/NZ decomposes via CRT. No algebraic method can "unmix" the p and q components.

2. **Analytic:** All L-function and trace formula methods cost ≥ O(√N) for individual coefficient extraction (Round 3 conclusion).

3. **Geometric:** Elliptic curves over Z/NZ decompose: E(Z/NZ) = E(F_p) × E(F_q). Point counting ≡ factoring (Kunihiro-Koyama/Dieulefait-Urroz).

4. **Cryptographic:** 27 years of PHA without attack. Every protocol building on PHA remains secure. No research group has published even a partial break.

5. **Complexity-theoretic:** φ(N) mod ℓ sits in the same limbo as QR, DCR — problems that have resisted decades of separation attempts.

6. **Generic model:** In the generic ring model, everything is as hard as factoring. While this is an over-idealization, it confirms no "algebraic shortcut" exists.

7. **Experimental:** E1-E18 experiments confirm the barrier extends across Jacobi, carry, modexp, Frobenius, power residue, and all 111 tested features.

### 10.2 The Formal Gap

Despite this convergence, a formal proof that φ(N) mod ℓ is as hard as factoring would be a major result in computational complexity. The obstacles are the same that prevent proving P ≠ NP:
- **Relativization:** Baker-Gill-Solovay (1975) — there exist oracles relative to which P = NP and oracles where P ≠ NP
- **Natural proofs:** Razborov-Rudich (1997) — natural proof strategies cannot prove superpolynomial lower bounds under strong pseudorandom function assumptions
- **Algebrization:** Aaronson-Wigderson (2009) — algebraic techniques cannot resolve P vs NP

These barriers apply equally to proving factoring-related separations.

### 10.3 What a Breakthrough Would Look Like

If φ(N) mod ℓ COULD be computed in poly(log N):
- A single oracle call gives ~log₂(ℓ) bits
- ~76 calls (ℓ up to 383) would reach the RSA-2048 Coppersmith threshold
- This would yield polynomial-time factoring of RSA-2048
- Consequence: RSA, DSA, and all factoring-based cryptography would be broken

If φ(N) mod ℓ is PROVEN as hard as factoring:
- This would establish the first formal reduction from a partial information problem to full factoring
- It would resolve a 27-year-old open question (PHA independence)
- It would likely require techniques beyond current complexity theory

### 10.4 The Three Walls and What Would Breach Them

The problem is protected by three independent walls. An exploit requires breaching at least two simultaneously.

#### Wall 1: Evaluation — can't compute even one φ(N) mod ℓ

Every known method either CRT-decomposes or costs ≥ O(√N). Potential breaches:

**A. A "reciprocity law" for φ(N) mod ℓ.** The Jacobi symbol (a/N) = (a/p)·(a/q) is a CRT product — yet computable without factoring, via quadratic reciprocity. Reciprocity swaps the "hard" modulus p for the "easy" modulus a. No analogous swap is known for (p-1)(q-1) mod ℓ. No proof one can't exist.

**B. Exploiting the reducibility of ρ_{Δ,691}.** For most primes ℓ, the mod-ℓ Galois representation ρ_{Δ,ℓ} is irreducible with image ⊇ SL(2, F_ℓ). At ℓ = 691, it's **reducible**: ρ ≅ 1 ⊕ χ¹¹ (the Eisenstein congruence). Each 1-dimensional piece is a character computable in poly(log N). But extracting τ(N) mod 691 from these characters at composite N requires knowing how they combine — which is multiplicativity again.

**C. Non-multiplicative Kloosterman series (MOST PLAUSIBLE).** The Petersson/Rademacher formula:
```
τ(N) = C · Σ_{c≥1} S(1,N;c)/c · J₁₁(4π√N/c)
```
is NOT term-by-term multiplicative. Could a truncated version yield τ(N) mod 691 from poly(log N) terms? Currently costs O(√N) to O(N) terms. The "beyond endoscopy" program (Altug) aims at exactly this cancellation. **Why it probably fails:** Kloosterman sums S(1,N;c) for c coprime to N are CRT-separable; the non-CRT terms occur at c = p, q, N — requiring knowledge of p, q to isolate.

**D. Smooth conductor workaround.** Hiary's O(q^{1/3}) algorithm requires smooth modulus; squarefree N = pq gets no improvement. If someone found a covering/lifting trick embedding the squarefree-conductor problem into a smooth-conductor one, sub-√N evaluation would follow. No method known.

#### Wall 2: Coppersmith Threshold — n/4 bits needed

A single φ(N) mod 691 gives ~9.4 bits vs. the 512 bits needed for RSA-2048. Potential breaches:

**E. Lower the threshold.** The n/4 = β² comes from Coppersmith's lattice dimension analysis with β = 1/2 for balanced semiprimes. Improving this requires fundamentally new lattice techniques or a non-lattice approach to finding small modular roots. Ryan (EUROCRYPT 2025) confirmed n/4 is optimal within the current framework.

**F. Non-Coppersmith use of partial φ(N).** Could residues be used differently than CRT + Coppersmith? If φ(N) mod ℓ = 0, then ℓ | (p-1)(q-1), testable probabilistically (check g^{(N-1)/ℓ} ≡ 1 mod N). But this only handles the divisibility case. Knowledge of which primes divide p-1 could feed Pollard's p-1, but requires the residues to already be available.

#### Wall 3: CRT Decomposition — structural barrier

Every algebraic object over Z/NZ decomposes. Potential breaches:

**G. A non-local computation with amplifiable signal.** Carry propagation is non-CRT (E10-E12 proved this), but signals decay as N^{-0.35}. Could a specific non-local function amplify rather than bury the factor signal? E11 tested 111 features across 9 categories — all flat.

**H. Cross-structure interaction.** CRT decomposes each algebraic structure independently. Could the *interaction* between two structures (e.g., a homomorphism E(Z/NZ) → (Z/NZ)*) fail to decompose? Pairings are exactly this — but Galbraith-McKee proved pairing computation over Z/NZ implies factoring.

#### Assessment of Attack Scenarios

| Scenario | Wall Targeted | Plausibility | Obstruction |
|----------|--------------|-------------|-------------|
| A. Reciprocity law for φ mod ℓ | Wall 1 | Very low | No structural analogue of quadratic reciprocity |
| B. Reducible ρ_{Δ,691} | Wall 1 | Very low | Characters combine via multiplicativity |
| C. Kloosterman truncation | Wall 1 | Low (best hope) | Non-CRT terms at c=p,q require knowing factors |
| D. Smooth conductor lift | Wall 1 | Very low | No embedding known |
| E. Lower Coppersmith n/4 | Wall 2 | Very low | Optimal within lattice framework |
| F. Non-Coppersmith partial φ | Wall 2 | Low | All known uses reduce to CRT + lattice |
| G. Amplifiable non-local signal | Wall 3 | Very low | 111 features tested, all flat |
| H. Cross-structure pairing | Wall 3 | Very low | Galbraith-McKee: pairing mod N ⟹ factoring |

**The most plausible attack** would combine scenario C (Kloosterman series mod 691 with beyond-endoscopy cancellation) with scenario F (a non-Coppersmith use of the resulting partial information). But scenario C requires the Kloosterman terms at c coprime to N to cancel mod 691 faster than they cancel over Z — no evidence for this — and the non-CRT terms at c sharing factors with N are the very terms that encode the factorization.

### 10.5 The Correct Framing

The question "can we compute φ(N) mod ℓ?" is **not** a number theory question — it is a **complexity theory** question. All number-theoretic content has been extracted: the information exists (E13 proves 63 bits), the inversion is trivial (Newton's identity root-finding), and the evaluation barrier is structural (CRT + multiplicativity + dimensional). What remains is whether the evaluation barrier is provably hard, which is a question about computational models, not about number theory.

The three-wall analysis shows that even a partial breach of one wall is insufficient — an attacker must simultaneously find an efficient evaluation method AND accumulate enough bits to exploit. The convergence of barriers from algebra, analysis, geometry, cryptography, complexity theory, and experiment makes this exceedingly unlikely.

---

## 11. Complete Reference List

### Phi-Hiding Assumption
1. Cachin-Micali-Stadler (1999), EUROCRYPT — phi-hiding assumption, PIR protocol
2. Schridde-Freisleben (2008), ASIACRYPT — PHA validity analysis
3. Herrmann (2011), AFRICACRYPT — improved multi-prime PHA cryptanalysis
4. Xu-Hu-Sarkar-Zhang-Huang-Peng (2016), CT-RSA — multi-prime PHA cryptanalysis
5. Tosu-Kunihiro (2012), ACISP — optimal multi-prime PHA bounds
6. Kiltz-O'Neill-Smith (2010), CRYPTO — lossy RSA from PHA, RSA-OAEP instantiability
7. Kakvi-Kiltz (2012), EUROCRYPT — tight RSA-FDH security from PHA
8. Lewko-O'Neill-Smith (2013), EUROCRYPT — regularity of lossy RSA, PKCS security
9. Hemenway-Ostrovsky-Rosen (2015), TCC — non-committing encryption from PHA
10. Gentry-Ramzan (2005), ICALP — single-database PIR, constant rate
11. Gentry-MacKenzie-Ramzan (2005), CCS — PAKE from hidden smooth subgroups

### Coppersmith Method
12. Coppersmith (1996), EUROCRYPT — finding small roots, factoring with known bits
13. Coppersmith (2001), "Finding Small Solutions to Small Degree Polynomials"
14. Howgrave-Graham (1997) — lattice reformulation of Coppersmith
15. Herrmann-May (2008), ASIACRYPT — factoring given any bits
16. May-Nowakowski-Sarkar (2022), EUROCRYPT — approximate divisor multiples, 1/3 CRT
17. Ryan (2025), EUROCRYPT — automated multivariate Coppersmith
18. Feng-Nitaj-Pan (2024), CiC — small-e partial key exposure
19. Ajani-Bright (2024), ISSAC — SAT + lattice hybrid factoring
20. Ernst-Jochemsz-May-de Weger (2005), EUROCRYPT — full-size exponent PKE
21. Boneh-Durfee-Frankel (1998) — n/4 bits of d for small e

### Complexity Theory and Reductions
22. Miller (1976), J. Comp. Syst. Sci. — φ(N) → factoring, primality under ERH
23. Bach-Miller-Shallit (1986), SIAM J. Comput. — σ(N) ≡ factoring
24. Morain-Renault-Smith (2018/2023), AAECC — deterministic factoring with φ oracle
25. Hittmeir-Pomykala (2020), Fund. Inform. — deterministic factoring, multiple primes
26. Du-Volkovich (2021), FSTTCS — approximating ω(N) from φ oracle
27. Rabin (1979) — square roots mod N ≡ factoring
28. May (2004), CRYPTO — RSA key ≡ factoring (deterministic)
29. Aggarwal-Maurer (2009/2013), J. Cryptology — generic ring model, RSA ≡ factoring
30. Boneh-Venkatesan (1998), EUROCRYPT — algebraic reductions from factoring to RSA
31. Goldwasser-Micali (1984) — QR assumption, semantic security
32. Paillier (1999), EUROCRYPT — DCR assumption
33. Alexi-Chor-Goldreich-Schnorr (1988), SIAM J. Comput. — bit security of RSA
34. Hastad-Naslund (2004), JACM — all bits of RSA are hard
35. Manders-Adleman (1978), J. Comp. Syst. Sci. — NP-completeness of bounded QR
36. McCurley (1987/1994) — reductions among number-theoretic problems
37. Shoup (1997) — generic group model lower bounds
38. Brassard (1979) — relativized cryptography
39. Cai-Threlfall (2004) — QR/QNR in UP ∩ coUP

### Elliptic Curves over Composites
40. Kunihiro-Koyama (1998), EUROCRYPT — |E(Z/NZ)| ≡ factoring
41. Dieulefait-Urroz (2020), Mathematics — deterministic EC point count ≡ factoring
42. Galbraith-McKee (2005), ANTS — Tate pairing mod N implies factoring
43. Dabrowski-Pomykala-Shparlinski (2023), J. Complexity — EC oracle factoring
44. Lenstra (1987) — elliptic curve factoring method
45. Schoof (1985) — polynomial-time point counting on EC over F_p
46. Satoh (2000) — canonical lifting for point counting
47. Kedlaya (2001) — p-adic cohomological point counting

### Recent Partial Key Exposure
48. Jiang-Zhou-Liu (2024), Cybersecurity — PKE with exponent blinding
49. Jiang-Zhou-Liu (2024), TCS — non-consecutive blocks in prime power RSA
50. Zheng (2025), CT-RSA — common prime RSA partial key exposure
51. Zheng-Nitaj et al. (2026), CT-RSA — generalized RSA variant attacks
52. Zheng-Feng-Nitaj-Pan (2025), ACISP — continued fractions + Coppersmith
53. Zhao-Zhang-Cao et al. (2024), CiC — implicit factoring with shared any bits

### Recent Factoring and Quantum
54. Regev (2023), arXiv:2308.06572 — efficient quantum factoring
55. Ragavan-Vaikuntanathan (2024) — space-efficient quantum factoring
56. Litinski-Nickerson (2025), STOC — Jacobi factoring circuit
57. Gutmann-Neuhaus (2025), ePrint 2025/1237 — debunking quantum factoring records
58. Ryan (2025), EUROCRYPT — multivariate Coppersmith (n/4 unchanged)
59. Tesoro et al. (2024), arXiv:2410.16355 — tensor network factoring
60. Friedlander (2025), arXiv:2504.21168 — summation-based factoring
61. Mehta-Rana (2026), ePrint 2026/219 — phi(N) evaluation (likely flawed)
62. Bansimba et al. (2025), arXiv:2507.06706 — ML approximation of phi(N) (useless)

### Additional
63. Bach (1984), Berkeley TR — discrete log mod composite
64. Gasarch (2024) — blog survey of NP ∩ coNP problems
65. Calude-Jain-Khoussainov-Li-Stephan (2017), STOC — parity games quasi-polynomial
66. Lackenby (2021) — unknotting quasi-polynomial
67. Arpin (2022), arXiv:2203.03531 — level structure on supersingular isogeny graphs
68. ROCA (CCS 2017) — Coppersmith attack on Infineon TPM keys
69. Maurer (1995) — epsilon·n oracle bits suffice for factoring

---

## 12. Conclusions

### What This Round Establishes

1. **Every algebraic method for computing φ(N) mod ℓ is blocked** — group orders, Carmichael function, element orders, index calculus, Pohlig-Hellman, BSGS, ECM, pairings. All reduce to factoring or hit CRT decomposition.

2. **Every elliptic curve approach is blocked** — Schoof, SEA, Satoh, Kedlaya, ECPP all require prime modulus. Point counting ≡ factoring (Kunihiro-Koyama/Dieulefait-Urroz). Division polynomials mod N either reveal factors (ECM) or give CRT-separable output.

3. **The Coppersmith threshold is unchanged at n/4 since 1996.** No 2020-2026 paper improves it. Improvements are in automation, lattice efficiency, and special cases (CRT-exponents, non-consecutive bits).

4. **The PHA remains unbroken after 27 years** for standard RSA moduli. No new reductions, attacks, or advances since 2016.

5. **The complexity-theoretic gap is real** but almost certainly reflects hardness, not an overlooked algorithm. Seven independent lines of evidence converge on this conclusion.

### What Remains Unknown

- Is computing φ(N) mod ℓ provably as hard as factoring? (No — requires techniques beyond current complexity theory)
- Could there be a non-algebraic, non-analytic method? (Conceivable in principle, but no candidate exists)
- Is the PHA provably independent of factoring? (No — would require oracle separation)

### Relationship to Project

The E13 Eisenstein congruence channel demonstrates that **63 bits of factor information exist** and are **extractable in O(N) time**. This round confirms the O(N) barrier is structural: every known method for computing even a single bit of φ(N) without factoring N is blocked by CRT decomposition, the absence of Frobenius over Z/NZ, or the generic hardness of the multiplicative group order problem. The question of whether this barrier is provably absolute is equivalent to resolving fundamental open problems in computational complexity.

**The φ(N) mod ℓ gap is real, well-studied, and resistant to 27 years of cryptographic attack. It is not exploitable for factoring.**
