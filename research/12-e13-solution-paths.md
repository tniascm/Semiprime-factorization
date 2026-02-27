# E13 Solution Paths: Computing τ(N) mod 691 at Composite N

## The Problem

E13 established that 7 Eisenstein congruence channels yield 63 bits of factor
information per semiprime N = pq, with 100% accuracy. The pipeline:

```
τ(N) mod ℓ → s_{k-1} mod ℓ → (p mod ℓ, q mod ℓ) → CRT → Coppersmith → factor
```

Steps 2-5 are poly(log N). The bottleneck is step 1: computing τ(N) mod ℓ.
By Ramanujan's congruence, τ(N) ≡ σ₁₁(N) mod 691, so the problem reduces to:

> **Compute p¹¹ + q¹¹ mod 691 given only N = pq.**

This document surveys all known approaches, their status, and open questions.

---

## PATH 1: Petersson Trace Formula (Kloosterman Route)

### Idea
The Petersson formula avoids the divisor-sum term entirely:

```
|τ(N)|²/⟨Δ,Δ⟩ = 2πi⁻¹² Σ_{c≥1} S(1,N;c)/c · J₁₁(4π√N/c)
```

For level 1, weight 12, Δ is unique so this gives |τ(N)|² directly.

### Cost Analysis
- S(1,N;c) for each c costs O(c) (sum over (Z/cZ)*)
- Need C ~ √N terms for convergence
- Total: O(Σ_{c≤√N} c) = O(N)

### What Would Help
- A fast algorithm for Kloosterman sums S(m,n;c) at large c
- Accelerated convergence via smoothing (Kuznetsov-type)
- The Weil bound |S(m,n;p)| ≤ 2√p does NOT help compute the value
- S(m,n;c) is multiplicative in c: for c = c₁c₂, S factors via CRT —
  requires factoring c, not N. When c is prime, evaluation is O(c).

### Recent Work
- Booker-Lee-Strömbergsson (arXiv:2101.05663, 2021): twist-minimal trace
  formulas reduce spectral sums but not Kloosterman counts
- Altug (arXiv:2505.18967, May 2025): beyond endoscopy with ramification;
  Poisson summation cancels Eisenstein terms asymptotically, not individually

### Status: OPEN but convergence gap appears fundamental

---

## PATH 2: Edixhoven-Couveignes at Composite Hecke Index

### Idea
EC algorithm computes τ(p) mod ℓ in poly(ℓ, log p) for PRIME p by finding
tr(Frob_p) on J₀(1)[ℓ]-torsion. Mascot (arXiv:2004.14683, 2022) achieves
20-100x practical speedup via p-adic methods, computing τ(p) for p > 10^1000.

For composite N, use the Hecke correspondence T_N directly (defined
geometrically as cyclic N-isogenies) without decomposing T_N = T_p · T_q.

### Obstruction
- Frobenius is defined only at primes; no "Frob_N" for composite N
- The degree of T_N on X₀(1) is σ₁(N) = 1+p+q+pq, encoding divisors
- tr(T_N | J₀(1)[ℓ]) = τ(p)·τ(q) mod ℓ regardless of evaluation method
- Counting cyclic N-isogenies from E amounts to computing σ₁(N) locally

### What Would Help
- A base field/ring where the CRT splitting E(Z/NZ) ≅ E(F_p)×E(F_q)
  doesn't apply (would need N inert, but N is composite)
- A non-Frobenius endomorphism of J₀(1)[ℓ] with trace related to τ(N)
- Peter Bruin (PhD thesis, Leiden 2010): extended EC to squarefree levels,
  still prime index only
- Peng Tian (arXiv:1905.10036, 2022): non-Eisenstein primes, prime index

### Status: STRUCTURALLY BLOCKED by Frobenius definition

---

## PATH 3: Mod-ℓ Galois Representations Directly

### Idea
Work entirely in GL₂(F₆₉₁). The mod-691 representation
ρ_{Δ,691}: Gal(Q̄/Q) → GL₂(F₆₉₁) has tr(ρ(Frob_p)) = τ(p) mod 691.

For N = pq: τ(N) mod 691 = tr(ρ(Frob_p))·tr(ρ(Frob_q)).

### Key Question
Can tr(A)·tr(B) be determined from the product AB in GL₂(F₆₉₁)?

From AB: tr(AB), det(AB) = det(A)det(B). But tr(A)·tr(B) ≠ tr(AB) generally.
The tensor product A⊗B ∈ GL₄ has tr(A⊗B) = tr(A)·tr(B), but evaluating
the 4-dimensional representation at "Frob_N" still requires Frob_p, Frob_q.

### τ(n) mod small primes — what's computable?

| ℓ | τ(n) mod ℓ determined by | Computable at composite N? |
|---|---|---|
| 2 | n is odd perfect square ⟺ τ(n) odd (Serre) | YES — but τ(pq) always even (zero info) |
| 3 | Mod-3 Galois rep, splitting field explicit | NO — Frobenius at composites undefined |
| 5 | Mod-5 Galois rep (Serre, Swinnerton-Dyer) | NO — same obstruction |
| 7 | Mod-7 Galois rep | NO |
| 23 | Mod-23 Galois rep (23 | τ(n) iff χ₂₃(n)=0 or (n/23)=QR) | PARTIAL — Lehmer: τ(n)≡0 mod 23 iff n∈{0,1,2,3,4,6,8,9,12,13,16,18} mod 23. This IS poly(log N)! But gives only {0 vs nonzero}, not the full residue. |
| 691 | σ₁₁(n) mod 691 (Ramanujan congruence) | NO — requires factoring |

### The ℓ = 23 Opening
Lehmer's result: τ(n) ≡ 0 mod 23 is determined by n mod 23. For n = pq:
this gives n mod 23 = (p mod 23)(q mod 23), a quadratic residuosity test.
But this is just the Jacobi symbol — known CRT mechanism, zero factor info.

### Status: BLOCKED for all ℓ that carry factor information

---

## PATH 4: Voronoi Summation

### Idea
Voronoi summation transforms Σ_{n≤X} τ(n)e(nα) into a dual sum:

```
Σ τ(n)e(na/q) → (q-dependent transform) → Σ τ(n) · (integral kernel)(n/q²)
```

For α = a/N with q = N, the dual involves S(ā,n;N) terms.

### Cost
- The dual sum converges in O(N/q²) = O(1/N) terms (!) when q = N
- BUT the Kloosterman sum S(ā,n;N) costs O(N) per evaluation
- Net: O(1) · O(N) = O(N), no improvement

### What Would Help
- Voronoi summation is most powerful when q is SMALL (test function supported
  on short intervals). For q = N, the smoothing is too extreme.
- Using q ≪ N: you get Σ τ(n)e(na/q) for small q, which is an average
  over many n — not the individual τ(N).

### Status: BLOCKED — wrong regime for large composite q

---

## PATH 5: Amplification / Spectral Isolation

### Idea
The Duke-Friedlander-Iwaniec amplification method boosts one eigenform's
contribution in a spectral average. Could it isolate τ(N) from:

```
Σ_f |a_f(N)|²/⟨f,f⟩ = Petersson terms
```

### Obstruction
Amplification works by choosing test vectors that correlate with the target
form. For level 1 weight 12, there's only ONE eigenform (Δ), so
amplification is vacuous — there's nothing to amplify against.

For higher weight/level (multiple eigenforms), amplification gives
subconvexity bounds on individual coefficients but NOT exact values.
Amplification reduces error terms from O(N^{1/2+ε}) to O(N^{1/2-δ+ε})
for some δ > 0, but does not compute individual a_f(N).

### Deeper Analysis: Amplification at Level N = pq

Even at level N = pq, where dim S₁₂(Γ₀(N)) ≈ N/6 and amplification is
non-trivial, the method still fails because amplification is structurally
an inequality-producing technique. The DFI amplifier forms:

  S = Σ_f |Σ_l x_l λ_f(l)|² · |a_f(N)|² / ⟨f,f⟩

The step of discarding non-target spectral terms (which are non-negative)
is an irreversible information loss. Recovering exact τ(N) would require
subtracting ALL other contributions — at least as hard as the original problem.

**Spectral reciprocity (Blomer-Khan, Duke Math J. 2019):** Relates moments
at level q twisted by λ_f(l) to moments at level l twisted by λ_g(q).
For N = pq: the dual side involves λ_g(N) = λ_g(p)·λ_g(q) by Hecke
multiplicativity — the factorization is encoded in the dual problem.

**Nelson's orbit method (2021):** Replaces classical amplification with
microlocal analysis for Lie group representations. Still produces
subconvexity BOUNDS, not exact values: L(1/2, π) ≪ C(π)^{1/4-δ}.

**Exact formula (Petersson):** The unique viable exact formula is the
Petersson trace formula (dim S₁₂ = 1): τ(N) = C · Σ_{c≥1} S(1,N;c)/c · J₁₁(4π√N/c).
This converges and gives exact τ(N) but costs O(√N) to O(N).

### Status: BLOCKED — amplification gives bounds not values; exact formulas cost O(√N)

---

## PATH 6: Approximate / Partial Information

### Idea
Bach-Charles blocks EXACT τ(N) computation. What about approximate?

### Analysis
- |τ(N)| ≤ 4N^{11/2} ≈ 2^{5.5n} for n-bit N
- τ(N) mod 691 requires ~10 bits from the bottom
- Need essentially all 5.5n bits — approximation doesn't help
- Sign of τ(N): sign(τ(pq)) = sign(τ(p))·sign(τ(q)), requires knowing
  individual signs, not computable without factors
- Sato-Tate: statistical distribution of τ(p)/2p^{11/2}, does not help
  for individual composites

### Status: BLOCKED — mod-ℓ reduction requires near-exact value

---

## PATH 7: The φ(N) mod 691 Gap

### Idea
σ₁₁(N) mod 691 ← p¹¹+q¹¹ mod 691 ← (p+q) mod 691 ← φ(N) mod 691

Since φ(pq) = (p-1)(q-1) = N - p - q + 1, knowing φ(N) mod 691 gives
(p+q) mod 691, which gives {p mod 691, q mod 691} (quadratic formula).

### Complexity Status

**Computing φ(N) fully is equivalent to factoring.** But computing φ(N) mod 691?

- NO algorithm known to compute φ(N) mod 691 in poly(log N)
- NO proof that it's as hard as factoring (the standard reduction uses
  the FULL value of φ(N))
- This is a genuine complexity-theoretic gap

### The Pollard p-1 Test (Partial)
Compute gcd(a^{690}-1, N):
- Result = 1: neither p≡1 nor q≡1 mod 691 (~99.7% of semiprimes)
- Result = p or q: exactly one factor ≡1 mod 691 (factors N directly, ~0.3%)
- Result = N: both ≡1 mod 691 (negligible)

This probes ONE residue class (p≡1 mod 691) via Fermat's little theorem.
No algebraic identity exists for testing p≡r mod 691 for arbitrary r.

### Deeper Analysis: The Coppersmith Connection

**Coppersmith's theorem (1996):** Given p mod M where M > N^{1/4}, one can
factor N in poly(log N) time. Since (p+q) mod M gives p mod M via quadratic
formula mod M, knowing φ(N) mod M where M > N^{1/4} suffices to factor.

**CRT accumulation:** If we had oracles for φ(N) mod ℓ for ALL primes ℓ up
to B, CRT gives φ(N) mod M where M = primorial(B) ~ e^B. Need M > N^{1/4},
so B > (n/4)·ln(2) ≈ 0.173n. For RSA-2048: B > 355.

**Single oracle φ(N) mod 691:** Gives only ~9.4 bits of p+q. The Coppersmith
threshold for RSA-2048 is ~512 bits. A single congruence is far below threshold.

### The Phi-Hiding Assumption (Cachin-Micali-Stadler 1999)

The cryptographic assumption "given N = PQ, deciding whether ℓ | φ(N) is
hard" is adopted as an INDEPENDENT assumption, NOT proven equivalent to
factoring. Key facts:
- Factoring hard ⟹ phi-hiding plausible (but NOT proven)
- Phi-hiding hard does NOT imply factoring hard
- No reduction in either direction is known
- Schridde-Freisleben (ASIACRYPT 2008): fails for N = PQ^{2e}, holds for
  standard RSA moduli

### The Complexity Gap (Formalized)

The problem "compute φ(N) mod 691 given N = pq" sits in a genuine gap:

**Lower bound:** No poly(log N) algorithm known. All approaches (σ₁₁(N)
evaluation, q-expansion, trace formula) require O(N) or O(√N) work.

**Upper bound:** No proof of hardness. Miller's reduction uses FULL φ(N).
The phi-hiding assumption is adopted precisely because no factoring
reduction exists.

**The problem is in NP ∩ coNP** (like factoring) but NOT known to be in P
or known to be equivalent to factoring. This is analogous to the Decisional
Composite Residuosity Assumption (Paillier): believed hard, used in crypto,
not proven equivalent to factoring.

### Status: OPEN — genuine complexity-theoretic gap, no proof either way

---

## PATH 8: Algebraic Geometry / Motives

### Scholl's Motive M(Δ)
The motive attached to Δ lives in H¹¹ of the 10-fold Kuga-Sato variety.
Its ℓ-adic realization gives the Galois representation ρ_{Δ,ℓ}. The
crystalline realization at prime p gives τ(p) via crystalline Frobenius.

For composite N: no "crystalline Frobenius" at N = pq. The Kuga-Sato
variety over Z/NZ decomposes via CRT.

### Elliptic Curves over Z/NZ
E(Z/NZ) ≅ E(F_p) × E(F_q) by CRT. Every geometric invariant factors.
Even |E(Z/NZ)| = |E(F_p)|·|E(F_q)| = (p+1-a_p)(q+1-a_q), one equation
in two unknowns. Multiple curves give multiple equations but solving them
is equivalent to factoring (ECM family).

### Isogeny Graphs
The ℓ-isogeny graph over F_p (supersingular curves) is a Ramanujan graph.
Over Z/NZ it decomposes as product of graphs over F_p and F_q. Charles-
Goren-Lauter hash function security relies on this not being efficiently
computable without factoring.

### Overconvergent Cohomology over Rings
Berthelot's rigid cohomology requires a base field. Z/NZ is not a field.
No meaningful generalization exists. Pollack-Stevens overconvergent modular
symbols work p-adically (prime base).

### Status: STRUCTURALLY BLOCKED by CRT at every level

---

## PATH 9: Shifted Convolution Sums

### Idea
Σ_{n≤X} τ(n)τ(n+N) relates to the spectral decomposition of the shifted
convolution problem. For the shift h = N, this carries information about N.

### Analysis
The shifted sum Σ τ(n)τ(n+h) is controlled by the spectral theory of
Maass forms and involves the Rankin-Selberg L-function L(s, Δ×Δ̃). The
main term is a polynomial in h, and the error depends on eigenvalues of
the Laplacian on X₀(1).

For individual h = N, the sum does not isolate τ(N). It is an average
over all n of τ(n)τ(n+N), not τ(N) itself. The sum length X must be
comparable to N for the main term to dominate.

### Deeper Analysis (Blomer-Harcos, Holowinsky, Motohashi)

**Spectral decomposition (Blomer-Harcos, Duke Math J. 2008):** The shifted
convolution Dirichlet series L_h(s,f,f) has meromorphic continuation with
poles at s = 1/2 ± ir_j (Maass eigenvalues). The main term involves Ramanujan
sums c_d(h) and divisor sums over h — computing these for h = N requires
factoring N.

**Holowinsky's sieve (Duke Math J. 2009):** Gives UPPER BOUNDS on averaged
shifted convolution sums. The saving of a power of log suffices for QUE but
contains zero pointwise information about individual τ(N).

**Circle method at h = N:** The major arc at a/q with q | N would involve the
local behavior of Σ τ(n)e(2πina/p). But: (a) Δ is a cusp form (vanishing at
cusps suppresses contributions exponentially), (b) we don't know p to isolate
this arc, (c) the integral over ALL arcs does not separate factor-dependent parts.

**Motohashi formula:** Relates fourth moments of ζ to spectral cubic moments.
Both sides are averages; neither isolates individual Fourier coefficients.

**Petersson/Rademacher series:** The ONLY individual-value formula for τ(N)
via shifted sums is the convergent Kloosterman series: costs O(N). The Niebur
convolution formula τ(n) = n⁴σ(n) - 24Σ i²(35i²-52in+18n²)σ(i)σ(n-i) also
costs O(N) and requires σ(N) which itself needs factoring.

### Status: BLOCKED — averages not individuals; factor-dependent terms require factoring

---

## PATH 10: Quantum Approaches

### Shor's Algorithm
Factors N in poly(log N) quantum time. Once p,q known, τ(N) is trivial.
But this doesn't help classically.

### Quantum Class Numbers
Hallgren (2005): class group in quantum poly(log D). Applied to trace
formula Term A: cost becomes O(√N · poly(log N)) quantumly. Still need
Term B (divisor sum), which quantum Shor solves by factoring first.
Net: quantum already has Shor, trace formula adds nothing.

### Open Question
Is there a quantum algorithm that computes σ_k(N) or τ(N) without
implicitly factoring? No such algorithm is known. Shor's algorithm
finds the period of a^x mod N, which gives φ(N), which gives factors.
A direct quantum algorithm for σ₁₁(N) mod 691 would be genuinely new.

### Status: REDUNDANT (Shor already solves everything)

---

## PATH 11: Hardware Acceleration of O(√N) Algorithm

### Denis Charles Result
τ(n) computable in O(n^{1/2+ε}) under GRH using trace formula + BSGS.

### Practical Reach with Hardware
| Bit size | √N | Time at 10¹⁵ ops/sec | GNFS comparison |
|----------|-----|---------------------|-----------------|
| 64 | 2³² | milliseconds | trivial |
| 128 | 2⁶⁴ | ~500 years | minutes by GNFS |
| 256 | 2¹²⁸ | impossible | days by GNFS |
| 512 | 2²⁵⁶ | impossible | months by GNFS |

### Status: BOUNDED — exponential gap for cryptographic sizes

---

## PATH 12: Waldspurger / Period Integrals (EXPANDED)

### 12A. Waldspurger's Theorem — Precise Statement

**Setup.** For f ∈ S_{2k}(Γ₀(M)) a newform, the Shimura correspondence
gives a half-integral weight form g ∈ S_{k+1/2}^+(Γ₀(4M)) (in the
Kohnen plus-space) whose Hecke eigenvalues match those of f.

**Kohnen-Zagier formula (1981, Inventiones Math. 64, 175-198):**
For f of level 1 and weight 2k, and g its Shimura correspondent:

  |c(|D|)|² = L(k, f, χ_D) · (k-1)!/(π^k) · ⟨g,g⟩/⟨f,f⟩

where c(n) are Fourier coefficients of g, D is a fundamental discriminant
with (-1)^k D > 0, and L(s, f, χ_D) is the twisted L-function.

**For Δ specifically (f = Δ, weight 2k = 12, so k = 6):**
- The Shimura correspondent g ∈ S_{13/2}^+(Γ₀(4))
- dim S_{12}(SL₂(Z)) = 1, so by Kohnen's isomorphism, dim S_{13/2}^+ = 1
- g is the UNIQUE normalized form in S_{13/2}^+(Γ₀(4))
- The formula becomes: |c(|D|)|² = C · L(6, Δ, χ_D)
  where C = 5!/π⁶ · ⟨g,g⟩/⟨Δ,Δ⟩ is a fixed positive constant

**Critical observation:** The Kohnen-Zagier formula connects c(|D|)² to
L-VALUES, not to τ(|D|). Waldspurger tells us about L(1/2, Δ⊗χ_D),
which is a TWIST of the L-function of Δ by the quadratic character χ_D.
This is NOT the same as τ(D).

### 12B. What c(N) Does and Does NOT Tell You About τ(N)

**The mismatch:** Waldspurger's formula applies ONLY at fundamental
discriminants D (squarefree integers with D ≡ 0,1 mod 4). For N = pq
a semiprime:
- If p,q > 2 and p ≡ q ≡ 3 mod 4: then -N is a fundamental discriminant
- Otherwise: N or -N may not be fundamental

For non-fundamental discriminants D = n²D₀, the coefficient c(D) involves
more complicated expressions (Baruch-Mao 2007 generalize Kohnen-Zagier
to remove the fundamental discriminant condition, relating c(D)² to
L-values with different half-integral weight forms).

**The key formula chain:**
```
c(|D|)² → L(6, Δ, χ_D)     [Kohnen-Zagier]
L(6, Δ, χ_D) = Σ τ(n)χ_D(n)/n⁶  [Dirichlet series definition]
```

This L-value is a SUM over τ(n) weighted by the character χ_D(n) = (D/n),
NOT the individual coefficient τ(N). Even if we could compute c(N)²,
it would tell us about L(6, Δ, χ_N), not about τ(N).

**No path from c(N) to τ(N):**
- c(N) encodes central L-value information
- τ(N) is a single Fourier coefficient of the weight-12 form
- These are fundamentally different objects
- The Shimura correspondence is between SPACES, not between
  individual coefficients at the same index

### 12C. Computing L(6, Δ, χ_D) — Cost Analysis

**Approximate functional equation:** L(6, Δ, χ_D) can be computed using:

  L(6, Δ, χ_D) = Σ_{n ≤ X} τ(n)χ_D(n)/n⁶ · V(n/√|D|) + (epsilon factor)·(dual sum)

where X ~ √|D| is the critical length and V is a smooth cutoff.

**Cost:** O(|D|^{1/2+ε}) arithmetic operations, each involving:
- τ(n) for n ≤ √|D|: these are for SMALL n, computable by q-expansion
- χ_D(n) = Kronecker symbol: O(log n) per evaluation

For D = N = pq (n-bit), this is O(N^{1/2+ε}) = O(2^{n/2}) — sub-exponential
but NOT polynomial in log N. The conductor of χ_N is N, forcing the
critical line computation to use √N terms.

**No shortcut via Waldspurger:** Even if c(|D|) could somehow be computed
faster (via theta lifts, see 12E below), this gives L(6, Δ, χ_D), not τ(N).

### 12D. The Shintani Lift — Explicit Construction

**Shintani's map (inverse Shimura):** Takes f ∈ S_{2k}(Γ₀(M)) and produces
g ∈ S_{k+1/2}(Γ₀(4M)) via cycle integrals:

  c(D) = Σ_{Q ∈ Cl(D)} ∫_{γ_Q} f(z) · Q(z,1)^{k-1} dz

where the sum is over SL₂(Z)-classes of binary quadratic forms of
discriminant D, and γ_Q is the geodesic in H/SL₂(Z) associated to Q.

**For Δ (k=6, weight 12):**

  c(D) = Σ_{Q ∈ Cl(D)} ∫_{γ_Q} Δ(z) · Q(z,1)⁵ dz

**Cost analysis:**
- Number of classes h(D): O(√|D|) on average (can be larger)
- Each cycle integral requires numerical integration of Δ along a geodesic
- For D = N = pq: h(-N) ≈ √N on average
- Total cost: O(√N · (integration cost per class))

This is again O(N^{1/2+ε}), not poly(log N). The class number h(D) = O(√D)
is the same barrier seen in the trace formula approach.

### 12E. Theta Lifts and See-Saw Duality

**Theta correspondence framework:** The Shimura/Shintani lifts can be
realized as theta lifts for the dual pair (SL₂, PB×) where B is a
quaternion algebra:

  θ(f)(g') = ∫_{SL₂(Z)\H} f(z) · Θ(z, g') dz

where Θ is the Weil representation theta kernel.

**Computational picture:**
- The theta kernel Θ(z,g') involves ternary quadratic forms from the
  quaternion algebra B
- Pacetti-Tornaría (Exp. Math. 2008): for composite level N, construct
  weight 3/2 forms using Brandt matrices of quaternion orders
- Brandt matrix construction requires knowing the IDEAL CLASS structure
  of orders in B ramified at primes dividing N
- For N = pq: must consider B ramified at {p,∞} or {q,∞} or {p,q}
- Computing Brandt matrices at composite level: O(N) (dimension barrier)

**See-saw identity:**
The see-saw (GL₂ × GL₁ inside GL₂ × GL₂ containing GL₂ × SO(3))
gives period-integral identities like:

  ∫ f·θ_χ = ∫ (θ-lift of f) · χ

These reduce Waldspurger periods to theta lift evaluations, but do NOT
reduce the computational cost below O(√N).

**Inam-Wiese (Rocky Mountain J. Math. 2022, arXiv:2010.11239):**
Fast bases for S_{k+1/2}(Γ₀(4)) using Rankin-Cohen operators, enabling
q-expansion computation of half-integral weight forms. However:
- Computing c(N) still requires N terms of the q-expansion
- For N = pq, this is O(N) = exponential in bit-length
- No known speedup for extracting a single coefficient c(N)

### 12F. Gross-Zagier Formula and Higher Weight Generalizations

**Classical Gross-Zagier (weight 2, elliptic curves):**
For E/Q of conductor N and K = Q(√D) imaginary quadratic:

  L'(1, E/K) = c · ĥ(P_K)

where P_K is a Heegner point and ĥ is the Néron-Tate height. This relates
the DERIVATIVE of the L-function to arithmetic heights.

**Higher weight generalization (S.-W. Zhang, 1997-2001):**
For f ∈ S_{2k}(Γ₀(N)) of even weight 2k ≥ 4:

  L'(f/K, k) = c · ⟨z_K, z_K⟩_{BB}

where z_K is a Heegner cycle on the (2k-2)-fold Kuga-Sato variety over
the modular curve X₀(N), and ⟨·,·⟩_{BB} is the Beilinson-Bloch height
pairing (Gillet-Soulé arithmetic intersection theory).

**For Δ (weight 12, level 1):**
- The Kuga-Sato variety is the 10-fold fiber product of the universal
  elliptic curve over X₀(1)
- Heegner cycles are CM cycles in H¹⁰ of this 10-fold
- Zhang's formula gives: L'(Δ/K, 6) = c · ⟨z_K, z_K⟩_{BB}
- This relates the DERIVATIVE L'(Δ/K, 6) to arithmetic geometry

**Obstruction for factoring:**
1. The formula involves L'(f/K, k), not L(f/K, k) or τ(N)
2. Computing Heegner cycles: requires CM points, class number O(√D)
3. The Kuga-Sato variety over Z/NZ decomposes via CRT (same obstruction)
4. Height computations require local contributions at all primes dividing N
5. Nekovár (p-adic heights): works p-adically, requires prime base

**Gross-Kohnen-Zagier theorem:**
The generating series Σ_D [y_D] · q^|D| (with Heegner divisors y_D in
the Jacobian) is a modular form of weight 3/2. This is a MODULARITY
result — it says the generating series transforms correctly — but
evaluating individual terms requires computing Heegner divisors at
cost O(h(D)) = O(√D) per discriminant.

### 12G. Arithmetic Intersection Theory (Kudla Program)

**Kudla's program:** Generating series of special cycles on arithmetic
Shimura varieties are Siegel modular forms. Specifically:

  Σ_T Z(T) · q^T ∈ Arithmetic Chow group valued Siegel modular form

where Z(T) are Kudla-Rapoport cycles on unitary/orthogonal Shimura
varieties, indexed by positive semi-definite matrices T.

**For our problem:**
- The arithmetic Siegel-Weil formula: ⟨Z(T), Z(T')⟩_arith = (derivative of local densities)
- Computing intersection numbers: requires evaluating representation
  densities of hermitian forms at all primes
- At primes p,q dividing N: local density computation requires knowing
  p and q individually (the local factor at p depends on reduction type
  of the Shimura variety mod p)

**Cost:** Local densities at prime l: O(poly(log l)). But at composite N:
need densities at p and q SEPARATELY — i.e., requires factoring N.

**Kudla-Rapoport conjecture (proved by Li-Zhang, 2022):**
Global arithmetic intersection = product of local densities × derivative
of L-function. This factorization through local data is PRECISELY the
obstruction: computing the local factor at p requires knowing p.

### 12H. Ichino-Ikeda Formula and Triple Products

**Ichino's formula (2008):** For automorphic forms π₁, π₂, π₃ on GL₂:

  |∫ φ₁φ₂φ₃|² / (||φ₁||²||φ₂||²||φ₃||²) = C · L(1/2, π₁×π₂×π₃) / (∏_v I_v)

where I_v are LOCAL period integrals and C is an explicit constant.

**For extracting τ(N):** If we take π₁ = π₂ = π₃ = πΔ (representation
attached to Δ), the triple product L-function L(s, Δ×Δ×Δ) has Euler
product with local factor at prime p involving τ(p).

At composite N = pq: the global period integral factors as product of
local integrals over all primes. The local integral at p depends on the
local representation π_{Δ,p}, and at q on π_{Δ,q}. Computing these local
factors individually requires knowing p and q.

### 12I. Recent Work (2019-2025)

**Hu-Nelson (arXiv:1810.11564, 2019):** New test vectors for Waldspurger's
period integral using minimal vectors (not newforms). Establishes:
- Local integral size ~ q^{-l/4} where l = conductor exponent
- Combined with relative trace formula for HYBRID subconvexity bounds
- Achievement: bounds as strong as Weyl bound in proper parameter range
- Relevance to factoring: NONE — gives bounds on L-values, not exact values

**Hu-Yin (arXiv:1907.11428, 2019):** Waldspurger's period integral for
newforms in new ramification cases. Explicit local integrals.

**Cai-Shu-Tian (Algebra & Number Theory, 2014):** Explicit Gross-Zagier
and Waldspurger formulae — makes all local factors computable.

**Key insight from test vector theory:** The global Waldspurger period
factors as:
  P_global = L(1/2, Π⊗χ) × ∏_v P_v(local test vectors)

This product decomposition over all places v is PRECISELY the CRT
obstruction: each local factor P_v at v = p or v = q encodes local
representation data that depends on knowing p and q individually.

### 12J. The Saito-Kurokawa Lift (Indirect Connection)

The Saito-Kurokawa lift σ_k: S_{2k-2}(SL₂(Z)) → S_k(Sp₄(Z)) factors as:

  S_{2k-2} --Shintani-→ S_{k-1/2}^+ --→ J_{k,1} --Maass-→ S_k(Sp₄(Z))

For Δ (weight 12), the lift goes: S₁₂ → S_{13/2}^+ → J₇,₁ → S₇(Sp₄(Z)).
The Fourier-Jacobi coefficients of the Siegel modular form in the Maass
Spezialschar encode the same information as the half-integral weight
Fourier coefficients c(D).

This adds no computational advantage: the bottleneck remains computing
c(N) or equivalently the Siegel form's coefficient at a composite index.

### 12K. Summary: All Period Integral Approaches

| Approach | What it computes | Cost | Connection to τ(N) | Obstruction |
|----------|-----------------|------|-------------------|-------------|
| Waldspurger/Kohnen-Zagier | c(D)² ~ L(6,Δ,χ_D) | O(√N) | NONE (different object) | AFE needs √N terms |
| Shintani cycle integrals | c(D) via geodesic integrals | O(√N) | Indirect via L-value | h(D) = O(√D) classes |
| Gross-Zagier (higher wt) | L'(Δ/K, 6) via Heegner heights | O(√N) | NONE (derivative, not value) | CM point enumeration |
| Theta lifts | Half-integral weight forms | O(N) | Same as Waldspurger | q-expansion cost |
| Kudla intersections | Arithmetic intersection numbers | Needs factoring | Local factors at p,q | CRT decomposition |
| Ichino triple product | L(1/2, Δ×Δ×Δ) via periods | Needs factoring | Local factors | CRT decomposition |
| Baruch-Mao generalized | c(D)² for non-fundamental D | O(√N) | Same as Waldspurger | Same cost |
| Hu-Nelson test vectors | Subconvexity bounds | O(N^{1/2-δ}) | Bounds only, not exact | Still sub-exponential |

**Universal obstruction:** Every period integral formula has two features
that block poly(log N) computation of τ(N):

1. **The formula connects L-values to periods, not τ(N) to periods.**
   Waldspurger gives c(D)² ~ L(1/2, f⊗χ_D), and L(1/2, f⊗χ_D) is a
   sum over τ(n)·χ_D(n)/n^s — not τ(D) alone.

2. **Every known evaluation method costs ≥ O(√N):**
   - Approximate functional equation: √N terms from the conductor
   - Cycle integrals: h(D) = O(√D) geodesic integrals
   - Theta series: O(N) q-expansion terms
   - Local period integrals: require factoring N to evaluate at p, q

3. **Bach-Charles blocks the reverse direction too:** Even if you could
   compute L(1/2, Δ⊗χ_N) in poly(log N), extracting τ(N) from the
   L-value would require inverting the Dirichlet series — equivalent to
   knowing the Euler product, which requires factoring.

### Status: BLOCKED — O(√N) minimum cost, no τ(N) connection, CRT obstruction on local factors

---

## PATH 13: Character Sums and Isogeny Graphs at Composite Moduli

### Character Sums

All character sums over (Z/NZ)* decompose via CRT for N = pq:
- **Gauss sums:** G(χ,N) = G(χ_p,p)·G(χ_q,q). |G| = √N for primitive χ (no info).
- **Jacobi sums:** J(χ₁,χ₂,N) = J(p)·J(q). APRCL restricts divisors to t classes
  mod auxiliary s, but t too large for random semiprimes.
- **Kloosterman sums:** S(a,b;N) = S(a,b;p)·S(a,b;q). Cost O(N) to compute.
- **Incomplete sums** Σ_{x≤X} e(f(x)/N): Do NOT factor via CRT (carry structure),
  but peaks at noise level (N^{-0.35} to N^{-0.47}, confirmed by E10/E12).

**L(1/2, χ_N) is computable in O(√N) without factoring** (Jacobi symbol via
Euclidean algorithm). Value encodes class number h(-4N), but extraction requires
≥ O(N^{1/4}) (SQUFOF). Not poly(log N).

**The quadratic residuosity problem is the irreducible hard core:** Given (n/N) = +1,
distinguishing QR/QR from QNR/QNR is generically equivalent to factoring.

### Isogeny Graphs

- **Over Z/NZ:** E(Z/NZ) ≅ E(F_p) × E(F_q) by CRT. Isogeny graph = tensor product.
- **Castryck-Decru (2022):** Broke SIDH using auxiliary torsion, but requires protocol-
  specific information leak with no factoring analogue (no torsion auxiliary from N).
- **Charles-Goren-Lauter:** EndRing ↔ IsogenyPath reductions. Factoring is an INPUT
  that helps solve isogeny problems, NOT an output.
- **Eichler orders at level N = pq:** Mass formula = (D-1)/12·(p-1)(q-1) = product
  of local factors. Computing mass gives φ(N), but already requires factoring.

### Status: BLOCKED — complete CRT decomposition; QR problem is hard core

---

## PATH 14: Rankin-Selberg L-function L(s, Δ×Δ)

### Idea
L(s, Δ×Δ) = Σ |τ(n)|²/n^s has N-th coefficient |τ(N)|². Extract it
via Perron's formula or inverse Mellin transform.

### The Point-Evaluation / Coefficient-Extraction Duality
There is a fundamental duality:
- **Evaluating L(s₀)**: Compute a weighted sum of known coefficients at
  a point. Cost: O(√(analytic conductor)) via approximate functional equation.
- **Extracting the N-th coefficient**: Recover a single term from L(s).
  Cost: O(N^{1/2+ε}) at minimum via Perron/inverse Mellin.

These are dual problems. The approximate functional equation is optimized
for point evaluation, not coefficient extraction. For coefficient extraction,
the contour integral over a non-compact path fundamentally requires O(√N)
quadrature points.

### Hiary's Algorithm and Limitations
- Hiary (Annals 2011): computes ζ(1/2+iT) in O(T^{1/3+ε}), improving √T
- Uses efficient quadratic exponential sums (van der Corput B-process)
- Extension to Dirichlet L-functions (arXiv:1205.4687): O(q^{1/3+ε}) when
  q is smooth (perfect power or highly composite)
- **For squarefree N = pq: NO improvement over O(√N)** — the Postnikov
  character formula reduction requires smooth modulus
- Vishe (arXiv:1202.6303): L(1/2, f⊗χ_q) in O(q^{5/6}) for smooth q
- For degree-4 L(s, Δ×Δ): no sub-√T improvement known

### Odlyzko-Schonhage Amortization
Evaluates a Dirichlet series at O(N) equally spaced points in O(N^{1+ε})
total — amortized O(N^ε) per value. But for a SINGLE value, still O(N^{1/2+ε}).
No help for isolated τ(N) computation.

### Character Twists
L(s, Δ⊗χ_N): the N-th coefficient is χ_N(N)·τ(N) = 0 (since gcd(N,N)=N).
Twisting annihilates the very coefficient we want. Recovering τ(N) from a
family of twists requires inverting an O(N)-dimensional linear system.

### Status: BLOCKED — coefficient extraction costs O(N^{1/2}), Hiary inapplicable to squarefree conductor

---

## SYNTHESIS: RANKING OF PATHS

### Genuinely Open (no proof of impossibility)

1. **PATH 7: φ(N) mod 691** — The hardest gap. Computing φ(N) mod ℓ for
   small ℓ is NOT proven equivalent to factoring. No algorithm known.
   This is where a breakthrough would have maximum impact.

2. **PATH 1: Petersson/Kloosterman acceleration** — If the Kloosterman
   series could converge in poly(log N) terms via algebraic cancellation.
   Beyond endoscopy (Altug 2025) achieves this asymptotically but not
   for individual N.

3. **PATH 6 (Denis Charles): Push O(N^{1/2}) down** — Batch computation
   of H(4N-t²) for structured discriminant families. Cremona-Sutherland
   2024 improves individual computation; batch unexplored.

### Structurally Blocked

4. PATH 2 (EC for composites): No Frobenius at composites
5. PATH 3 (mod-ℓ Galois reps): Multiplicativity persists mod ℓ
6. PATH 4 (Voronoi): Wrong regime for q = N
7. PATH 8 (motives/geometry): CRT decomposition at every level
8. PATH 5 (amplification): Bounds not values; spectral reciprocity
   preserves multiplicativity in dual problem
9. PATH 13 (character sums/isogenies): Complete CRT decomposition;
   QR problem is the irreducible hard core; Castryck-Decru needs
   auxiliary torsion with no factoring analogue
10. PATH 14 (Rankin-Selberg): Coefficient extraction duality costs O(√N);
    Hiary inapplicable to squarefree conductor

### Not Applicable / Insufficient

11. PATH 6 (approximation): mod-ℓ needs exact value
12. PATH 9 (shifted convolution): Averages not individuals; Petersson exact
    series costs O(N); factor-dependent terms require factoring
13. PATH 10 (quantum): Shor already solves everything
14. PATH 11 (hardware): Exponential gap persists
15. PATH 12 (Waldspurger/period integrals): O(√N) cost, no τ(N) connection,
    CRT on local factors

---

## ADDITIONAL FINDINGS

### Non-Eigenforms and the Bach-Charles Bypass Question

Bach-Charles relies on multiplicativity a(pq) = a(p)·a(q) for eigenforms.
Could non-eigenforms bypass this? No:
- S₁₂(SL₂(Z)) is ONE-DIMENSIONAL (only Δ, an eigenform). No non-eigenforms exist.
- In higher dimension: non-eigenform coefficients are LINEAR COMBINATIONS of
  eigenform coefficients. Bach-Charles applies to each component.
- The mod-691 reduction gives σ₁₁(N), which is itself multiplicative.

### Lehmer's Conjecture and Lacunarity

- Lehmer's conjecture (τ(n) ≠ 0 for all n) verified to ~8.16 × 10²³
  (Derickx-van Hoeij-Zeng 2013). Withdrawn proof attempt (2025).
- Balakrishnan-Craig-Ono (2020): τ(n) ∉ {±1,±3,±5,±7,±691} for n > 1.
- Bellaiche-Soundararajan (2015): density of {n : τ(n) ≢ 0 mod 691} is
  zero (decays like (log x)^{-α}).
- The mod-691 Galois rep ρ_{Δ,691} is REDUCIBLE: ρ ≅ 1 ⊕ χ¹¹ (mod 691),
  precisely because of the Eisenstein congruence.
- None of these distribution results help COMPUTE τ(N) at individual composites.

---

## KEY REFERENCES

### Core Results
- Bach-Charles (2007), arXiv:0708.1192 — hardness of computing eigenforms
- Edixhoven-Couveignes (2006), arXiv:math/0605244 — poly(log p) for primes
- Denis Charles (2006), Ramanujan Journal — O(N^{1/2+ε}) under GRH
- Serre (1973), "Congruences et formes modulaires" — mod-ℓ representations

### Practical Implementations
- Mascot (2022), arXiv:2004.14683 — p-adic EC, 20-100x speedup
- Mascot (2013), arXiv:1211.1635 — complex-analytic EC implementation
- Bruin (2010), Leiden thesis — generalized EC
- Peng Tian (2022), arXiv:1905.10036 — non-Eisenstein extensions

### Trace Formulas
- Booker-Lee-Strömbergsson (2021), arXiv:2101.05663 — twist-minimal
- Altug (2025), arXiv:2505.18967 — beyond endoscopy with ramification
- Popa-Zagier (2017), arXiv:1711.00327 — simple proof of Eichler-Selberg

### Divisor Sum Complexity
- Boneh-Venkatesan (1998) — breaking RSA may be easier than factoring
- Aggarwal-Maurer (2009) — breaking RSA generically ≡ factoring

### Recent Factoring
- Ragavan-Regev (STOC 2025) — Jacobi factoring circuit (quantum)
- Ryan (EUROCRYPT 2025) — multivariate Coppersmith (n/4 unchanged)
- Regev (JACM 2025) — efficient quantum factoring
- Trey Li (ePrint 2025/1681) — Hecke problem is NP-hard

### L-function Algorithms and Rankin-Selberg
- Hiary (2011), Annals of Math. 174 — O(T^{1/3+ε}) for ζ(1/2+iT)
- Hiary (2012), arXiv:1205.4687 — Dirichlet L-functions with smooth modulus
- Vishe (2012), arXiv:1202.6303 — L(1/2, f⊗χ_q) for smooth q
- Odlyzko-Schonhage (1988) — amortized Dirichlet series evaluation
- Blomer-Harcos (2008), Duke Math J. 144 — shifted convolution spectral decomposition
- Blomer-Khan (2019), Duke Math J. 168 — spectral reciprocity
- Holowinsky (2009), Duke Math J. 146 — sieve for shifted convolution
- Nelson (2021), arXiv:2109.15230 — orbit method subconvexity
- Petrow-Young (2020), Annals of Math. — Weyl bound for L(1/2,χ)

### Modular Form Congruences
- Swinnerton-Dyer (1973) — congruences for τ(n) mod 2,3,5,7,23,691
- Lehmer (1947) — τ(n) ≡ 0 mod 23 iff n mod 23 ∈ specific set
- Serre (1966) — τ(n) odd ⟺ n is odd perfect square

### Character Sums and Isogeny Graphs (PATH 13)
- Van Dam-Seroussi (2002) — quantum Gauss sum estimation (DLP reduction)
- Castryck-Decru (2022), ePrint 2022/975 — broke SIDH via Kani reducibility
- Le Merdy-Wesolowski (2025), EUROCRYPT — unconditional EndRing equivalences
- Charles-Goren-Lauter (2009) — CGL hash from isogeny graphs
- Arpin (2022), arXiv:2203.03531 — level structure on isogeny graphs
- Cohen-Lenstra (1984) — APRCL primality and Jacobi sums

### Phi-Hiding and Partial Information (PATH 7)
- Cachin-Micali-Stadler (1999), EUROCRYPT — phi-hiding assumption
- Coppersmith (1996) — factoring with n/4 bits of p
- Morain-Renault-Smith (2023), arXiv:1802.08444 — deterministic factoring with φ oracle
- May-Nowakowski-Sarkar (2022), EUROCRYPT — approximate divisor multiples

### Lehmer and Distribution
- Balakrishnan-Craig-Ono (2020), arXiv:2005.10345 — excluded values of τ
- Bellaiche-Soundararajan (2015) — nonzero coefficients mod p
- Derickx-van Hoeij-Zeng (2013) — Lehmer verification to 8.16 × 10²³

### Period Integrals and Waldspurger (PATH 12 expansion)
- Waldspurger (1981), J. Math. Pures Appl. 60, 375-484 — foundational theorem
- Kohnen-Zagier (1981), Inventiones Math. 64, 175-198 — explicit formula for level 1
- Shimura (1973), Annals of Math. 97 — Shimura correspondence
- Shintani (1975), Nagoya Math. J. — Shintani lift construction
- Baruch-Mao (2007), GAFA 17, 333-384 — generalized Kohnen-Zagier to totally real fields
- Gross-Zagier (1986), Inventiones Math. 84 — heights of Heegner points
- Zhang (1997-2001), Annals of Math. — higher weight Gross-Zagier via Kuga-Sato
- Yuan-Zhang-Zhang (2013), Annals Math Studies 184 — Gross-Zagier on Shimura curves
- Kudla-Rapoport-Yang (2006), Annals Math Studies 161 — special cycles on Shimura curves
- Li-Zhang (2022) — proof of Kudla-Rapoport conjecture
- Ichino (2008), Compositio Math. — triple product period integral formula
- Hu-Nelson (2019), arXiv:1810.11564 — new test vectors, hybrid subconvexity
- Hu-Yin (2019), arXiv:1907.11428 — Waldspurger period integral for newforms
- Cai-Shu-Tian (2014), Algebra & Number Theory — explicit Gross-Zagier/Waldspurger
- Prasanna (2009), Inventiones Math. 176, 521-600 — SSW correspondence arithmetic
- Pacetti-Tornaría (2008), Exp. Math. 17, 459-472 — computing L-values, composite level
- Inam-Wiese (2022), Rocky Mountain J. Math. 52 — fast half-integral weight bases
- Ken Ono — Shimura correspondence and Ramanujan τ(n) function
- Nekovár — p-adic heights of Heegner cycles (Kuga-Sato extension)
- Watkins — Heegner point computations (Magma implementation)
