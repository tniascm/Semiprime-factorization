# Access Model Requirements for Poly(log N) Factoring via Langlands

## 1. The Complexity Constraint

**Hard requirement:** Any factoring algorithm must run in poly(log N) = poly(n) time,
where n = log₂(N) is the input length. This means:

- O(N^α) for any fixed α > 0 is **exponential** in n — not acceptable
- O(√N) = O(2^{n/2}) — this is trial division complexity, the baseline
- The goal is O(n^c) for some constant c

**Implication for level-N spaces:** dim S_k(Γ_0(N)) = O(N). Any computation
that enumerates basis elements, builds matrices, or iterates over eigenforms
costs at least O(N) = O(2^n). This kills all "work inside the space" approaches.

---

## 2. What Our Experiments Have Closed

### Finite-Place / Local Observables (E7-E7e)

**Closed:** Any pointwise function of Jacobi symbols S(t) = Φ(J(f₁(t),N), ..., J(f_k(t),N))
factors as h(a)·g(b) in CRT coordinates and has flat DFT.

**Observable class:** 19 specific observables tested + CRT product structure
argument covering all pointwise Jacobi functions.

**Cost:** O(N log N) for DFT (exponential), but even the information content
is zero — the DFT is provably flat.

### Twisted GL(2) L-functions (E8a-E8b)

**Closed:** L(s, f ⊗ χ_N) for level-1 eigenforms (weights 12-36) twisted by
the Kronecker character χ_N.

**Why they fail:** χ_N(p) = 0 at p|N, so the local Euler factor at bad primes
is "omitted" — the only local information is "this prime divides N" (= gcd test).

**Global coupling:** Exists in principle (confusable pairs differ, critical line
R² = 0.21) but too weak to extract. Multi-form amplification (19 forms) yields
R² ≈ 0 under LOOCV.

**Root number:** ε(f ⊗ χ_N) = (-1)^{k/2} for ALL level-1 forms, independent
of N. Algebraic proof: the two χ_N(-1) factors cancel. Closes root number
as an information carrier for this entire family.

### Level-N Space Computations (E9, abandoned)

**Closed by complexity, not by experiment:** dim S_k(Γ_0(N)) = O(N). Any
computation on this space (Hecke matrices, traces, eigenvalues) takes at
least O(N) time — exponential in input size.

**Theoretical content:** The Eichler-Selberg trace formula provides a "succinct"
representation of Tr(T_ℓ | S_k(Γ_0(N))) as O(√ℓ) class number terms. But
evaluating the local embedding numbers at p|N requires knowing the factorization.

---

## 3. The Missing-Bit Barrier (Precise Formulation)

### The core problem

For t with J(t²-4, N) = +1 and gcd(t²-4, N) = 1, the Jacobi symbol tells us:

    χ_{t²-4}(p) · χ_{t²-4}(q) = +1

This means either χ(p) = χ(q) = +1 (both QR) or χ(p) = χ(q) = -1 (both QNR).
Distinguishing these cases is the **quadratic residuosity problem**, believed
equivalent to factoring under standard assumptions.

### What would break it

For any discriminant D with gcd(D, N) = 1:

    r_D(N) = #{representations of N by principal form of disc D}
           = (1 + χ_D(p))(1 + χ_D(q))
           = 1 + (χ_D(p) + χ_D(q)) + χ_D(N)

We can compute χ_D(N) in O(log²N) (Jacobi/Kronecker symbol). The "missing
piece" is χ_D(p) + χ_D(q) = r_D(N) - 1 - χ_D(N).

**Therefore:** A poly(log N) algorithm for r_D(N) — the number of representations
of N by a single quadratic form — would break the missing-bit barrier and
enable factoring.

### Known complexity of r_D(N)

- **Cornacchia's algorithm:** O(√N) = O(2^{n/2}) per form [exponential]
- **Brute force:** Enumerate (x,y) with Q(x,y) = N: O(√N) [exponential]
- **Via theta series:** r_D(N) is the N-th Fourier coefficient of θ_Q(z).
  Extracting a single coefficient requires O(N) evaluations or O(√N) via AFE
  [exponential]
- **Via class field theory:** r_D(N) = Σ_{d|N} χ_D(d) when summed over ALL
  forms of discriminant D. But for a SINGLE form, no known shortcut exists.

**No known poly(log N) method exists for computing r_D(N).**

---

## 4. Allowed Primitive Operations (poly(log N))

The following operations are known to be computable in poly(log N) = poly(n):

| Operation | Cost | Notes |
|-----------|------|-------|
| Arithmetic mod N (+, ×, mod) | O(n²) | Standard |
| gcd(a, N) | O(n²) | Euclidean algorithm |
| Jacobi symbol (a/N) | O(n²) | Quadratic reciprocity |
| Modular exponentiation a^k mod N | O(n³) | Square-and-multiply |
| Modular inverse a⁻¹ mod N | O(n²) | Extended Euclidean |
| Primality testing | Õ(n⁶) | AKS (deterministic) |
| Elliptic curve point operations mod N | O(n²) per op | Group law |
| Polynomial evaluation mod N | O(deg · n²) | Horner's method |

### What these CANNOT do (conditional on factoring being hard)

- Compute χ_D(p) or χ_D(q) individually (= factoring)
- Compute r_D(N) for a specific form (= factoring, empirically)
- Distinguish QR/QR from QNR/QNR when J = +1 (= quadratic residuosity)
- Compute the order of the group (Z/NZ)* (= factoring via Pohlig-Hellman)

---

## 5. What a Successful "Langlands Primitive" Would Need to Do

A new primitive P(N, aux) that enables poly(log N) factoring must satisfy:

1. **Computability:** P(N, aux) is computable in poly(log N) time for some
   auxiliary parameter aux of poly(log N) size.

2. **Factor sensitivity:** P(N, aux) depends on the factorization of N in a
   way not reducible to {+, ×, gcd, Jacobi, modexp} operations.

3. **Non-CRT:** P(N, aux) is NOT expressible as f(N mod p) · g(N mod q)
   or any simple CRT-product decomposition.

4. **Extractability:** From poly(log N) evaluations of P(N, aux_1), ..., P(N, aux_k),
   one can recover p and q (or a nontrivial factor) in poly(log N) time.

### Candidate mechanisms (not yet ruled out)

**A. Succinct spectral projector / resolvent**

The trace formula provides:
    Tr(T_ℓ | S_k(Γ_0(N))) = Σ (spectral terms) = Σ (geometric terms)

If the geometric side has a "succinct" evaluation that doesn't require
enumerating divisors of N, this could be a poly(log N) primitive.

**Status:** The Eichler-Selberg formula's geometric terms DO require divisor
enumeration (local embedding numbers at p|N). No known alternative evaluation
method avoids this.

**What would help:** A way to evaluate the FULL product ∏_{p|N}(1+χ_D(p))
from N alone, without knowing individual primes. This equals r_D(N)/h(D)
(representation count divided by class number), circling back to §3.

**B. Analytic continuation / functional equation shortcut**

L-functions satisfy functional equations that relate values at s to values at
k-s. These functional equations involve conductors, root numbers, and gamma
factors that depend on the factorization. If evaluating the functional equation
"implicitly" at a single point could be done in poly(log N) without knowing
the conductor's factorization, this might work.

**Status:** The functional equation for L(s, f ⊗ χ_N) involves the conductor
N² and root number ε = (-1)^{k/2} (for level-1, quadratic twist). The root
number is independent of the factorization (E8b algebraic proof). The gamma
factors depend only on k (fixed). So the FE doesn't carry factor information
in this family.

For other families (level N forms), the FE involves Atkin-Lehner eigenvalues
w_p, w_q, which DO depend on the factorization. But computing these requires
working at level N (exponential).

**C. Arithmetic topology / QFT partition function**

Kim's arithmetic Chern-Simons theory and related "partition function = L-function"
ideas provide a different mathematical framework where primes play the role of
knots and Spec(Z) plays the role of a 3-manifold. In principle, partition
functions can sometimes be computed without enumerating all states.

**Status:** Entirely theoretical. No algorithmic content. No complexity analysis.
No known way to evaluate any of these objects even in exponential time for
specific N.

---

## 6. The Formal No-Go Template

### Conjecture (informal, to be made precise)

Any function F(N) computable in poly(log N) from (Z/NZ, +, ×, gcd, Jacobi, modexp)
cannot distinguish QR/QR from QNR/QNR cases with non-negligible advantage.

### What our experiments provide (empirical evidence for this conjecture)

| Experiment | Observable class | Result |
|:---:|:---|:---:|
| E7c | Pointwise Jacobi functions (8 types) | Flat DFT |
| E7c | CRT product structure argument | Covers ALL pointwise Jacobi |
| E7e | Arithmetic weights, Dirichlet twists | No signal |
| E7e | Multi-discriminant, SVD, kurtosis | Anti-peaks (N^{-0.25}) |
| E8a | Twisted L-function values | Reduce to gcd |
| E8b | 19-form amplification | R² ≈ 0 |

### What would be needed for a formal proof

1. Formalize the "poly(log N) computation from ring primitives" model
2. Show that any such computation can be simulated by a bounded-depth
   arithmetic circuit over Z/NZ
3. Show that such circuits cannot distinguish QR/QR from QNR/QNR
   (this is related to the quadratic residuosity assumption)

This would essentially prove: QR assumption ⟹ no poly(log N) Langlands
factoring via ring primitives. The gap is whether any Langlands construction
provides a primitive OUTSIDE this ring model.

---

## 7. Open Questions (for web survey)

1. **Succinct trace formula evaluation:** Is there any method to compute
   Tr(T_ℓ | S_k(Γ_0(N))) without knowing the factorization of N? Can the
   geometric side of the Selberg trace formula be evaluated modularly
   (working mod N instead of mod p, mod q)?

2. **Fast representation number computation:** Is r_D(N) for a specific
   quadratic form Q of discriminant D computable in less than O(√N) time?
   Any subexponential method?

3. **Analytic continuation of automorphic kernels:** Can Braverman-Kazhdan
   or Ngô's kernel be evaluated at specific points without computing the
   full transform? Is there a "fast" BK transform analogous to FFT?

4. **p-adic methods:** Can p-adic L-functions or p-adic modular forms
   provide factor information through their special values? Are there
   poly(log N) p-adic computations that access spectral data?

5. **Algebraic K-theory / motivic cohomology:** Do any algebraic invariants
   of Spec(Z/NZ) carry factor information and have succinct evaluation?

6. **Quantum vs classical gap:** Shor's algorithm exploits the quantum
   Fourier transform to find periods in poly(n) time. Is there a classical
   analogue of "implicit access to the period" via automorphic methods?
   (This connects to derandomization of hidden subgroup problems.)

---

## 8. Summary of Status

**Closed corridors (with evidence type):**
- Finite-place Jacobi observables (structural proof + experiments)
- Twisted GL(2) L-functions, level 1 (experiments + χ(p)=0 structural argument)
- Level-N space computations (complexity barrier, O(N) minimum)
- Multi-form amplification (experiment, R² ≈ 0)

**Remaining corridors (speculative, no evidence for or against):**
- Succinct spectral projectors (no known evaluation method)
- Analytic continuation shortcuts (no known poly(log N) method)
- Arithmetic topology / QFT invariants (no algorithmic content)
- p-adic methods (unexplored)

**The fundamental bottleneck:**
Computing r_D(N) — the representation of N by a quadratic form — in poly(log N).
All known methods require O(√N). Breaking this would break factoring.
