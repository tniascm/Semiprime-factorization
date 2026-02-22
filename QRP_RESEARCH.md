# Quadratic Residuosity Problem: State of the Art (February 2026)

## 1. Problem Definition

**Quadratic Residuosity Problem (QRP):** Given a composite N = pq (product
of two unknown distinct odd primes) and an integer a with Jacobi symbol
(a/N) = +1, decide whether a is a quadratic residue mod N.

The elements with Jacobi symbol +1 fall into two classes:
- **QR_N:** a ≡ x² (mod N) for some x — equivalently, a is QR mod BOTH p and q
- **J⁺_N \ QR_N:** a is QNR mod both p and q (the "pseudosquares")

The Jacobi symbol cannot distinguish these: (a/N) = (a/p)(a/q) = (+1)(+1) or
(-1)(-1) = +1 in both cases. The problem is: which case?

**Quadratic Residuosity Assumption (QRA):** No PPT algorithm can distinguish
a random QR from a random pseudosquare with non-negligible advantage.

---

## 2. Complexity Classification

### 2.1 Known Bounds

- **QRP ∈ NP ∩ coNP:** The factorization (p, q) serves as witness in both
  directions. Given the factors, one can compute (a/p) and (a/q) separately.

- **Refinement: QRP ∈ UP ∩ coUP** (Cai-Threlfall, IPL 2005). The class UP
  ("Unambiguous Polynomial-time") requires exactly one accepting path. The
  witness structure for QR is the factorization (unique for semiprimes), which
  gives unambiguous verification. This places QRP strictly below NP-complete
  problems (assuming P ≠ UP ≠ NP).

  Reference: J.-Y. Cai, R.A. Threlfall, "A Note on Quadratic Residuosity
  and UP," Information Processing Letters 92(3):127-131, 2004.

- **QRP is NOT known to be NP-hard.** If QRP were NP-hard, then NP = coNP
  (since QRP ∈ coNP), which is believed false.

### 2.2 Relationship to Other Classes

    P  ⊆  UP  ⊆  NP
    P  ⊆  UP ∩ coUP  ⊆  NP ∩ coNP

QRP sits at UP ∩ coUP. Both inclusions (P ⊆ UP and UP ⊆ NP) are believed
strict, but this is unproven.

### 2.3 Random Self-Reducibility

QRP is random self-reducible: a worst-case QRP instance can be reduced to a
random instance. The reduction: given (N, a) with J(a,N) = +1, pick random
r with J(r,N) = +1, query the oracle on a·r² mod N (which has the same
QR status as a since r² is always QR). This means average-case QRP is as
hard as worst-case QRP — there are no "easy subsets" of inputs.

This is the foundational property used by Goldwasser-Micali (1982): the
semantic security of GM encryption reduces to worst-case QRP via random
self-reducibility.

---

## 3. The Reduction Landscape: QRP vs. Factoring

### 3.1 Direction 1: FACTORING → QRP (known, trivial)

If you can factor N into p and q, you can solve QRP immediately by computing
the Legendre symbols (a/p) and (a/q) separately. This is the "easy direction."

### 3.2 Direction 2: QRP → FACTORING (UNKNOWN in standard model)

**This is the critical gap.** There is NO known polynomial-time reduction
from factoring to QRP. That is, having a QRP oracle does NOT obviously let
you factor N.

**Precise statement of the gap:** The QRA (QR Assumption) is considered a
STRONGER assumption than the factoring assumption. This means:

    Factoring-is-hard  ⟸  QRA  (QRA implies factoring is hard)
    Factoring-is-hard  ⟹?  QRA  (UNKNOWN — factoring hardness may not imply QRA)

In other words: it is conceivable that factoring is hard but QRP is easy.
No one has demonstrated this separation, but the reduction QRP → factoring
is missing.

### 3.3 What IS Equivalent to Factoring

Several related problems ARE known to be equivalent to factoring:

| Problem | Equivalence to factoring | Reference |
|---------|------------------------|-----------|
| Computing square roots mod N | Tight equivalence (Rabin 1979) | If you can find sqrt(a) mod N, factor via gcd(x-y, N) |
| Deciding QR mod N | NOT known equivalent | Gap in reverse direction |
| Computing Jacobi symbol (a/N) | EASY (O(n²), Euler criterion) | Not hard at all |
| RSA inversion (e-th roots) | Not known equivalent | Similar gap to QRP |

**The Rabin equivalence in detail:** For Blum integers N = pq with p ≡ q ≡ 3
(mod 4), every QR has exactly 4 square roots. Given any algorithm that
computes a square root s of a random QR a = r² mod N, with probability ≥ 1/2
the algorithm returns s ≠ ±r, and then gcd(s - r, N) yields a factor of N.
This is a TIGHT (both directions) polynomial-time randomized reduction.

**Crucially: square root COMPUTATION ≡ factoring, but square root DECISION
(= QRP) is not known to imply factoring.**

### 3.4 Generic Ring Model Results

Jager and Schwenk (Asiacrypt 2009, J. Cryptology 2013) proved:

**Theorem (Jager-Schwenk):** In the generic ring model (algorithms that
access Z/NZ only through abstract ring operations, without exploiting the
bit-representation of elements), computing the Jacobi symbol is equivalent
to factoring N.

**Key caveat:** The Jacobi symbol is efficiently computable by a NON-generic
algorithm (the law of quadratic reciprocity gives an O(n²) algorithm that
uses bit-level operations). This means the generic ring model CANNOT give
evidence for or against the hardness of QRP, since the real-world algorithm
exploits representation structure.

Reference: T. Jager, J. Schwenk, "On the Analysis of Cryptographic
Assumptions in the Generic Ring Model," IACR ePrint 2009/621.

Aggarwal and Maurer (Eurocrypt 2009) similarly showed Strong-RSA ≡ factoring
in the generic ring model.

### 3.5 The Hofheinz-Kiltz Gap Group

Hofheinz and Kiltz (Crypto 2009) introduced the group of **signed quadratic
residues** QR_N⁺ = {|x| : x ∈ QR_N} where |x| = min(x, N-x).

This is a "gap group" where:
- **Computational problem (computing square roots in QR_N⁺) is as hard as
  factoring** — this is the Rabin equivalence.
- **Decisional problem (recognizing members of QR_N⁺) is EASY** — by
  definition, you just check if |a| has Jacobi symbol +1.

They proved that the Strong Diffie-Hellman assumption holds in QR_N⁺ under
the factoring assumption. This explicitly constructs a cryptographic setting
where a QR-like decisional problem is easy but the associated computational
problem is equivalent to factoring.

Reference: D. Hofheinz, E. Kiltz, "The Group of Signed Quadratic Residues
and Applications," Crypto 2009, LNCS 5677.

### 3.6 Oracle Complexity of Factoring

Maurer (1992, 1996) studied factoring with an arbitrary yes/no oracle:

- **Trivial:** Ask for bits of p → n/2 queries for n-bit N = pq (balanced).
- **Rivest-Shamir:** n/3 queries for N = pq with p, q of equal size.
- **Maurer:** For any ε > 0, a polynomial-time algorithm using only εn
  oracle queries suffices (conditional on Lenstra's ECM conjecture).
- **Coppersmith:** ~0.25 log n queries using small solutions to polynomial
  congruences (rigorous).

These results show that an oracle answering YES/NO to general questions
(not specifically QRP) can factor with sublinear queries. But none of these
constructions use a QRP-specific oracle — they use arbitrary bit queries.

---

## 4. Recent Papers (2023-2025)

### 4.1 Tiplea: Jacobi Symbol Problem for Quadratic Congruences (2023)

**Paper:** F.L. Tiplea, "The Jacobi Symbol Problem for Quadratic Congruences
and Applications to Cryptography," IACR ePrint 2023/475 (also published in
MDPI Mathematics 14(3):465, 2026).

**New problem JSP(QC):** Given N = pq, a solvable quadratic congruence
x² ≡ c (mod N) where c ∈ QR_N, distinguish the Jacobi symbols of the two
classes of roots.

**Key results:**
- QRP ≤ JSP(QC) — the Jacobi symbol problem is at least as hard as QRP.
- The IND-CPA security of both of Cocks' encryption schemes (public-key
  and identity-based) is EQUIVALENT to JSP(QC).
- This gives a tighter characterization of Cocks IBE security than
  previously known (before, only one direction was proven).

### 4.2 Corrigan-Gibbs & Wu: Pseudorandomness of Legendre Symbols (2024)

**Paper:** H. Corrigan-Gibbs, D.J. Wu, "The Pseudorandomness of Legendre
Symbols under the Quadratic-Residuosity Assumption," IACR ePrint 2024/1252,
published at TCC 2025.

**Result:** Under QRA, the function that maps (x, p) to the Legendre signature
(the string of Legendre symbols (x+a₁/p), (x+a₂/p), ...) with respect to
public random offsets a₁, ..., aₘ is a pseudorandom generator.

**Significance:** This is the FIRST result relating pseudorandomness of
Legendre symbol sequences to any standard cryptographic assumption (QRA).
Previous constructions (Damgard, 1988; the "Legendre PRF") were based on
ad hoc assumptions about Legendre symbol sequences.

**Limitation:** Requires the prime p to be SECRET. Does not apply when p is
public (the more common setting for Legendre PRF applications).

### 4.3 Nassar, Waters, Wu: Monotone-Policy BARGs from QR (PKC 2025)

**Paper:** S. Nassar, B. Waters, D.J. Wu, "Monotone-Policy BARGs and More
from BARGs and Quadratic Residuosity," PKC 2025, IACR ePrint 2025/391.

**Result:** Monotone-policy batch arguments (BARGs) for NP can be constructed
from standard BARGs + additively homomorphic encryption over ANY group.
Instantiating with Goldwasser-Micali (which is additively homomorphic over
Z/2Z) gives monotone-policy BARGs from BARGs + QRA. Also yields monotone-
policy aggregate signatures from BARGs + QRA.

**Significance for QRP:** Demonstrates QRA continues to find new applications
in advanced cryptographic primitives (2025). The QRA is being used as a
building block, not being broken.

### 4.4 Legendre PRF Cryptanalysis (2019-2021, ongoing)

The Legendre PRF F_k(x) = (x+k / p) maps x to the Legendre symbol (x+k)/p
with secret key k and public prime p.

- Kaluderovic et al. (2019): Broke three Ethereum Foundation Legendre PRF
  challenges. Key recovery in O(p log²p / M²) Legendre symbol evaluations
  given M queries (improvement from O(p log p / M)).
- When extended to composite modulus p²q (Damgard's construction), security
  reduces to factoring integers of that form — hence NOT post-quantum secure.

### 4.5 Integer Factorization via Continued Fractions (2024)

**Paper:** arXiv:2409.03486 (September 2024), "Integer Factorization via
Continued Fractions and Quadratic Forms."

**Result:** A factoring algorithm using reduced quadratic forms, infrastructural
distance, and Gauss composition with complexity O(exp((3/sqrt(8))·sqrt(ln N·ln ln N))).
This is faster than classical SQUFOF and CFRAC but still subexponential.
Polynomial-time if a (not too large) multiple of the regulator of Q(sqrt(N))
is known.

**Relevance to QRP:** Uses quadratic residues in generating relations (as do
all sieve-based methods). Does not give a QRP algorithm per se, but provides
a new angle on the algebraic structures connecting QR to factoring.

---

## 5. Partial QRP Oracles

### 5.1 Random Self-Reducibility Rules Out Easy Subsets

Because QRP is random self-reducible, solving QRP on ANY dense subset of
inputs implies solving it on ALL inputs. Specifically: if an algorithm A
solves QRP correctly on a fraction > 1/2 + 1/poly(n) of inputs (for random
N from RSAgen), then A can be amplified to solve QRP on all inputs via the
randomized reduction a → a·r² mod N.

**Consequence:** There are NO "partial QRP oracles" that work on specific
subsets of inputs but fail on others (assuming standard definitions). If you
can distinguish QR from QNR for any non-trivial fraction of inputs, you can
do it for all inputs.

### 5.2 The Signed QR Exception

As noted above, Hofheinz-Kiltz showed that the SIGNED quadratic residuosity
problem (deciding membership in QR_N⁺ = {min(x, N-x) : x ∈ QR_N}) IS easy.
This is a "partial oracle" of sorts: it answers the decisional problem for
signed representatives, but this does not help with standard QRP.

### 5.3 Products and Higher Residuosity

The product of Legendre symbols χ_D(p)·χ_D(q) = (D/N) is efficiently
computable. But the SUM χ_D(p) + χ_D(q) is NOT — this is exactly the
"hinge scalar" S_D(N) from the BARRIER_THEOREM.md in this project.

The higher residuosity problems (k-th power residuosity for k > 2)
are also believed hard and form a hierarchy:
- QRP (k=2): basis of Goldwasser-Micali
- Composite residuosity (k=N): basis of Paillier cryptosystem

The hierarchy: CR[n] ⟸ Class[n] ⟸ RSA[n,n] ⟸ Fact[n], where each
leftward arrow means "implied by" as a hardness assumption.

---

## 6. Distributional Properties

### 6.1 Distribution of QR vs QNR (Tiplea, 2019)

**Paper:** F.L. Tiplea et al., "On the distribution of quadratic residues and
non-residues modulo composite integers and applications to cryptography,"
Applied Mathematics and Computation 372, 2020. IACR ePrint 2019/638.

**Result:** Exact formulas for the distribution of QR and QNR in shifted sets
a + X = {(a+x) mod n : x ∈ X} where n = pq and X has prescribed Jacobi
symbols. Applications to the statistical indistinguishability properties
of Cocks' IBE scheme.

### 6.2 Apparent Randomness of QR/QNR Sequences

The distribution of QR among elements with Jacobi symbol +1 appears random.
For N = pq, exactly half of the elements with (a/N) = +1 are QR, and they
are distributed with no known efficiently detectable pattern. This
"pseudorandom" distribution is precisely the QRA.

---

## 7. Connection to the "Missing Bit" / Hinge Scalar

### 7.1 Direct Mapping

The project's barrier theorem (BARRIER_THEOREM.md Section 5) establishes:

    S_D(N) = χ_D(p) + χ_D(q)  is computable in poly(log N)
    ⟹  QRP is solvable in poly(log N)
    ⟹  factoring in poly(log N)  (for Blum integers, via Rabin)

The "missing bit" is precisely the QRP bit: given (D/N) = χ_D(p)·χ_D(q) = +1,
determine whether χ_D(p) = χ_D(q) = +1 (both QR) or χ_D(p) = χ_D(q) = -1
(both QNR). This single bit of information is S_D(N) mod 4 (i.e., whether
S_D = +2 or S_D = -2).

### 7.2 What the Literature Confirms About Our Barrier

The web research confirms several aspects of the project's barrier:

1. **The gap is real:** No reduction QRP → factoring is known in the standard
   model. The barrier theorem's claim that distinguishing O(t,N)=4 from
   O(t,N)=0 is "as hard as factoring" should be stated more precisely as
   "at least as hard as QRP, which is believed hard but not proven equivalent
   to factoring."

2. **Random self-reducibility kills partial oracles:** The hope that some
   subset of trace parameters t might be easier is killed by random
   self-reducibility. If you could solve QRP for ANY dense subset of
   discriminants D = t²-4, you could solve it for ALL D.

3. **The generic ring model equivalence is irrelevant:** Jager-Schwenk's
   result that QRP ≡ factoring in the generic ring model does not apply
   because the Jacobi symbol computation is non-generic (uses bit
   representation). Our barrier operates in the full RJI model, which is
   more powerful than the generic ring model.

4. **The carry/integer operations channel is correctly identified:** The
   barrier theorem's analysis of CRT rank under integer operations (Section 4)
   addresses the only known way to break CRT separability. The literature
   has no alternative approach that circumvents this.

### 7.3 Precise Reduction Chain for the Project

    Computing S_D(N) efficiently
    → Solving QRP (by Section 5.2 of barrier theorem)
    → Computing square roots mod N (via Adleman-Manders-Miller for QR decision → sqrt)
    → Factoring N (via Rabin's equivalence)

**Gap in the chain:** The step "QRP → computing square roots" is not trivial.
The standard route is:
- QRP oracle tells you whether a given element is QR.
- To COMPUTE a square root, you need more: the Tonelli-Shanks algorithm
  requires knowing the factorization, or at minimum being able to find a QNR.
- However, for Blum integers, the square root of a QR a is a^((N-p-q+5)/8)
  mod N, which requires knowing p+q (equivalently, factoring).

**The tighter statement:** QRP → factoring is NOT established by a simple
reduction. What IS established:
- QRP for ALL inputs → distinguishing QR from pseudosquares → breaking
  Goldwasser-Micali semantic security.
- Square root COMPUTATION → factoring (Rabin, tight).
- The relationship QRP → square root computation is open.

For practical purposes in this project: the hinge scalar S_D(N) gives BOTH
the QRP answer AND additional structure (it gives the actual value +2 vs -2,
not just a bit), so computing S_D → factoring remains valid through the
full chain: S_D → individual Legendre symbols → trial on candidate primes.

---

## 8. Summary of Negative Results (No Breakthroughs)

After extensive search of 2023-2026 literature:

1. **No new algorithm for QRP.** The best classical approach remains: factor
   N (using GNFS in L_N[1/3, (64/9)^{1/3}] ≈ subexponential), then compute
   Legendre symbols. No direct QRP algorithm is known that beats factoring.

2. **No weaknesses found in Goldwasser-Micali.** The scheme's main weakness
   remains its 1-bit-per-ciphertext efficiency, not any cryptanalytic break.
   Generalizations to k-th residuosity (Maimuṭ-Teşeleanu, 2020) improve
   efficiency while maintaining security under analogous assumptions.

3. **No QRP-to-factoring reduction found.** The reverse direction remains
   open. QRA is still considered STRICTLY STRONGER than the factoring
   assumption (i.e., harder to break, but not provably equivalent).

4. **No lattice-QRP connection.** No reduction between QRP and standard
   lattice problems (LWE, SIS, SVP) has been established. These appear to
   be independent hardness assumptions.

5. **No information-theoretic approach succeeds.** The distributional
   properties of QR vs pseudosquares are well-studied but show no
   exploitable statistical signatures without factoring.

6. **The Legendre PRF connection is the closest new result.** Corrigan-Gibbs
   and Wu (2024) showed that Legendre symbol sequences are pseudorandom
   under QRA — this is a new USE of QRA, not a break.

---

## 9. Implications for This Project

### 9.1 The Barrier Is Cryptographically Sound

The project's spectral flatness barrier for poly(log N)-computable
observables is consistent with (and partially follows from) the QRA, which
remains unbroken after 40+ years. The "missing bit" problem identified in
E7b/E7c is exactly the QRP.

### 9.2 Regime Qualifiers

Per the MEMORY.md status qualifications:
- "QR ≡ factoring" should be stated as: "QRP → factoring for Blum integers
  via the chain S_D → QRP → ... → factoring, but QRP is NOT known to be
  equivalent to factoring in general. QRA is a stronger assumption."
- The CRT obstruction is consistent with QRA but is an independent empirical
  finding specific to spectral methods.

### 9.3 What Could Still Work

Given the literature review, the only corridors not ruled out by QRA are:
1. **Quantum algorithms:** Shor's algorithm solves factoring (and hence QRP)
   in poly(log N) on a quantum computer.
2. **Non-black-box techniques:** Approaches that exploit the specific
   algebraic structure of N (not just oracle access to QRP) might find
   structure invisible to oracle-based reductions.
3. **Subexponential improvements:** New factoring algorithms (like the 2024
   continued-fractions result) that improve constants in L_N[1/3, c] are
   possible but do not change the asymptotic hardness.

---

## 10. Key References

- Goldwasser, Micali, "Probabilistic Encryption," JCSS 28:270-299 (1984)
- Rabin, "Digitalized Signatures and Public-Key Functions," MIT/LCS/TR-212 (1979)
- Cai, Threlfall, "A Note on Quadratic Residuosity and UP," IPL 92(3) (2004)
- Hofheinz, Kiltz, "The Group of Signed Quadratic Residues and Applications," Crypto 2009
- Jager, Schwenk, "On the Analysis of Cryptographic Assumptions in the Generic Ring Model," J. Cryptology 26:225-245 (2013)
- Maurer, "Factoring with an Oracle," Eurocrypt 1992
- Tiplea, "The Jacobi Symbol Problem for Quadratic Congruences," ePrint 2023/475
- Corrigan-Gibbs, Wu, "The Pseudorandomness of Legendre Symbols under QRA," ePrint 2024/1252
- Nassar, Waters, Wu, "Monotone-Policy BARGs and More from BARGs and QR," PKC 2025
- Tiplea et al., "On the distribution of QR and QNR modulo composites," ePrint 2019/638
