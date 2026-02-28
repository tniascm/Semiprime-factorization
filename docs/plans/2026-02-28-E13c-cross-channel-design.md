# E13c: Cross-Channel Eisenstein Congruence Experiments

## Problem Statement

E13 established that 7 Eisenstein congruence channels carry 63.3 bits of factor
information for balanced semiprimes, with trivial (poly(log N)) inversion. The
barrier is purely computational: evaluating channel values a_N(f_k) mod ell
costs O(N) via q-expansion, and Bach-Charles proves any poly(log N) route
implies factoring.

**Question:** Can cross-channel algebraic structure reduce the evaluation cost
below O(N)?

## Background

### Channel Structure

Each channel (k, ell) gives a polynomial equation:

    F_i(e1) := s_{k_i - 1}(e1) - c_i = 0  (mod ell_i)

where e1 = p + q, pq = N (known), and s_m(e1) = p^m + q^m satisfies the
Newton identity recurrence s_m = e1 * s_{m-1} - N * s_{m-2}.

Seven channels:

| Channel | Weight k | Prime ell | Poly degree | Bits |
|---------|----------|-----------|-------------|------|
| 1       | 12       | 691       | 11          | ~8.0 |
| 2       | 16       | 3617      | 15          | ~10  |
| 3       | 18       | 43867     | 17          | ~14  |
| 4       | 20       | 283       | 19          | ~7.0 |
| 5       | 20       | 617       | 19          | ~8.0 |
| 6       | 22       | 131       | 21          | ~6.0 |
| 7       | 22       | 593       | 21          | ~8.0 |

### Known Barriers

1. **Q-expansion:** O(N) — must compute N Fourier coefficients
2. **Trace formula:** O(sqrt(N)) class number sum is factor-free, but
   Eisenstein correction E(N) = -1 - min(p,q)^{k-1} requires factors.
   E13b confirmed |E/tau| stays O(1) — correction is NOT negligible.
3. **Congruence identity:** sigma_{k-1}(N) is multiplicative — circular
4. **Bach-Charles:** exact poly(log N) eval of a_N(f) implies factoring

### Key Observation

All 7 channels encode the SAME pair (p, q) through different polynomial
equations. Current approach solves each independently then CRTs. Cross-channel
structure has not been exploited.

## Experiments

### E13c-1: Resultant Degree Analysis

**Goal:** Determine whether pairs of channel polynomials share non-trivial
algebraic structure when lifted to a common ring.

**Method:**
1. For ~50 semiprimes (8-16 bit), compute all 7 channel polynomials
2. For each of 21 pairs (i,j), lift F_i and F_j to Z/(ell_i * ell_j)[e1]
3. Compute GCD of the lifted polynomials
4. Record: generic degree bound deg(F_i)*deg(F_j), actual common root count

**Rationale:** The polynomials F_i all arise from Newton identity recurrences
with the SAME N. If this shared structure forces non-trivial GCD, it reveals
algebraic dependencies exploitable for cost reduction.

**Expected:** Generic case — no degree reduction. But Newton identity structure
might introduce algebraic dependencies between s_{11} and s_{15} mod composite
modulus.

### E13c-2: Newton Recurrence Interpolation

**Goal:** Test whether class number sums (factor-free, O(sqrt(N)) cost) from
multiple weights can be combined to solve for min(p,q).

**Method:**
1. Compute ClassSum_k(N) for weights k = 12, 16 via Eichler-Selberg trace
   formula (O(sqrt(N)), no factoring required)
2. Each gives: ClassSum_k(N) = tau_k(N) - E_k(N) where E_k(N) = -1 - min(p,q)^{k-1}
3. Form the 2-equation system with shared unknown min(p,q)
4. Test solvability using the ratio of Eisenstein corrections across weights

**Rationale:** The Eisenstein correction E_k(N) depends on min(p,q) raised to
different powers for different weights. The ratio E_{k1}/E_{k2} eliminates
min(p,q) and equals min(p,q)^{k1-k2}. If class sums provide enough information
to reconstruct this ratio, we can extract min(p,q).

**Key subtlety:** ClassSum_k(N) = tau_k(N) - E_k(N), so we're observing
tau_k - E_k, not E_k alone. The circularity is that tau_k is what we want.
The experiment tests whether the overdetermined system (5 weights, 1 shared
unknown) breaks this circularity.

### E13c-3: Galois Representation Consistency

**Goal:** Test whether Galois representation structure at composite N provides
constraints beyond multiplicativity.

**Method:**
1. Compute traces of symmetric powers Sym^m(rho_{f,ell}) at Frob_N for m=1,2,3
2. For each channel, these give s_m(alpha_p*alpha_q, ...) where alpha_p, beta_p
   are roots of x^2 - a_p*x + p^{k-1}
3. Test whether inter-channel constraints from the tensor product Frob_p x Frob_q
   produce independent equations beyond simple multiplicativity a_N = a_p * a_q

**Rationale:** The mod-ell Galois representation rho_{f,ell} for Eisenstein
congruence primes is reducible: rho = 1 + chi^{k-1}. At composite N, the
interaction between Frob_p and Frob_q in tensor products and exterior powers
may produce constraints invisible to the multiplicativity formula alone.

### E13c-4: Multi-Weight Trace Formula Elimination (Primary)

**Goal:** Test whether class number sums from multiple weights, combined over
the integers, can eliminate the Eisenstein correction and recover min(p,q).

**Method:**
1. Compute ClassSum_k(N) for all 5 distinct weights (k = 12, 16, 18, 20, 22)
   using Eichler-Selberg trace formula — O(sqrt(N)) per weight, factor-free
2. For each pair of weights (k1, k2), form the system:
   - ClassSum_{k1}(N) = tau_{k1}(N) + 1 + min(p,q)^{k1-1}
   - ClassSum_{k2}(N) = tau_{k2}(N) + 1 + min(p,q)^{k2-1}
3. Attempt to solve for min(p,q) using:
   a. Ratio method: if tau_k is small relative to E_k, the ratio approximates
      min(p,q)^{k1-k2}
   b. Newton iteration: treat min(p,q) as the single real unknown, use the
      overdetermined 5-equation system
   c. Lattice methods: view the system as a closest vector problem
4. Verify: does recovered min(p,q) equal the true smaller factor?

**The prize:** If this works, it gives O(sqrt(N)) factoring using only the
trace formula — no q-expansion, no Bach-Charles conflict (we never compute
a_N(f) directly).

**Why it might fail:** For balanced semiprimes, |tau_k(N)| ~ N^{(k-1)/2} and
|E_k(N)| ~ min(p,q)^{k-1} ~ N^{(k-1)/2}. Both terms are the same order, so
the class sum doesn't cleanly separate tau from E. The system is ill-conditioned.

**Why it might work:** Five weights give 5 equations with 6 unknowns
(5 tau values + min(p,q)). But the 5 tau values are NOT independent — they
satisfy Hecke relations and growth bounds. These additional constraints might
make the system overdetermined.

## Implementation

**Script:** `E13_bach_charles/run_E13c_cross_channel.sage`

**Test data:** 50 balanced semiprimes, 10 per size class (8, 10, 12, 14, 16 bit).
Small enough for fast q-expansion verification, large enough for pattern detection.

**Dependencies:**
- Existing `power_sum_poly()` and `solve_congruence_channel()` from E13
- Existing `hurwitz_class_number()` and trace formula from E13b (with bug fixes)
- `utils/semiprime_gen.py` for balanced semiprime generation

**Output:** `data/E13c_cross_channel.json`

**Verification:** All results cross-checked against known factors.

## Success Criteria

- **E13c-1:** Any non-trivial GCD (degree > 0) between lifted channel polynomials
- **E13c-2:** Solvable 2-equation system recovering min(p,q) from class sums
- **E13c-3:** Any inter-channel Galois constraint beyond multiplicativity
- **E13c-4:** Numerical recovery of min(p,q) from multi-weight class sums

Any single success would represent a novel structural result. E13c-4 success
would be a factoring algorithm improvement (O(sqrt(N)) vs O(N) for channel
evaluation).

## Risk Assessment

All four experiments may fail — the barriers are deep. Expected outcome:
E13c-1 confirms no algebraic GCD structure, E13c-2 and E13c-4 show
ill-conditioning that prevents numerical solution, E13c-3 shows Galois
constraints reduce to multiplicativity. This would definitively close the
cross-channel corridor.
