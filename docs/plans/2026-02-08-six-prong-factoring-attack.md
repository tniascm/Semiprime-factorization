# Six-Prong Attack Plan: Beating GNFS for RSA Factoring

## Date: 2026-02-08
## Status: Design Complete, Awaiting Implementation

---

## Context and Motivation

After building 17 Rust crates and running 15 experiments across spectral methods, L-functions, tensor networks, lattice reduction, group theory, compression, and AI-guided search, we have a clear picture:

**Current state of the art (2026):**
- GNFS: L[1/3, 1.923] — best known classical algorithm, unchanged since 1993
- TNSS (Tesoro/Siloi 2024): Claims polynomial (~l^66) scaling to 100-bit, but SLOWER than GNFS at all relevant sizes
- Regev quantum: O(n^{3/2}) gates, proven correct (Pilatte 2024), but requires quantum hardware
- No classical algorithm has beaten GNFS in 33 years

**Our experimental results:**
- Trace-lattice: Best result, factors up to 48-bit semiprimes in ~10s
- Conductor sampling: Hit sqrt(N) density barrier at 36-bit
- TTN/MPS: Implementation too slow (timeout at 16-bit)
- Chebotarev: Too slow, no factors found
- Class number: O(1) for discriminant analysis, but H(D) computation is O(|D|^{1/2})

**Key research insights:**
- Henry Cohn: "Only ~100 smart people have seriously tried. We're 2/3 of the way to polynomial."
- Pilatte's zero-density proof: Smooth representations exist with exploitable structure
- TNSS paper gap: They use TTN+OPES+LLL, we use MPS with no preprocessing
- Murru-Salvatori: Polynomial factoring if regulator of Q(sqrt(N)) is known

---

## The Six Prongs

### Prong 1: TNSS Rewrite (TTN + OPES + LLL)

**Goal**: Match the paper's 100-bit result, then push beyond.

**Crate**: Rewrite `tnss-factoring`

**Critical changes from current implementation:**

| Current | Target (matching paper) |
|---------|------------------------|
| MPS (1D chain) | TTN (binary tree) |
| Bond dim = 4-8 (fixed) | m = ceil(n^{2/5}) (scaled) |
| Random + DMRG | OPES sampling |
| No preprocessing | LLL basis reduction |
| Fixed scaling | Precision parameter c varied |

**Implementation steps:**

1. Schnorr lattice construction:
   - Factor base: first π primes
   - Basis matrix B_{f,c}: diagonal f(j) + scaled log-prime row
   - Precision parameter c controls CVP difficulty

2. LLL preprocessing:
   - Reduce basis with `lattice-reduction::lll` before optimization
   - This makes short vectors easier to find

3. TTN implementation:
   - Binary tree topology with depth = ceil(log2(n))
   - Leaf tensors: shape [2] (qubit states)
   - Internal tensors: shape [m, m, m] (left child, right child, parent bond)
   - Bond dimension m = ceil(n^{2/5})

4. OPES (Optimal Pair Extraction Sampling):
   - Sample from Boltzmann distribution of TTN Hamiltonian
   - Extract low-energy pairs as candidate short vectors
   - Much more efficient than full DMRG optimization

5. Smooth relation collection:
   - For each short vector, check if it encodes a smooth relation
   - Collect >= π + 1 relations
   - GF(2) Gaussian elimination to find null vectors
   - gcd(X+Y, N) extraction

**Parameter scaling (from paper):**
- Lattice rank: n ~ C₂ * l^μ / (log(C₁) - log(ρ_sr) + γ*log(l))^{ω/μ}
- Fitted: C₁=1.0, C₂=0.013, μ=1.61, ω=9.3
- Smoothness bound: π₂ = 2*n*l
- Bond dimension: m ~ n^{2/5}

**Verification**: Factor semiprimes at 16, 32, 48, 64, 80, 100, 128 bits. Plot time vs bits on log-log scale. Compare to paper's Figure 3.

---

### Prong 2: Pilatte Smooth Relations

**Goal**: Exploit zero-density estimates to find smooth relations faster than traditional sieving.

**Crate**: New `smooth-pilatte`

**Background**: Pilatte (2024) proved that elements of (Z/NZ)* can be written as short products of the first d small primes, where d = O(sqrt(n)). This means smooth representations EXIST with bounded exponent size. The proof uses zero-density estimates for L(s, chi).

**Implementation steps:**

1. Short product lattice:
   - Given N and factor base {p_1,...,p_d}, construct the lattice:
     L = { e ∈ Z^d : p_1^{e_1} * ... * p_d^{e_d} ≡ 1 (mod N) }
   - This is a lattice of smooth relations
   - LLL/BKZ on this lattice finds the shortest smooth relations

2. Structured sampling:
   - Instead of random sieving, use LLL-reduced basis vectors as "directions" to search
   - Each basis vector encodes a specific combination of small primes
   - Enumerate neighbors of basis vectors to find more smooth relations

3. Density-guided search:
   - Use the Canfield-Erdos-Pomerance theorem: probability that a random number ≤ x is y-smooth
   - Ψ(x,y)/x ≈ u^{-u} where u = log(x)/log(y)
   - Use this to choose optimal smoothness bounds adaptively

4. Integration:
   - Replace sieving in NFS with lattice-based smooth search
   - Measure: relations found per unit time vs traditional sieve
   - If faster, this directly improves NFS complexity

**Verification**: Compare smooth relation generation rate against traditional sieving for N from 48-bit to 128-bit. Plot relations/second vs N.

---

### Prong 3: Trace-Lattice + SAT Hybrid

**Goal**: Push past 48-bit using BigUint and SAT solvers.

**Crate**: Extend `trace-lattice`

**Phase 1: BigUint extension**
- Port dim_s2(), dim_s2_new(), Kronecker symbols to BigUint
- This removes the 52-bit overflow barrier
- Expected: factor up to 64-bit with just this change

**Phase 2: Additional spectral invariants**
- Sturm bound: For f ∈ S_2(Γ_0(N)), only need a_p for p ≤ N/12 to determine f
- Atkin-Lehner: The eigenvalues w_p and w_q determine the old/new decomposition
- Multiple weight formulas: dim S_k(Γ_0(N)) for k = 2, 4, 6 gives independent equations
- Modular dimension: dim S_2(Γ_0(N)) mod small primes m gives constraints without computing full dim

**Phase 3: SAT encoding**
- Variables: binary digits of p (n/2 bits)
- Constraints:
  1. N = p * q where q = N/p (multiplication)
  2. p + q ≈ S (from dimension formulas, S ≈ 12 * dim_old)
  3. p < q (WLOG)
  4. p is odd, q is odd (N is odd semiprime)
  5. Additional spectral constraints as unit-propagation hints
- Solver: Call CryptoMiniSat or Kissat via subprocess
- The spectral constraints narrow the search space from O(2^{n/2}) to O(2^{n/4}) or better

**Phase 4: Iterative refinement**
- Use trace-lattice to get approximate p+q
- Use SAT to find exact p in the neighborhood
- Each additional spectral invariant halves the SAT search space

**Verification**: Factor semiprimes at 48, 56, 64, 80, 96, 128 bits. Compare time to pure trace-lattice and pure SAT.

---

### Prong 4: Murru-Salvatori Continued Fractions

**Goal**: Implement L[1/2, 1.06] factoring, explore polynomial path via regulator.

**Crate**: New `cf-factor`

**Algorithm outline:**

1. Compute continued fraction expansion of sqrt(N):
   sqrt(N) = a_0 + 1/(a_1 + 1/(a_2 + ...))
   Convergents: p_k/q_k where p_k^2 - N*q_k^2 = (-1)^k * Q_k

2. For each k, the triple (Q_k, b_k, Q_{k+1}) is a quadratic form of discriminant 4N
   where b_k = 2*a_{k+1}*Q_{k+1} - b_{k-1}

3. Search for an AMBIGUOUS form: (a, 0, c) or (a, a, c) in the cycle
   An ambiguous form gives gcd(a, N) = factor

4. Gauss composition: Given forms f1 and f2, compute f1*f2
   This lets us "multiply" positions in the cycle, enabling baby-step/giant-step

5. Infrastructure distance: d(f) = sum of partial quotients to reach f
   The period length = 2*R where R is the regulator

6. If we know R (or an approximation), jump directly near ambiguous forms

**Connection to Prong 1 (class-number-oracle):**
- Our class number computation gives H(D) for D = -4N
- The regulator R and class number h are related: h*R ≈ sqrt(|D|)*L(1, chi_D)/π
- If we compute h (from class-number-oracle), we can estimate R
- This feeds the "polynomial if regulator known" path

**Implementation:**
1. BigUint continued fraction engine
2. Quadratic form cycle traversal
3. Gauss composition with Nucomp optimization
4. BSGS in the class group using infrastructure distance
5. Regulator estimation from class number
6. Full factoring pipeline: CF → forms → ambiguous search → gcd

**Verification**: Compare to SQUFOF on 32-64 bit semiprimes. Test regulator-assisted mode. Plot scaling.

---

### Prong 5: GNFS Pipeline

**Goal**: Production-quality NFS as baseline and optimization target.

**Crate**: Extend `classical-nfs`

**Missing components:**

1. Lattice sieving (the core NFS step):
   - Rational side: for each prime q in factor base, sieve {(a,b) : a + bm ≡ 0 (mod q)}
   - Algebraic side: for each prime ideal q above q, sieve {(a,b) : N_f(a,b) ≡ 0 (mod q)}
   - Implementation: line-by-line sieving with bucket sort (Pollard's method)
   - Parallelism: rayon for parallel sieve ranges

2. Large prime variation:
   - After sieving, cofactor the remaining value
   - Allow up to 2 large prime factors per side
   - Build a relation graph: nodes = large primes, edges = partial relations
   - Find cycles in the graph → complete relations
   - This typically 10x the number of useful relations

3. Block Lanczos over GF(2):
   - Standard matrix step for NFS
   - Sparse matrix-vector multiply over GF(2)
   - Block size = 64 (machine word)
   - Expected to find null vectors in O(w * N) where w = matrix weight, N = dimension

4. Square root:
   - From null vector, compute X = prod(a_i + b_i*alpha) and Y = prod(a_i + b_i*m)
   - X^2 = Y^2 (mod N)
   - Compute sqrt(X^2) in the number field Z[alpha]/f(alpha)
   - Use Montgomery's algorithm or Couveignes' algorithm

5. Parameter selection:
   - Polynomial degree d: 3 for N < 10^100, 4 for N < 10^150, 5 for larger
   - Factor base bounds: optimal B depends on N
   - Sieve area: A × B where A,B ≈ B^{1/2}

**Integration with other prongs:**
- Prong 2 (smooth relations) can replace sieving step
- Prong 3 (spectral data) provides side information
- This is the reference implementation to beat

**Verification**: Factor 60-bit, 70-bit, 80-bit, 90-bit semiprimes. Compare timing to YAFU/CADO-NFS if available.

---

### Prong 6: Regev Classicalization

**Goal**: Approximate Regev's quantum sampling step classically.

**Crate**: New `regev-classical`

**Regev's algorithm (simplified):**
1. Choose d = O(sqrt(n)) small primes p_1,...,p_d
2. QUANTUM: For each i, compute g^{p_i^{x_i}} mod N in superposition, QFT, measure
3. Get d noisy samples v_i ≈ k_i/r + noise where r = ord(g) in (Z/NZ)*
4. CLASSICAL: Run LLL on the lattice {z ∈ Z^d : sum z_i*v_i ∈ Z} to recover r
5. From r | lcm(p-1, q-1), extract factors

**Classical approximation strategies:**

1. Random walk sampling:
   - For random a, compute ord(a) via BSGS in O(N^{1/4})
   - This gives one sample of r (partial order)
   - Need d = O(sqrt(n)) such samples
   - Total: O(sqrt(n) * N^{1/4}) — exponential but interesting if samples are correlated

2. Smooth subgroup probing:
   - Compute a^{B!} mod N for B-smooth bound B
   - If result ≡ 1 (mod p) but not (mod q), gcd reveals factor
   - This IS Pollard p-1, but viewed through Regev's lens
   - Can we choose B-values guided by the lattice structure?

3. Birthday collision in exponent lattice:
   - Generate many random products g^{e_1*p_1} * ... * g^{e_d*p_d} mod N
   - Look for collisions (same value mod N means exponents differ by a multiple of r)
   - Expected: O(sqrt(r)) collisions needed — still exponential
   - But with structured enumeration (Schnorr-style), may need fewer

4. Tensor network sampling:
   - Use Prong 1's TTN to approximately sample from the quantum distribution
   - Each TTN sample is a noisy version of what Regev's quantum circuit produces
   - Feed to LLL post-processing

5. Hybrid: classical samples + LLL
   - Even imperfect classical samples narrow the lattice search
   - LLL can tolerate some noise in the input vectors
   - Key experiment: what noise level is tolerable?

**Verification**: Test each classical sampling method. For N where we know ord(g), measure sample quality. Plot: number of samples needed vs N for successful LLL recovery.

---

## Implementation Priority and Parallelism

**Wave 1 (Parallel):**
- Prong 1 (TNSS rewrite) — highest priority, clear gap to close
- Prong 3 Phase 1 (BigUint trace-lattice) — quick win, extend 48→64 bit
- Prong 4 (CF-factor basics) — independent, well-understood math

**Wave 2 (After Wave 1 benchmarks):**
- Prong 5 (NFS pipeline) — establishes baseline
- Prong 2 (Pilatte smooth) — needs NFS as integration target
- Prong 3 Phase 2-3 (spectral invariants + SAT)

**Wave 3 (After Wave 2 results):**
- Prong 6 (Regev classicalization) — most speculative, needs other prongs' infrastructure
- Cross-prong integration experiments

## Success Criteria

| Prong | Minimum Success | Stretch Goal |
|-------|----------------|-------------|
| 1. TNSS | Factor 64-bit | Factor 128-bit |
| 2. Pilatte | 2x smooth relation speedup | 10x speedup over sieving |
| 3. Trace+SAT | Factor 96-bit | Factor 256-bit |
| 4. CF-factor | Match SQUFOF | Beat SQUFOF by 2x |
| 5. NFS | Factor 80-bit | Factor 120-bit |
| 6. Regev | Any classical sample works | Subexponential classical Regev |

## Decision Framework

After each wave:
- **Positive scaling**: Double down, extend to larger sizes
- **Negative scaling**: Document barrier precisely, focus resources elsewhere
- **Cross-prong synergy found**: Prioritize the combination
