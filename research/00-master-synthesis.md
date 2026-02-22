# Master Synthesis: Cross-Field Connections & Attack Vectors

## Introduction

This document is the central synthesis of a multi-disciplinary research program based on a single thesis: **revolutionary discoveries across mathematics, physics, computer science, and artificial intelligence may contain overlooked connections that, when combined, could yield fundamentally novel approaches to integer factorization.**

Integer factorization -- specifically, factoring the product of two large primes -- is the hardness assumption underlying RSA, one of the most widely deployed cryptographic systems in history. The best known classical algorithm (the General Number Field Sieve) runs in sub-exponential but super-polynomial time. Shor's quantum algorithm factors in polynomial time but requires fault-tolerant quantum computers that do not yet exist at cryptographic scale. Between "classically hard" and "quantum easy," there is a vast unexplored landscape.

This research program surveys that landscape. We examine:

- **The Langlands program** and its promise of computational access to L-functions and prime distribution
- **Quantum algorithms** (Regev, Shor) and their classical lattice analogues (Schnorr)
- **AI-driven algorithm discovery** (AlphaEvolve, evolutionary search)
- **Optimization formulations** (QUBO, Ising models, simulated annealing)
- **Compression theory** (Kolmogorov complexity, neural compression, MLA)
- **Physics dualities** (S-duality, mirror symmetry, AdS/CFT)
- **Neural architectures** (transformers, SSMs, Mamba, MoE)
- **Group theory** (Galois theory, HSP, representation theory)
- **Multi-base number representations** (primorial base, RNS, cross-base entropy)

No single field is likely to crack factoring alone. But the boundaries between fields are where the most surprising connections hide. A compression technique from vision AI (DeepSeek MLA) turns out to use the same mathematical operation (low-rank projection) as a quantum factoring algorithm (Regev). A duality theorem from sequence modeling (Mamba-2 SSD) raises the question of whether sequential computations like trial division have undiscovered parallel forms. The Langlands program -- the deepest unifying vision in pure mathematics -- connects the group theory of (Z/nZ)* to automorphic forms and L-functions, potentially giving computational access to prime distribution patterns.

The following ten "Attack Vectors" represent the most promising cross-field connections we have identified. Each is rated by feasibility and potential impact at the end of this document.

---

## Attack Vector 1: Langlands -> L-functions -> Prime Distribution -> Factorization Structure

**The Chain of Logic:**

The Langlands correspondence (now partially proven via the geometric Langlands program, Fargues-Scholze, etc.) establishes deep connections between:
- **Galois representations**: How the absolute Galois group Gal(Q-bar/Q) acts on algebraic structures
- **Automorphic forms**: Highly symmetric functions on algebraic groups
- **L-functions**: Analytic functions encoding arithmetic data (generalizing the Riemann zeta function)

For RSA factoring, the relevant L-functions are Dirichlet L-functions L(s, chi) where chi is a character of (Z/nZ)*. These L-functions encode the distribution of primes in arithmetic progressions modulo n. Their zeros determine, via explicit formulas, the exact distribution of primes near n.

**The Speculative Attack:**

If the Langlands correspondence gives computational access to the zeros of L-functions associated to (Z/nZ)*, we might be able to:
1. Compute the distribution of primes congruent to various residues modulo n
2. Detect asymmetries in this distribution that are caused by n = pq (since the distribution modulo n decomposes, via CRT, into distributions modulo p and modulo q)
3. Reconstruct p and q from these distributional asymmetries

**Feasibility Assessment:**

The Langlands program is advancing rapidly (geometric Langlands was proven in 2024), but the connection to computational algorithms is tenuous. Computing L-function zeros requires knowing the character group of (Z/nZ)*, which requires knowing phi(n), which requires knowing the factorization -- a circularity. However, there may be ways to extract partial information about L-functions without full knowledge of the group structure, and recent work on p-adic L-functions and Iwasawa theory provides alternative computational approaches.

**Key References:** See research notes on Langlands program (files 01-03 in this series).

---

## Attack Vector 2: Regev's Lattice Quantum Factoring <-> Schnorr's Classical Lattice Factoring

**The Connection:**

Both Oded Regev's 2023 quantum factoring algorithm and Claus-Peter Schnorr's 2021 classical factoring claim use the same mathematical framework: high-dimensional lattice geometry. Specifically, both construct a lattice from the smooth relations of a semiprime n, and attempt to find short vectors in this lattice that correspond to factorizations.

- **Schnorr's approach** (2021): Constructs a lattice from primes <= B and claims that LLL/BKZ lattice reduction can find sufficiently short vectors to factor n. The claim is that for appropriate parameters, the shortest vector encodes the factorization. This was shown to fail for cryptographic-size numbers: the required lattice dimension grows too fast, and the gap between the shortest vector and the factoring-relevant vector is too large for classical lattice reduction.

- **Regev's approach** (2023): Uses quantum computation to construct a MUCH higher-dimensional lattice (dimension sqrt(n) vs. Schnorr's polylog(n)) and uses quantum superposition to find short vectors. The quantum part is essential: it allows constructing the high-dimensional lattice without exponential classical resources.

**The Key Question:** What exactly does quantum computation add? Regev's algorithm uses quantum mechanics for two things: (1) preparing superpositions over smooth relations (which classically requires exponential search), and (2) running the quantum Fourier transform to extract periodicity. Can either of these be partially de-quantized?

**Feasibility Assessment:**

Recent work on "quantum-inspired" classical algorithms (Tang, 2019; Chia et al., 2020) has de-quantized several quantum speedups for linear algebra problems. However, these results apply to problems where the input is given as a low-rank matrix, which is not directly the case for factoring. The lattice structure of the factoring problem may be amenable to classical techniques (e.g., lattice sieving, BKZ 2.0) if the right formulation is found. The gap between Schnorr's failure and Regev's success is precisely the gap that needs to be understood.

**Key References:** See research notes on quantum algorithms and lattice methods.

---

## Attack Vector 3: AlphaEvolve-Style AI Search for Novel Factoring Algorithms

**The Approach:**

AlphaEvolve (Google DeepMind, 2025) uses LLM-guided evolutionary search to discover novel algorithms. It broke Strassen's 56-year-old record for matrix multiplication complexity, finding a new algorithm for multiplying 4x4 matrices using fewer multiplications than previously known. The system works by:
1. Representing algorithms as code
2. Using an LLM to propose mutations (code modifications)
3. Evaluating mutations on test cases
4. Selecting the best-performing variants
5. Iterating

**Application to Factoring:**

Use AlphaEvolve (or a similar system) to evolve factoring algorithms. The search space is:
- **Seed algorithms**: Trial division, Pollard's rho, Pollard's p-1, quadratic sieve
- **Mutations**: Change iteration formulas, add/remove steps, modify parameters, combine subroutines
- **Fitness function**: Number of semiprimes factored per unit time on a test set of increasing difficulty

This is arguably the most promising near-term attack vector because:
1. It doesn't require new mathematics -- it searches for combinations of known techniques
2. It has a clear, measurable objective (factor more numbers faster)
3. It has a proven track record (AlphaEvolve's Strassen result)
4. The search space is rich but structured (factoring algorithms have known building blocks)

**Feasibility Assessment:**

High feasibility for finding improved heuristics and constant-factor improvements. Low feasibility for finding asymptotic breakthroughs (moving from L(1/3) to L(1/4) complexity). The reason: evolutionary search is good at optimizing within a paradigm but rarely discovers paradigm shifts. However, even heuristic improvements could be valuable for specific ranges of numbers, and the search might discover useful building blocks that humans can then analyze and generalize.

**Key References:** AlphaEvolve paper; FunSearch (DeepMind, 2023); AI for mathematics survey.

---

## Attack Vector 4: QUBO/Ising Formulation -- Factoring as Optimization

**The Approach:**

Integer factorization can be encoded as a Quadratic Unconstrained Binary Optimization (QUBO) problem. Write n = p * q where p and q are represented in binary: p = sum_i p_i * 2^i, q = sum_j q_j * 2^j. The constraint n = p * q becomes a system of quadratic equations in binary variables {p_i, q_j}. This can be mapped to:
- **QUBO matrix**: Minimize x^T Q x where Q encodes the multiplication constraints
- **Ising Hamiltonian**: H = -sum_{ij} J_{ij} * s_i * s_j - sum_i h_i * s_i where the ground state corresponds to the factorization

**Implementation Paths:**

1. **D-Wave quantum annealing**: Can solve QUBO instances directly on quantum hardware. Limited to ~5000 qubits currently, sufficient for factoring numbers up to ~20-30 bits. Proof of concept only.
2. **Simulated annealing**: Classical simulation of the Ising model. Can handle larger instances but struggles with the rugged energy landscape of factoring QUBO.
3. **Parallel tempering**: Run multiple replicas at different temperatures and swap configurations. Significantly improves exploration of the energy landscape.
4. **Simulated bifurcation**: Model the QUBO as a network of coupled oscillators near a bifurcation point. Recently shown to outperform simulated annealing on some QUBO instances.

**Feasibility Assessment:**

Medium feasibility. The QUBO encoding is straightforward and has been demonstrated on small numbers. The challenge is scaling: the energy landscape of the factoring QUBO has exponentially many local minima (corresponding to "almost factorizations"), and classical optimization algorithms get trapped. Quantum annealing might help by tunneling through barriers, but current hardware is too small. The most promising classical approach is parallel tempering with a well-designed temperature schedule, potentially guided by ML-predicted landscape features.

**Key References:** Dridi & Alghassi, 2017; Peng et al., 2019 (D-Wave factoring); simulated bifurcation literature.

---

## Attack Vector 5: Compression Insight -- Kolmogorov Complexity and Hidden Structure

**The Core Insight:**

A semiprime N = p * q has low Kolmogorov complexity. It is fully determined by two numbers (p and q), each of roughly n/2 bits, so K(N) <= n/2 + n/2 + O(log n) = n + O(log n). But N appears random: it passes statistical randomness tests, has roughly equal 0s and 1s in binary, and shows no obvious patterns. The gap between low intrinsic complexity and high apparent complexity is exactly the gap that factoring exploits.

**The Speculative Attack:**

Can compression algorithms detect this hidden structure?

1. **Lossless compression**: Standard compressors (gzip, zstd) treat N as a byte string and cannot compress it -- the structure is too deep. But what about compressors that understand arithmetic? A compressor with access to modular arithmetic operations might detect that N has a compact representation as a product.

2. **Neural compression**: Train a neural compressor on semiprimes. If the compressor learns to encode semiprimes more efficiently than random numbers of the same size, it has detected the factor structure. The compression model's internal representations might encode factor-relevant features.

3. **Kolmogorov complexity approximation**: Use program synthesis (a la Levin search) to find short programs that generate N. The shortest program is "print p * q," and finding this program IS factoring.

**Feasibility Assessment:**

Low-to-medium feasibility. The fundamental obstacle is that Kolmogorov complexity is uncomputable, and practical compressors are far from the Kolmogorov limit. However, the neural compression approach is testable: train an autoencoder on semiprimes and random numbers, and measure whether the learned latent representations differ. If they do, the model has detected factor structure. This connects directly to the MLA compression idea (Attack Vector 7).

**Key References:** Li & Vitanyi, *An Introduction to Kolmogorov Complexity*; neural compression literature; DeepSeek MLA analysis (file 09).

---

## Attack Vector 6: Physics Duality Speculation -- Is There a "Dual" Description of Factoring?

**The Motivation:**

Modern theoretical physics is built on dualities -- exact equivalences between seemingly different physical theories:
- **S-duality**: Swaps strong and weak coupling. A problem that is intractable at strong coupling becomes easy at weak coupling in the dual theory.
- **Mirror symmetry**: Swaps complex geometry and symplectic geometry. A hard computation on one Calabi-Yau manifold becomes easy on its mirror.
- **AdS/CFT**: A gravitational theory in (d+1) dimensions is equivalent to a non-gravitational quantum field theory in d dimensions. Hard gravitational computations (black holes) map to easier field theory computations (thermal states).
- **Mamba-2 SSD**: Sequential recurrence <-> parallel matrix operation (this is a mathematical duality, not physics, but it has the same structure).

**The Speculative Question:**

Is there a "dual" formulation of factoring where the problem that is hard (finding factors of a product) becomes easy?

Several candidates for such a duality:
1. **Fourier duality**: Shor's algorithm IS a Fourier duality result -- the periodicity of a^x mod n is hidden in the time domain but visible in the frequency domain. The QFT reveals it. Is there a classical analogue?
2. **Modular/geometric duality**: By the theory of schemes, Z/nZ is a geometric object (Spec(Z/nZ) is a two-point space {(p), (q)}). Could geometric methods in algebraic geometry detect these "points"?
3. **Information-theoretic duality**: The mutual information I(N; p) = H(p) (knowing N determines p up to two choices). But computing this mutual information is as hard as factoring. Is there a dual quantity that's easier to compute?
4. **Computational duality**: The SSD result shows sequential <-> parallel duality for certain computations. Is there a duality between "multiply" (easy) and "factor" (hard)?

**Feasibility Assessment:**

Very low feasibility in the near term -- this is the most speculative attack vector. However, it has the highest potential payoff: a genuine duality would transform factoring from hard to easy, not just improve it incrementally. The most concrete sub-direction is the algebraic geometry approach (Spec(Z/nZ)), which connects to the Langlands program and might yield computationally useful insights.

**Key References:** Polchinski, *String Theory*; Kontsevich, *Homological Algebra of Mirror Symmetry*; SSD/Mamba-2 (file 11).

---

## Attack Vector 7: DeepSeek MLA Compression <-> Factorization

**The Connection:**

DeepSeek's Multi-Head Latent Attention (MLA) compresses high-dimensional key-value representations into low-rank latent vectors. This is mathematically the same operation as what lattice-based factoring algorithms do: project high-dimensional number-theoretic data into a lower-dimensional space where structure becomes visible.

**The Proposed Experiment:**

1. **Data representation**: Encode each semiprime N as a high-dimensional vector: v(N) = [N mod 2, N mod 3, N mod 5, ..., N mod p_k, binary digits, base-6 digits, quadratic residue symbols, ...]

2. **MLA-style compression**: Train a neural network with an MLA-like architecture where:
   - The "query" is the target: "what are the factors?"
   - The "keys" are the number-theoretic features of N
   - The "values" are the features of known factorizations
   - The latent compression projects all of this into a low-dimensional space

3. **Analysis**: If the model learns to compress effectively (low reconstruction error on factor-correlated quantities), analyze the learned projection matrices. They encode which linear combinations of number-theoretic features are most informative for factoring.

4. **Extraction**: Use the learned projections to construct a new classical algorithm: compute the informative linear combinations for a target N, and use them to constrain the factor search.

**Feasibility Assessment:**

Medium feasibility. The experiment is straightforward to implement and can be validated on small semiprimes. The key risk is that the learned projections might not generalize: the compression that works for 32-bit semiprimes might be completely different from what works for 64-bit semiprimes, because the number-theoretic structure changes with scale. However, even a negative result would be informative: if the model CANNOT compress semiprime data efficiently despite the low intrinsic dimensionality (just 2 numbers), this tells us something deep about the hardness of factoring.

**Key References:** DeepSeek MLA (file 09); lattice methods; neural compression.

---

## Attack Vector 8: SSM-Attention Duality <-> Sequential-Parallel Duality in Factoring

**The Insight:**

The Mamba-2 State Space Duality theorem proves that sequential recurrence and parallel matrix operation compute the same function for structured linear systems. This is a rigorous example of a computation that appears inherently sequential (process one element at a time, maintaining state) but has an exact parallel equivalent (multiply a matrix by a vector).

**Application to Factoring:**

Factoring by trial division is the archetypal "sequential" algorithm: test divisors 2, 3, 5, 7, 11, ... checking if each divides N. This sequential process can be partially parallelized (test many divisors simultaneously), but the search space is exponential.

The SSD perspective suggests a different question: **is there a matrix M such that M * v(N) directly encodes the factors?** In the SSM framework:
- The "input sequence" is the sequence of candidate divisors d_1, d_2, ...
- The "state" after processing d_i is some accumulation of information about N's divisibility
- The "output" at each position is whether d_i divides N

If this can be formulated as a linear SSM, then the SSD theorem guarantees a parallel matrix form. The matrix would be the "factoring matrix" -- multiply it by the input and read off the factors.

**Feasibility Assessment:**

Low feasibility because modular arithmetic (N mod d) is nonlinear, and the SSD theorem applies to linear recurrences. However, modular arithmetic can be linearized in certain representations (e.g., via the Chinese Remainder Theorem, or by working in a number-theoretic transform domain). If the right linearization exists, SSD applies and gives a parallel factoring algorithm. This is deeply speculative but mathematically precise enough to be testable.

**Key References:** Mamba-2 SSD (file 11); number-theoretic transforms; CRT representations.

---

## Attack Vector 9: Group Theory as Unifying Framework

**The Thesis:**

Every approach to factoring is, at its core, a group-theoretic statement about (Z/nZ)*. By understanding the group theory deeply enough, we can see connections between seemingly unrelated approaches and potentially find new ones.

**The Unification:**

| Approach | Group-Theoretic Statement |
|----------|--------------------------|
| RSA security | Determining the structure of (Z/nZ)* ~ Z/(p-1)Z x Z/(q-1)Z is hard |
| Shor's algorithm | Quantum HSP for abelian groups solves this |
| Langlands | Automorphic representations of GL_n encode group structure |
| Lattice methods | Short vectors in lattices built from group relations reveal structure |
| Compression | Low-rank structure of the group's multiplication table can be compressed |
| QUBO/Ising | Ground state of a Hamiltonian encoding group constraints gives structure |
| AI search | Learning the group structure from samples |

**The Key Structural Facts:**

1. **(Z/nZ)* is abelian**: This is what makes Shor work. Abelian groups have efficient quantum Fourier transforms.
2. **(Z/nZ)* has a known order RANGE**: phi(n) is close to n, and phi(n) = n - p - q + 1. Knowing phi(n) exactly gives the factors.
3. **The group decomposes as a direct product**: (Z/pZ)* x (Z/qZ)*. This decomposition is the factorization.
4. **Element orders leak information**: The order of a random element in (Z/nZ)* is lcm(ord_p(a), ord_q(a)), which depends on the group decomposition.
5. **Characters (homomorphisms to C*) encode arithmetic**: Dirichlet characters mod n encode prime distribution. Characters that "see" the decomposition modulo p and modulo q separately contain factor information.

**Feasibility Assessment:**

High feasibility as a framework; variable feasibility for specific algorithms derived from it. The group-theoretic perspective has already yielded all known factoring algorithms. The question is whether the framework reveals NEW algorithms -- specifically, new efficiently computable group invariants that depend on the decomposition. The most promising directions are representation-theoretic (using character theory and L-functions) and cohomological (using group cohomology invariants).

**Key References:** Group theory (file 10); Langlands (files 01-03); all other files.

---

## Attack Vector 10: Multi-Base Pattern Analysis

**The Insight:**

The representation of a number matters. In base 10, the number 30 = 2 * 3 * 5 looks unremarkable. In "primorial base" (where digits represent coefficients of primorial products), 30 = 1 * 30 + 0, which is a single digit -- its factor structure is immediately visible. Different bases expose different structural properties:

- **Binary**: Efficient for computation but hides multiplicative structure
- **Base 6**: The product of the first two primes (2*3); semiprimes with small factors show patterns
- **Primorial base**: Digits directly encode relationships with small primes
- **Residue Number System (RNS)**: Represent N by its residues modulo several coprime moduli; arithmetic is componentwise
- **Balanced ternary**: Can expose signed-digit patterns relevant to factor search
- **Mixed-radix representations**: Custom bases designed to expose specific factor relationships

**The Proposed Analysis:**

1. **Cross-base entropy**: For a semiprime N, compute its Shannon entropy in each of several bases. Compare to the entropy of random numbers of the same size. Semiprimes might have anomalous entropy in bases related to their factors.

2. **Cross-base correlation**: Compute the mutual information between digit sequences of N in different bases. If base-b1 digits are unusually correlated with base-b2 digits (compared to random numbers), this indicates structure that could be exploited.

3. **ML feature engineering**: Use multi-base representations as features for a neural factoring model. Let the model learn which bases and which digit positions are most informative. The learned attention patterns would reveal which representations expose factor structure.

4. **Number-theoretic transforms**: The Number Theoretic Transform (NTT) is the Fourier transform in Z/pZ. Apply NTT to the digit sequence of N in various bases and look for spectral signatures of factorability.

**Feasibility Assessment:**

Medium feasibility. The experiments are computationally cheap and can be run at scale. The risk is that the signal-to-noise ratio is too low: for cryptographic-size semiprimes, any multi-base pattern is likely to be exponentially weak. However, for smaller numbers (say, 64-bit semiprimes), patterns might be detectable, and understanding them could guide the development of algorithms that scale.

**Key References:** Number representation theory; RNS arithmetic; NTT; multi-base ML features.

---

## Priority Ranking

The following ranking balances **feasibility** (can we actually do this with current technology?) against **potential impact** (how much would it advance factoring if successful?).

### Tier 1: High Feasibility, Meaningful Impact (Start Here)

| Rank | Attack Vector | Feasibility | Impact | Rationale |
|------|--------------|-------------|--------|-----------|
| **1** | **AV3: AlphaEvolve-style AI search** | High | Medium-High | Proven methodology (AlphaEvolve broke a 56-year record). Clear fitness function. Can be started immediately with existing tools. Even incremental improvements are publishable and useful. |
| **2** | **AV4: QUBO/Ising formulation** | Medium-High | Medium | Well-understood encoding. Simulated annealing + parallel tempering are mature. Can benchmark against known algorithms. Provides concrete empirical results. |
| **3** | **AV10: Multi-base pattern analysis** | Medium-High | Medium | Computationally cheap experiments. Clear experimental protocol. Even negative results are informative. Good foundation for ML-based approaches. |

### Tier 2: Medium Feasibility, High Impact (Invest Selectively)

| Rank | Attack Vector | Feasibility | Impact | Rationale |
|------|--------------|-------------|--------|-----------|
| **4** | **AV7: MLA compression <-> factorization** | Medium | High | Directly implementable experiment. Connects proven engineering (MLA) to factoring. Risk of non-generalization across scales, but the experiment itself is well-defined. |
| **5** | **AV9: Group theory as unifying framework** | Medium | High | Deep mathematical foundation. Every factoring algorithm is a group theory result. New group invariants = new algorithms. Requires mathematical expertise but has centuries of theory to build on. |
| **6** | **AV5: Compression insight** | Medium | Medium-High | Neural compression experiments are feasible. Kolmogorov complexity theory provides guidance. Connection to MLA (AV7) and multi-base analysis (AV10) creates synergies. |
| **7** | **AV2: Regev-Schnorr lattice gap** | Medium | High | Understanding what quantum adds to lattice factoring could reveal classical shortcuts. Requires deep mathematical analysis but the question is precisely stated. |

### Tier 3: Low Feasibility, Potentially Transformative (Long-Term Bets)

| Rank | Attack Vector | Feasibility | Impact | Rationale |
|------|--------------|-------------|--------|-----------|
| **8** | **AV8: SSM-Attention duality <-> factoring** | Low-Medium | Very High | The SSD theorem is proven, but applying it to nonlinear modular arithmetic is speculative. If a linearization exists, the payoff is enormous. |
| **9** | **AV1: Langlands -> L-functions -> factoring** | Low | Very High | The mathematics is deep and advancing rapidly, but the connection to computational algorithms is tenuous. Circularity issues (computing L-functions requires knowing the group). Long-term bet on pure mathematics. |
| **10** | **AV6: Physics duality speculation** | Very Low | Transformative | The most speculative vector. If a genuine factoring duality exists, it would be the most important algorithmic discovery in history. But there is no concrete evidence one exists, and the search space is unbounded. |

### Recommended Research Strategy

**Phase 1 (Months 1-3):** Execute AV3 (AI algorithm search), AV4 (QUBO optimization), and AV10 (multi-base analysis) in parallel. These are the most feasible and will generate concrete empirical results.

**Phase 2 (Months 3-6):** Based on Phase 1 results, pursue AV7 (MLA compression experiment) and AV5 (neural compression). Use multi-base features from AV10 as input representations.

**Phase 3 (Months 6-12):** Invest in AV9 (group theory framework) and AV2 (Regev-Schnorr gap analysis) as mathematical investigations. These require deeper expertise but connect to all other vectors.

**Ongoing:** Maintain AV1 (Langlands), AV6 (physics duality), and AV8 (SSD duality) as background reading and long-term theoretical directions. Monitor for breakthroughs in these fields that might create new opportunities.

The key insight is that these attack vectors are not independent -- they reinforce each other. Multi-base analysis (AV10) provides features for MLA compression (AV7). Group theory (AV9) provides the mathematical framework for understanding why any approach works. AI search (AV3) can discover combinations of techniques from all other vectors. The research program should be run as an interconnected portfolio, not as isolated projects.

---

## Experimental Findings

### Experimental Findings from Rust Implementation

1. **Quadratic Sieve Results**: QS successfully factors semiprimes up to ~40 bits with appropriate factor base bounds. Factor base bound selection is critical — too small and no smooth relations are found, too large and linear algebra becomes expensive. Parallel sieving via rayon provides near-linear speedup.

2. **ECM Stage 2 Effectiveness**: Stage 2 continuation significantly expands the range of factors found. For 48-bit semiprimes, Stage 2 with B2=2M finds factors that Stage 1 with B1=50K misses. Multi-curve parallelism (128 curves) provides high success probability.

3. **NFS Polynomial Selection**: Base-m polynomial selection works correctly — f(m) ≡ 0 (mod n) verified for all test cases. Rational-side sieving produces smooth relations. However, the simplified NFS only beats QS for sufficiently large inputs; for numbers under ~60 bits, QS is more reliable.

4. **Ducas Verification of Schnorr's Claim**: Our Rust implementation of Schnorr's lattice-based factoring confirms Ducas' finding — success rates near 0% at the parameter sizes Schnorr predicted. LLL reduction does shorten basis vectors, but the short vectors don't correspond to useful smooth relations. The Hermite factor after LLL is consistent with theoretical predictions.

5. **QUBO/Ising Factoring**: Proper carry-bit encoding produces QUBO matrices where the correct factorization has exactly zero energy. Parallel tempering successfully factors numbers up to ~8 bits (15, 77). Beyond that, the search space grows exponentially — the number of QUBO variables scales as O(p_bits * q_bits) for auxiliary product variables plus carry variables.

6. **Compression Differential Test**: Preliminary results show minimal compression ratio differences between semiprimes and random odd numbers of the same bit size. This suggests that semiprimes do not have easily exploitable compression structure — consistent with their high Kolmogorov complexity.

7. **Group Structure Analysis**: Sampling element orders in (Z/nZ)* and computing their LCM successfully recovers λ(n) for small semiprimes. The Pohlig-Hellman decomposition correctly breaks order computation into prime-power subgroups. Baby-step giant-step provides O(√order) discrete log computation in subgroups.

8. **MLA Latent Space Clustering**: Linear autoencoders trained on number-theoretic features show that reconstruction error decreases with training, but silhouette scores for factor-ratio-based clusters are modest. This suggests that linear projections alone don't capture enough factor structure — nonlinear approaches may be needed.

9. **Multi-Base Representation**: Cross-base anomaly detection shows that semiprimes occasionally exhibit statistically significant entropy differences from random numbers in specific bases, but the effect is small and base-dependent. RNS representation directly reveals small prime factors (N mod p = 0).

10. **Quantum-Inspired Classical**: Classical period finding works but is exponential — practical only for small numbers. The continued fraction post-processing (Shor's classical step) correctly extracts periods from simulated measurements. Grover speedup estimates confirm the quadratic advantage: for 64-bit search spaces, quantum would require ~2^32 evaluations vs classical ~2^64.

11. **SIQS Multi-Polynomial Sieving**: Self-Initializing Quadratic Sieve with Tonelli-Shanks modular square roots and CRT polynomial combination successfully factors 48-bit semiprimes. Multi-polynomial approach generates fresh sieving polynomials automatically, avoiding the single-polynomial bottleneck of basic QS. This confirms that polynomial switching is essential for scaling QS beyond small inputs.

12. **Brent's Cycle Detection**: Pollard's rho with Brent's improvement and batch GCD accumulation significantly reduces the number of expensive GCD operations. Power-of-2 step doubling with backtracking phase handles edge cases where batch GCD returns n. The batching strategy (accumulating products modulo n before computing GCD) provides a measurable constant-factor speedup over naive per-step GCD.

13. **Parallel Ensemble Factoring**: Racing multiple methods concurrently (6 Brent seeds + 2 p-1 bounds + trial division) with AtomicBool cooperative shutdown. First method to find a factor signals all others to stop. Practical for automated factoring where the optimal method isn't known in advance. This ensemble approach is the most reliable automated strategy tested -- it consistently finds factors regardless of input structure, since different methods exploit different factor properties (small factors via trial division, smooth p-1 via Pollard p-1, general factors via rho).

14. **Evolutionary Algorithm Search (AlphaEvolve-style)**: Genetic programming with a DSL of 9 composable factoring primitives (mod_pow, gcd, random_element, iterate, accumulate_gcd, subtract_gcd, square, add_const, multiply_mod) successfully evolves programs that factor small semiprimes. Seed programs encoding Pollard's rho, trial division, and Fermat's method provide baseline fitness. Evolution via tournament selection, subtree crossover, and mutation improves fitness over generations. The approach confirms AV3's thesis that algorithmic search over factoring strategies is tractable -- the question is whether evolved programs can discover genuinely novel heuristics beyond recombinations of known techniques.

15. **SSD Linearization Experiment**: All three linearization strategies (binary indicator lifting, NTT domain, CRT decomposition) correctly compute N mod d for all divisors, confirming mathematical equivalence between sequential and parallel forms. However, none provides a practical speedup. The fundamental insight: trial division's bottleneck is NOT sequential dependency (each N mod d is already independent) but the exponential number of candidate divisors. The SSD theorem addresses the wrong bottleneck. This is a valuable negative result that formalizes why factoring resists the sequential-to-parallel transformation that works for sequence models.

### Revised Attack Vector Rankings (Post-Experimental)

Ranked by feasibility based on experimental evidence (updated with SIQS, Brent, and ensemble findings):

1. **Parallel ensemble factoring** — Most practical automated strategy for unknown inputs. Racing multiple methods (Brent rho variants, p-1, trial division) with cooperative shutdown reliably factors numbers regardless of structure. Eliminates the need to guess which method suits a given input.
2. **AlphaEvolve-style AI search** — Most promising near-term for discovering novel combinations. Our KNN predictor and weighted features show ML can learn factor-correlated patterns at small scale. The question is scaling.
3. **SIQS / Quadratic Sieve** — Multi-polynomial SIQS extends QS to 48-bit+ semiprimes reliably. Polynomial switching is the key enabler. Best classical method for mid-range semiprimes below the NFS crossover.
4. **Quantum factoring (Regev/Gidney)** — Most promising long-term. Our classical simulations confirm the algorithm structure works; only qubit count limits it.
5. **ECM for finding small factors** — Practical today. Stage 2 significantly extends range. Best for numbers with one small factor.
6. **GNFS improvements** — Incremental improvements possible. Our simplified NFS confirms the polynomial selection + sieve + LA pipeline works.
7. **Lattice-based factoring** — Debunked for RSA sizes. Experimentally confirmed Ducas' finding.
8. **QUBO/Ising formulation** — Scales poorly. Correct in principle but exponential in practice.
9. **Group structure exploitation** — Requires knowing order of (Z/nZ)*, which is as hard as factoring.
10. **Compression-based detection** — No significant signal found. Semiprimes are effectively random at the byte level.
11. **MLA/latent space** — Linear projections insufficient. Nonlinear architectures needed.
12. **Multi-base anomaly detection** — Weak signals only. Not a viable attack vector.

### Phase 2 Experimental Findings (5 New Experiments, Feb 2026)

16. **Class Number Oracle (Eichler-Selberg)**: Discriminant analysis runs in O(1) time (3-16us for all sizes 16-64 bit) because it only depends on fixed small primes l=2,3,5. However, H(D) computation (the actual bottleneck) scales as O(|D|^{1/2}) for exact method. Shanks BSGS is SLOWER than exact for |D| < 500K due to hash table overhead. For Eichler-Selberg to work as factoring, we need H(D) at D ~ N^2, making both methods impractical. Verdict: H(D) bottleneck kills the spectral path.

17. **Sublinear Conductor Detection**: All 4 sampling methods (random, subgroup_chain, cross_correlation, conductor_witness) fail at 36+ bits with 5000 samples. The fundamental barrier: characters with conductor < N have density ~1/sqrt(N) in (Z/NZ)*. Random sampling works up to 32-bit. One lucky hit at 48-bit (213 samples). No method achieves sublinear detection. Verdict: density barrier confirmed, no shortcut found.

18. **TTN vs MPS Tensor Networks**: Both MPS and TTN timeout at 10 seconds even for 16-bit semiprimes. Root cause: our implementation uses fixed bond dimension 4, no LLL preprocessing, no OPES sampling, and MPS instead of TTN. The TNSS paper (Tesoro/Siloi 2024) factors 100-bit using TTN with bond dim ~n^{2/5}, LLL-reduced lattice, and OPES sampling. Verdict: implementation gap, not theoretical barrier.

19. **Trace Formula Lattice Attack**: Successfully factors semiprimes up to 48-bit in ~10 seconds. Uses dimension formulas for S_2(Gamma_0(N)) (which give dim_old = (p+q)/12 + corrections) and LLL lattice reduction. Fails at 52+ bits due to u64 overflow. The dimension space grows as N/12 (linear), making full space computation O(N). But the factoring method operates on dimension formulas, not the full space. Verdict: most promising result, needs BigUint extension.

20. **Chebotarev Density Discovery**: Extremely slow (15.7s for 16-bit, 217.5s for 20-bit = ~14x per 4 bits). Does NOT find factors. Element order computation via BSGS is O(N^{1/4}) per element, and statistical signal from order densities is too weak. Verdict: not viable.

### Research Survey Update (2024-2026 Literature)

Key papers discovered:
- **Tesoro & Siloi (Oct 2024)**: TNSS factors 100-bit RSA with polynomial scaling ~l^66. Slower than GNFS at all relevant sizes. Uses TTN, OPES, LLL preprocessing.
- **Pilatte (Apr 2024)**: Proved Regev's number-theoretic conjecture unconditionally using zero-density estimates. Potential classical smooth number implications.
- **Ragavan-Vaikuntanathan (CRYPTO 2024)**: Regev with O(n log n) qubits matching Shor's space.
- **Murru & Salvatori (Sep 2024)**: CF+QF factoring at L[1/2, 1.06], polynomial if regulator known.
- **IWSEC 2024 (Sato et al.)**: Lattice factoring confirmed exponential up to 90-bit.
- **Al-Hasso (Oct 2025)**: Probabilistic CVP with 100x fewer lattice instances.
- No classical algorithm has beaten GNFS L[1/3, 1.923] in the 2024-2026 period.

### Revised Rankings (Post-Phase 2)

1. **Trace-lattice factoring** — Best empirical result (48-bit). Needs BigUint + SAT hybrid.
2. **TNSS with proper TTN** — Paper shows 100-bit possible. Implementation gap to close.
3. **GNFS pipeline** — The baseline to beat. Our implementation incomplete.
4. **Pilatte smooth relations** — Unexplored classical potential from Regev proof.
5. **Murru-Salvatori CF** — New approach, polynomial if regulator known.
6. **Regev classicalization** — Most speculative but highest theoretical payoff.
7. **Parallel ensemble** — Most practical for automated factoring.
8. **ECM** — Practical for small factors, doesn't help with balanced semiprimes.
9. Everything else — confirmed dead ends or marginal improvements.
