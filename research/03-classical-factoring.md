# Classical Factoring Algorithms

An annotated bibliography of classical (non-quantum) integer factorization algorithms, including hybrid quantum-classical proposals and claimed breakthroughs that have been scrutinized or debunked.

---

### 1. General Number Field Sieve (GNFS)
- **Authors**: John Pollard (concept, 1988); Hendrik Lenstra, Carl Pomerance, Arjen Lenstra, and others (development and analysis, 1990-1993)
- **Year**: 1990-1993 (developed); ongoing (implementations)
- **Source**: [Lenstra, Lenstra, Manasse, Pollard (1993)](https://doi.org/10.1090/S0025-5718-1993-1197831-4); [Pomerance survey](https://math.dartmouth.edu/~carlp/PDF/paper130.pdf)
- **Core Idea**: The General Number Field Sieve is the fastest known classical algorithm for factoring integers larger than about 100 digits. It operates in two phases: (1) **sieving** — finding pairs (a, b) such that both a + b*m (on the rational side) and the norm of a + b*alpha (on the algebraic side) are smooth (factor only over small primes), using a carefully chosen number field Q(alpha) where f(alpha) = 0 for a polynomial f with f(m) = 0 mod N; (2) **linear algebra** — combining these relations via Gaussian elimination over GF(2) to find a square congruence x^2 = y^2 (mod N), yielding factors via gcd(x-y, N). The heuristic complexity is L_N(1/3, c) where c = (64/9)^{1/3} ~ 1.923 and L_N(s, c) = exp(c * (ln N)^s * (ln ln N)^{1-s}).
- **Mathematical Foundation**: Algebraic number theory (number fields, rings of integers, ideal factorization), lattice sieving, the Coppersmith-Howgrave-Graham method for polynomial selection, structured Gaussian elimination over sparse matrices (Block Lanczos or Block Wiedemann algorithms), and smoothness probability estimates from analytic number theory (the Dickman rho function).
- **RSA Relevance**: This is the primary threat to RSA from classical computing. The largest RSA-type number factored by GNFS is RSA-250 (250 decimal digits, 829 bits), completed in February 2020 using approximately 2,700 core-years of computation. Extrapolating, RSA-2048 (617 digits) would require approximately 10^{26} operations, far beyond current or foreseeable classical computing capacity. GNFS complexity grows sub-exponentially but is super-polynomial, which is why RSA remains secure against classical attacks at sufficient key sizes.
- **Status**: Proven (heuristic complexity; rigorous proof under certain number-theoretic assumptions)
- **Open Questions**: Can the constant c = (64/9)^{1/3} be improved? Is there a classical algorithm with complexity L_N(s, c) for s < 1/3? Can GNFS be meaningfully parallelized on modern GPU/TPU architectures? Would a proof of the Riemann hypothesis improve the smoothness bounds used in the analysis?

---

### 2. CADO-NFS: Open-Source GNFS Implementation
- **Authors**: INRIA team (Paul Zimmermann, Emmanuel Thome, Pierrick Gaudry, and others)
- **Year**: 2008 -- present (actively maintained)
- **Source**: [GitHub — cado-nfs](https://github.com/cado-nfs/cado-nfs); [CADO-NFS documentation](https://cado-nfs.gitlabpages.inria.fr/)
- **Core Idea**: CADO-NFS (Crible Algebrique: Distribution, Optimisation — Number Field Sieve) is the premier open-source implementation of the General Number Field Sieve. It handles all phases of GNFS: polynomial selection (using Kleinjung's algorithms), lattice sieving (with optimized siever supporting large-prime variations), filtering (merging and structured Gaussian elimination), linear algebra (Block Wiedemann over GF(2) with MPI parallelism), and square root computation. It has been used for most record-breaking factorizations, including RSA-250.
- **Mathematical Foundation**: Same as GNFS (entry 1), plus extensive algorithmic engineering: ECM-based cofactorization, bucket sieving for cache efficiency, the sublattice sieving technique, optimal parameter selection guided by Murphy's E-value for polynomial quality, and distributed computing via task scheduling.
- **RSA Relevance**: CADO-NFS is the tool that would be used to factor RSA keys classically. Its efficiency directly determines the practical security margin of RSA. The codebase is highly optimized for modern hardware (multi-core, MPI clusters) and represents decades of algorithmic engineering. Any improvement to GNFS theory would likely be implemented in CADO-NFS first. Understanding its performance characteristics (sieving speed per core, linear algebra memory requirements) is essential for estimating RSA security margins.
- **Status**: Proven (mature, production-quality software)
- **Open Questions**: Can GPU acceleration significantly speed up the sieving phase? What is the practical limit of distributed GNFS (communication overhead vs. computation)? Can machine learning be used to optimize polynomial selection or sieving parameters?

---

### 3. Schnorr's Lattice-Based Factoring Claim
- **Authors**: Claus-Peter Schnorr
- **Year**: 2021
- **Source**: [ePrint 2021/232](https://eprint.iacr.org/2021/232)
- **Core Idea**: Schnorr claimed a polynomial-time factoring algorithm based on lattice reduction. The proposed method constructs a lattice from the first several primes and the target number N, then uses SVP (Shortest Vector Problem) solvers to find short vectors that yield factoring relations. Schnorr claimed this "destroys the RSA cryptosystem" by providing a polynomial-time factoring algorithm. However, the analysis was flawed: the claimed approximation factor for the SVP solution was too optimistic, and the lattice dimensions required for the method to work grow polynomially in theory but are infeasible in practice.
- **Mathematical Foundation**: Lattice reduction (LLL, BKZ algorithms), the Shortest Vector Problem (SVP), the connection between short lattice vectors and smooth numbers, and Babai's nearest plane algorithm. The approach attempts to use the structure of the prime number lattice to find smooth relations, bypassing the sieving step of GNFS.
- **RSA Relevance**: **Debunked.** Leo Ducas (CWI Amsterdam) and others conducted thorough experimental and theoretical analyses. Ducas showed that for randomly generated composites of cryptographic size, Schnorr's method produced 0 valid factoring relations out of 1000 attempts. The fundamental issue is that the lattice gap (ratio of first to second shortest vector) is too small for BKZ to find the required short vectors in polynomial time. The lattice dimensions needed for the method to produce useful relations grow super-polynomially, making it no better than (and likely worse than) GNFS.
- **Status**: Debunked
- **Open Questions**: Could a more sophisticated lattice reduction algorithm (beyond BKZ) salvage some version of the approach? Is there a different lattice construction that avoids the gap problem? The broader question of whether lattice-based methods can contribute to factoring remains open, even though Schnorr's specific claim is refuted.

---

### 4. Lenstra's Elliptic Curve Method (ECM)
- **Authors**: Hendrik W. Lenstra Jr.
- **Year**: 1987
- **Source**: [Annals of Mathematics, Vol. 126 (1987)](https://doi.org/10.2307/1971363)
- **Core Idea**: The Elliptic Curve Method (ECM) is a factoring algorithm whose running time depends primarily on the size of the smallest prime factor of N, not on N itself. It works by choosing a random elliptic curve E over Z/NZ and a point P on E, then computing a large multiple kP using elliptic curve arithmetic. If p | N and the order |E(F_p)| happens to be B-smooth (all prime factors <= B), then kP = O in E(F_p), meaning the x-coordinate computation encounters a non-invertible element modulo N, revealing a factor via GCD. The complexity is L_p(1/2, sqrt(2)) where p is the smallest factor.
- **Mathematical Foundation**: Elliptic curves over finite fields, Hasse's theorem (|#E(F_p) - p - 1| <= 2*sqrt(p), so the group order is within [p + 1 - 2*sqrt(p), p + 1 + 2*sqrt(p)]), Montgomery curves (for efficient arithmetic with x-coordinate-only operations), the theory of smooth numbers (probability that a random number near p is B-smooth), and stage-2 optimizations (Brent-Suyama, PRAC chains). The Sato-Tate conjecture (now theorem) governs the distribution of group orders as the curve varies.
- **RSA Relevance**: ECM is the best method for finding "small" prime factors (up to ~80 digits). For RSA keys with properly chosen primes (both p and q around n/2 bits), ECM offers no advantage over GNFS. However, ECM is crucial for: (1) cofactorization in GNFS sieving (testing whether cofactors are prime or have small factors), (2) factoring numbers that might have been poorly generated (e.g., one small prime factor), and (3) as a component of other algorithms. The connection to the Langlands program via modularity (see File 1, entry 3) raises the theoretical question of whether modular form data could be used to select optimal curves.
- **Status**: Proven
- **Open Questions**: Can modularity/Sato-Tate data be exploited for better curve selection in practice? What is the record ECM factor? (As of 2025: a 83-digit factor found by Ryan Propper.) Can GPUs or TPUs dramatically accelerate ECM? Is there an "ECM for lattices" analogue?

---

### 5. Bao Yan et al.: Schnorr Lattice + QAOA Hybrid
- **Authors**: Bao Yan, Ziqi Tan, Shijie Wei, Haocong Jiang, Weilong Wang, Hong Wang, Lan Luo, Qianheng Duan, Yiting Liu, Wenhao Shi, Yangyang Fei, Xiangdong Meng, Yu Han, Zheng Shan, Jiachen Chen, Xuhao Zhu, Chuanyu Zhang, Feitong Jin, Hekang Li, Chao Song, Zhen Wang, Zhi Ma, H. Wang, Gui-Lu Long
- **Year**: 2022
- **Source**: [arXiv:2212.12372](https://arxiv.org/abs/2212.12372)
- **Core Idea**: This paper from a Chinese research group proposed a hybrid quantum-classical factoring approach combining Schnorr's lattice method with the Quantum Approximate Optimization Algorithm (QAOA). The idea is to use a small quantum computer running QAOA to solve the Closest Vector Problem (CVP) or SVP instances that arise in Schnorr's lattice construction, rather than using classical BKZ. The authors claimed to factor a 48-bit number using only 10 superconducting qubits on a real quantum processor, and extrapolated that RSA-2048 could be broken with O(log N / log log N) ~ 372 qubits. This extrapolation generated significant media attention and controversy.
- **Mathematical Foundation**: QAOA (variational quantum algorithm for combinatorial optimization), Ising model formulation of lattice problems, Schnorr's lattice construction (see entry 3), and QUBO (Quadratic Unconstrained Binary Optimization) reformulation of SVP/CVP. The quantum component solves a small optimization problem that corresponds to finding short vectors in the Schnorr lattice.
- **RSA Relevance**: The headline claim (RSA-2048 with ~372 qubits) is almost certainly incorrect. The approach inherits all the fundamental problems of Schnorr's method (debunked; see entry 3): the lattice construction does not produce useful factoring relations at cryptographic scale. Additionally, QAOA has no proven exponential speedup for optimization problems, and the 48-bit factoring demonstration used problem-specific optimizations that do not generalize. Multiple experts (Scott Aaronson, others) noted that the exponential bottleneck is in the lattice problem size, not the qubit count. The paper is best understood as an interesting experimental demonstration of hybrid quantum-classical computation on a small problem, not as a threat to RSA.
- **Status**: Speculative (experimental demonstration valid for small numbers; extrapolation to RSA-2048 is unjustified)
- **Open Questions**: Can QAOA provide any meaningful speedup for lattice problems compared to classical BKZ? Is there a different lattice construction (not Schnorr's) for which a small quantum processor could help? What is the actual scaling of the quantum resource requirements as a function of the number of digits?

---

### 6. D-Wave Quantum Annealing for RSA Factoring
- **Authors**: Wang Baonan, Hu Feng, Yao Haonan, Wang Chao, and others
- **Year**: 2025
- **Source**: [Tsinghua Science and Technology (2024/2025)](https://www.sciopen.com/article/10.26599/TST.2024.9010028)
- **Core Idea**: This work formulates integer factorization as a QUBO (Quadratic Unconstrained Binary Optimization) problem and solves it on D-Wave's quantum annealer. The approach represents the unknown prime factors p and q as binary variables, expresses the constraint N = p * q as a pseudo-Boolean polynomial, and maps this to an Ising Hamiltonian whose ground state encodes the factors. The authors demonstrated factoring of numbers up to 22 bits on D-Wave hardware. They also explored hybrid approaches combining quantum annealing with classical pre-processing to reduce the number of required qubits.
- **Mathematical Foundation**: QUBO formulation of multiplication constraints, Ising model Hamiltonians, quantum annealing (adiabatic quantum computation), penalty function methods for constraints, and minor embedding (mapping logical qubits to the physical connectivity graph of the D-Wave chip, typically Pegasus topology). The number of logical qubits scales as O(n) for an n-bit number, but minor embedding inflates this significantly.
- **RSA Relevance**: Very limited. The QUBO/Ising approach to factoring has fundamental scaling problems: (1) the number of logical qubits is O(n) where n is the bit length, but the connectivity requirements cause the physical qubit count to blow up quadratically or worse under minor embedding; (2) quantum annealing has no proven speedup over classical optimization for general QUBO problems; (3) the 22-bit factoring demonstration is 6 orders of magnitude away from RSA-2048 in bit length, and the required resources scale polynomially at best (more likely exponentially due to the annealing gap closing). The approach only works efficiently for "special-form" integers where significant bits of the factors are known in advance.
- **Status**: Active Research (but fundamentally limited)
- **Open Questions**: Can quantum annealing provide any speedup over classical QUBO solvers (simulated annealing, etc.) for factoring? Can the minor embedding overhead be reduced with better chip topologies? Is there a QUBO formulation that avoids the exponential precision requirements for large numbers?

---

### 7. Toom-Cook Optimized NFS
- **Authors**: Ilkhom Saydamatov (and related work)
- **Year**: 2025
- **Source**: [Concurrency and Computation: Practice and Experience (Wiley)](https://onlinelibrary.wiley.com/doi/10.1002/cpe.8365)
- **Core Idea**: This work explores optimizations to the Number Field Sieve using Toom-Cook multiplication algorithms for the large-integer arithmetic that dominates the sieving phase. Toom-Cook-k generalizes Karatsuba multiplication by evaluating polynomials at k points, reducing the multiplication of n-digit numbers from O(n^2) to O(n^{log(2k-1)/log(k)}). Applied to GNFS, this can accelerate the cofactorization step (where large products must be computed and tested for smoothness) and the polynomial arithmetic in the algebraic sieve. The paper explores specific configurations (Toom-3, Toom-4) optimized for the word sizes and operand ranges typical of GNFS computations.
- **Mathematical Foundation**: Toom-Cook polynomial interpolation, sub-quadratic multiplication algorithms, the connection between arithmetic complexity and sieving throughput in GNFS, and cache-aware algorithm design for modern processors. The approach sits at the intersection of algorithmic number theory and high-performance computing.
- **RSA Relevance**: Modest but practical. Any constant-factor speedup to GNFS sieving translates directly to a larger practically-factorable number. While this does not change the asymptotic complexity (still L_N(1/3, c)), it can shift the practical boundary by a few digits. For RSA security parameter estimation, these engineering improvements must be accounted for. The work also contributes to the broader ecosystem of GNFS optimization that includes CADO-NFS (entry 2).
- **Status**: Active Research
- **Open Questions**: What is the optimal Toom-Cook variant for GNFS cofactorization at different digit sizes? Can these optimizations be combined with GPU acceleration? How much practical speedup is achievable over the current CADO-NFS implementation?

---

### 8. Quadratic Sieve (QS)
- **Authors**: Carl Pomerance
- **Year**: 1981 (original); 1985 (multiple polynomial variant by Silverman)
- **Source**: [Pomerance (1985) — The Quadratic Sieve Factoring Algorithm](https://doi.org/10.1007/3-540-39757-4_17)
- **Core Idea**: The Quadratic Sieve is a factoring algorithm that was the fastest known method before GNFS and remains the simplest sub-exponential factoring algorithm to understand and implement. It works by sieving for values of Q(x) = (x + floor(sqrt(N)))^2 - N that are B-smooth, then combining these relations via linear algebra over GF(2) to find a square congruence x^2 = y^2 (mod N). Unlike GNFS, QS works in a single number field (Q itself), making the mathematics and implementation much simpler. The complexity is L_N(1/2, 1), which is sub-exponential but slower than GNFS's L_N(1/3, c) for large numbers.
- **Mathematical Foundation**: Smoothness of quadratic residues, the Gaussian elimination approach to finding square products, sieving by prime powers (similar to the Sieve of Eratosthenes), the large prime variation (allowing one or two large prime factors in the smooth relations), and the multiple polynomial variant (MPQS) which distributes the sieving over many polynomials for better parallelism.
- **RSA Relevance**: QS is practical and the method of choice for numbers up to about 100 decimal digits. Beyond that, GNFS surpasses it. For RSA-2048, QS is far less efficient than GNFS. However, QS remains important as: (1) a pedagogical entry point to sub-exponential factoring, (2) a practical tool for "medium-sized" numbers, (3) a benchmark for comparing new factoring approaches, and (4) the basis for the Self-Initializing QS (SIQS) variant used in many factoring libraries (e.g., msieve, YAFU).
- **Status**: Proven
- **Open Questions**: Can QS be meaningfully accelerated on modern GPU architectures? What is the exact crossover point with GNFS (generally believed to be around 100-110 digits)? Can ideas from QS be combined with quantum subroutines for a hybrid advantage?

---

### 9. Pollard's Rho and p-1 Methods
- **Authors**: John M. Pollard
- **Year**: 1975 (rho method); 1974 (p-1 method)
- **Source**: [Pollard, "A Monte Carlo method for factorization," BIT 1975](https://doi.org/10.1007/BF01933667); [Pollard, "Theorems on factorization and primality testing," PCPS 1974](https://doi.org/10.1017/S0305004100049252)
- **Core Idea**: **Pollard's rho**: A probabilistic factoring algorithm that detects a collision in the sequence x_{n+1} = x_n^2 + c (mod N), where the collision reveals a factor because the sequence's behavior modulo p (an unknown factor) has a shorter cycle than modulo N. Uses Floyd's cycle detection (tortoise and hare) with O(sqrt(p)) expected iterations and O(1) memory. Expected to find a factor p in O(p^{1/4}) GCD operations. **Pollard's p-1**: Exploits smooth group orders. If p-1 is B-smooth for a prime factor p of N, then computing a^{B!} mod N and taking gcd(a^{B!} - 1, N) reveals p. This works because a^{B!} = 1 (mod p) by Fermat's little theorem when B! is a multiple of p-1.
- **Mathematical Foundation**: **Rho**: Birthday paradox applied to the multiplicative group modulo p; Floyd's or Brent's cycle-detection algorithms; the heuristic that the pseudo-random iteration x -> x^2 + c behaves like a random function modulo p. **p-1**: Fermat's little theorem, smoothness of p-1, and stage-2 extensions (allowing one large prime factor in p-1). Williams' p+1 method is a natural generalization using Lucas sequences.
- **RSA Relevance**: Both methods are important in the factoring toolkit but pose no threat to properly generated RSA keys. Pollard's rho has complexity O(N^{1/4}), which for RSA-2048 means O(2^{512}) operations — far beyond feasibility. Pollard's p-1 only works when p-1 has only small factors, which is why RSA key generation requires "strong primes" (where p-1 has at least one large prime factor). However, both methods are valuable for: (1) quickly detecting weak keys, (2) factoring small cofactors in GNFS, and (3) as building blocks for more sophisticated algorithms.
- **Status**: Proven
- **Open Questions**: Can Pollard's rho be meaningfully parallelized beyond the van Oorschot-Wiener distinguished-points approach? Are there pseudo-random iterations better than x^2 + c for the rho method? Can the p-1 method be generalized to exploit other group structures (beyond the multiplicative group)?

---

## Complexity Landscape Summary

| Algorithm | Complexity | Best For | RSA-2048 Feasibility |
|-----------|-----------|----------|---------------------|
| Trial division | O(sqrt(N)) | Tiny numbers | Impossible |
| Pollard's rho | O(N^{1/4}) | < 30 digits | Impossible |
| Pollard's p-1 | O(B * log N) | Smooth p-1 | Only for weak keys |
| ECM | L_p(1/2, sqrt(2)) | Small factors (< 80 digits) | Only for weak keys |
| Quadratic Sieve | L_N(1/2, 1) | 50-100 digits | Impossible |
| GNFS | L_N(1/3, 1.923) | > 100 digits | ~10^{26} ops (impossible) |
| Schnorr lattice | Claimed poly | Any | **Debunked** |
| QAOA hybrid | Claimed O(log N) qubits | Any | **Unsubstantiated** |
| D-Wave annealing | Unknown scaling | Special-form | Only small numbers |

The table makes clear that no classical or hybrid classical-quantum algorithm currently threatens RSA-2048. The GNFS complexity wall at L_N(1/3, c) appears fundamental — breaking through to L_N(1/4, c) or better would be a revolution in computational number theory.
