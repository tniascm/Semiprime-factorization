# Quantum Factoring Algorithms

An annotated bibliography of quantum approaches to integer factorization, from foundational algorithms to state-of-the-art resource estimates for breaking RSA.

---

### 1. Shor's Algorithm
- **Authors**: Peter W. Shor
- **Year**: 1994
- **Source**: [Proceedings of FOCS 1994](https://doi.org/10.1109/SFCS.1994.365700); [SIAM Journal on Computing, 1997](https://doi.org/10.1137/S0097539795293172)
- **Core Idea**: Shor's algorithm reduces integer factorization to the problem of finding the period of the modular exponentiation function f(x) = a^x mod N, then uses quantum Fourier transform (QFT) to find this period exponentially faster than any known classical method. Given the period r of a^x mod N for random a, one computes gcd(a^{r/2} +/- 1, N) to extract factors with high probability. The algorithm requires O(n^2 log n log log n) quantum gates for an n-bit number, achieving exponential speedup over the best classical algorithms (sub-exponential GNFS).
- **Mathematical Foundation**: Quantum Fourier transform over Z/rZ, period-finding reduction of factoring (based on the structure of the multiplicative group (Z/NZ)*), modular exponentiation circuits, continued fraction expansion for extracting the period from QFT measurement outcomes, and the Chinese Remainder Theorem. The correctness relies on the fact that for random a, the period r is even and a^{r/2} != -1 (mod N) with probability >= 1/2.
- **RSA Relevance**: This is the foundational quantum threat to RSA. Shor's algorithm factors N = pq in polynomial time on a fault-tolerant quantum computer, completely breaking RSA, Diffie-Hellman, and elliptic curve cryptography. The largest number factored on real quantum hardware using Shor's algorithm is 15 = 3 x 5 (IBM, 2001), with contested claims of 21 = 3 x 7 using compiled/optimized circuits. The gap between 21 and RSA-2048 (617 digits) is enormous, but the algorithm itself is provably efficient — the bottleneck is hardware.
- **Status**: Proven (algorithmic correctness and polynomial complexity)
- **Open Questions**: What is the minimum number of logical (error-corrected) qubits needed to factor RSA-2048? Can the circuit depth be reduced further (see Regev's improvement below)? Can Shor's algorithm be adapted to work with partial or noisy period information? Is there a dequantized version of the period-finding subroutine that could run classically?

---

### 2. Regev's Multidimensional Quantum Factoring Algorithm
- **Authors**: Oded Regev
- **Year**: 2023
- **Source**: [arXiv:2308.06572](https://arxiv.org/abs/2308.06572)
- **Core Idea**: Regev proposed the first asymptotic improvement to quantum factoring in nearly 30 years. Instead of Shor's single-dimensional period finding, the algorithm uses a multidimensional approach: it performs sqrt(n) + 4 independent quantum circuit executions, each using O(n^{3/2}) gates (compared to Shor's O(n^2)), and then combines the results via classical lattice reduction (LLL/BKZ). Each quantum run computes a modular exponentiation to a smaller power, producing a "noisy" lattice point; the classical post-processing recovers the period from these points using the geometry of numbers. Total gate complexity: O-tilde(n^{3/2}), a quadratic improvement in the exponent.
- **Mathematical Foundation**: Lattice reduction (LLL algorithm, Babai's nearest plane algorithm), the geometry of numbers (Minkowski's theorem), multidimensional quantum Fourier transform, Gaussian superpositions, and discrete Gaussian distributions over lattices. The algorithm exploits the fact that sqrt(n) noisy samples from a lattice can determine the lattice via LLL, whereas a single sample (as in Shor) requires higher precision (more gates).
- **RSA Relevance**: Reduces the quantum gate count for factoring RSA-2048 by a factor of roughly sqrt(2048) ~ 45x compared to Shor, at the cost of needing sqrt(n) sequential runs (each requiring a fresh quantum computation and measurement). This tradeoff is favorable for near-term quantum hardware where circuit depth is the primary bottleneck but repeated executions are feasible. However, the classical post-processing (lattice reduction) may introduce practical complications, and the qubit count is higher than Shor's unless combined with space-efficient techniques (see Ragavan-Regev).
- **Status**: Proven (algorithmic correctness); Active Research (practical resource estimates)
- **Open Questions**: What are the exact constant factors in the gate count for RSA-2048? Is the lattice reduction step practically efficient for cryptographic-size inputs? Can the number of quantum runs be reduced below sqrt(n)? Does the algorithm compose well with quantum error correction?

---

### 3. Ragavan-Regev Space-Efficient Quantum Factoring
- **Authors**: Seyoon Ragavan, Oded Regev
- **Year**: 2023
- **Source**: [ePrint 2023/1501](https://eprint.iacr.org/2023/1501.pdf)
- **Core Idea**: Ragavan and Regev combined the best of both worlds: Regev's reduced gate count of O-tilde(n^{3/2}) with Shor's qubit efficiency of O(n) qubits. The original Regev algorithm required O(n^{3/2}) qubits due to the multidimensional QFT; this variant uses a space-efficient implementation of the modular exponentiation and Fibonacci-style register reuse to bring the qubit count back down. The result is a quantum factoring algorithm with O-tilde(n^{3/2}) gates AND O(n) qubits, plus sqrt(n) sequential runs with classical post-processing.
- **Mathematical Foundation**: Same lattice-based framework as Regev's algorithm, combined with space-efficient quantum arithmetic (reversible computation, Bennett's pebble game for space-time tradeoffs), and techniques from quantum circuit optimization. The space reduction comes from carefully scheduling the quantum computation to reuse qubits, trading a small increase in depth for a large decrease in width.
- **RSA Relevance**: This is the current theoretical state-of-the-art for quantum factoring in terms of the gate-count-times-qubit-count metric. For RSA-2048 (n = 2048 bits), this gives roughly 2048 qubits (logical) and ~2048^{1.5} ~ 93,000 gates per run, with ~49 runs. Compared to Shor's ~2048 qubits and ~2048^2 ~ 4.2M gates in a single run. The practical question is whether the multi-run approach with classical lattice reduction is more feasible than a single deep Shor circuit — this depends on the relative costs of circuit depth vs. repeated execution in a given quantum error correction scheme.
- **Status**: Proven (algorithmic correctness)
- **Open Questions**: Detailed resource estimates for RSA-2048 under realistic error correction models are needed. How does the lattice reduction step scale with the noise in the quantum measurements? Can the approach be further optimized for specific quantum architectures (e.g., surface codes)?

---

### 4. Gidney's Sub-Million-Qubit RSA-2048 Factoring Estimate
- **Authors**: Craig Gidney
- **Year**: 2025 (May)
- **Source**: [arXiv:2505.15917](https://arxiv.org/abs/2505.15917)
- **Core Idea**: Gidney presented a concrete quantum circuit construction for factoring RSA-2048 using fewer than 1 million noisy (physical) qubits in approximately 1 week of computation. This represents a 20x reduction in qubit count compared to his own 2019 estimate (20M qubits, 8 hours). The key innovations are: (1) approximate residue arithmetic — using approximate rather than exact modular reduction during the exponentiation, tolerating small errors that are corrected classically; (2) yoked surface codes — a new error correction layout that couples multiple logical qubits more efficiently; (3) magic state cultivation — an improved method for producing the T-gates needed for universal quantum computation, replacing the more expensive magic state distillation; (4) windowed arithmetic — batching modular multiplications to reduce the total Toffoli gate count to ~6.5 x 10^9, a 100x reduction from the 2019 estimate.
- **Mathematical Foundation**: Quantum error correction (surface codes), lattice surgery for logical gate implementation, the Toffoli gate as a universal quantum gate (with magic states), Montgomery modular multiplication adapted for quantum circuits, and probabilistic error analysis for approximate arithmetic. The resource estimate assumes a physical error rate of 10^{-3} and a code distance chosen to achieve the target logical error rate over the full computation.
- **RSA Relevance**: This is the most detailed and optimistic concrete estimate for breaking RSA-2048 with a quantum computer. Sub-1M physical qubits is within the roadmaps of several quantum hardware companies (IBM, Google, Quantinuum) for the late 2020s to early 2030s. The 1-week runtime is operationally feasible. If these estimates are accurate, RSA-2048 could be broken within 5-10 years of achieving the required qubit count and error rates. This paper is the primary basis for NIST's urgency in standardizing post-quantum cryptography.
- **Status**: Active Research (circuit construction proven correct; hardware requirements not yet achieved)
- **Open Questions**: Can the qubit count be pushed below 100K? What is the actual wall-clock time accounting for classical control overhead? Are the physical error rate assumptions (10^{-3}) achievable at the required scale? Can the approach be adapted to factor numbers with more than 2048 bits (e.g., RSA-4096)?

---

### 5. Gidney-Ekera 2019 Baseline: 20M Qubits for RSA-2048
- **Authors**: Craig Gidney, Martin Ekera
- **Year**: 2019
- **Source**: [arXiv:1905.09749](https://arxiv.org/abs/1905.09749)
- **Core Idea**: This paper established the baseline resource estimate that dominated discussions of quantum threats to RSA for five years. Gidney and Ekera showed that RSA-2048 could be factored in approximately 8 hours using 20 million noisy superconducting qubits, assuming surface code error correction with a physical error rate of 10^{-3}. The construction uses Shor's algorithm with optimized modular exponentiation circuits, costing approximately 2.3 x 10^{12} Toffoli gates. They used a combination of windowed arithmetic, oblivious carry runways, and measurement-based uncomputation to minimize both qubit count and circuit depth.
- **Mathematical Foundation**: Surface code quantum error correction, magic state distillation (for T/Toffoli gates), space-time volume optimization of quantum circuits, reversible arithmetic circuit design, and the Ekera-Hastad variant of Shor's algorithm (which uses a shorter second register to reduce qubit count at the cost of a small classical post-processing step).
- **RSA Relevance**: This paper quantified the quantum threat to RSA-2048 concretely for the first time with realistic error correction overhead. The 20M qubit estimate made the threat seem distant (current devices have ~1000 qubits), but Gidney's 2025 update (see entry 4) reduced this by 20x. The paper also demonstrated that the dominant cost is magic state distillation (producing high-fidelity T-gates), which motivated the subsequent development of magic state cultivation.
- **Status**: Proven (circuit construction); Superseded (by Gidney 2025 for resource estimates)
- **Open Questions**: Essentially resolved by Gidney 2025, but the question of how error rates scale with system size in real hardware remains critical. The 2019 paper's 10^{-3} error rate assumption is being approached but not yet demonstrated at scale.

---

### 6. Implementation Analysis of Regev's Algorithm
- **Authors**: Various (implementation and resource estimation studies)
- **Year**: 2025
- **Source**: [arXiv:2511.18198](https://arxiv.org/abs/2511.18198)
- **Core Idea**: This work provides concrete resource estimates for implementing Regev's multidimensional quantum factoring algorithm (entry 2) on fault-tolerant quantum hardware. The analysis compiles Regev's abstract algorithm into surface-code circuits and compares the total space-time volume against optimized Shor implementations. Key findings include the practical overhead of the multidimensional QFT, the impact of the lattice reduction post-processing step on success probability, and the tradeoff between circuit depth reduction and the need for multiple sequential runs. The analysis determines under what parameter regimes Regev's approach outperforms Shor's in terms of total physical resources.
- **Mathematical Foundation**: Quantum circuit compilation to surface codes, space-time volume analysis, lattice reduction complexity (LLL runs in polynomial time but with large constants for cryptographic-size inputs), and error propagation analysis through approximate quantum arithmetic. The analysis must account for the fact that each of Regev's sqrt(n) runs produces a noisy lattice point, and the classical post-processing must recover the exact lattice from noisy samples.
- **RSA Relevance**: Determines whether Regev's theoretical improvement over Shor translates to a practical improvement for RSA-2048. Preliminary analyses suggest that for current and near-term quantum architectures (where circuit depth is the primary bottleneck), Regev's approach may offer a modest advantage, but the multi-run overhead and lattice reduction costs partially offset the per-run savings. For RSA-2048 specifically, Gidney's optimized Shor implementation (entry 4) likely remains more practical due to extensive engineering optimizations.
- **Status**: Active Research
- **Open Questions**: What is the crossover point (in terms of number size) where Regev's approach definitively outperforms optimized Shor? Can the lattice reduction post-processing be parallelized or accelerated using quantum subroutines? How sensitive is the algorithm to shot noise in the quantum measurements?

---

## Timeline Assessment

| Milestone | Year | Qubits (Physical) | Target |
|-----------|------|--------------------|--------|
| Shor's algorithm published | 1994 | N/A (theoretical) | Any N |
| First quantum factoring (15 = 3x5) | 2001 | 7 (NMR) | 4 bits |
| Gidney-Ekera baseline | 2019 | 20M (estimated) | RSA-2048 |
| Regev improvement | 2023 | ~O(n) logical | Any N |
| Ragavan-Regev space-efficient | 2023 | O(n) logical | Any N |
| Gidney updated estimate | 2025 | <1M (estimated) | RSA-2048 |
| Current largest QC | 2025 | ~1,200 (IBM Condor) | N/A |

**Gap analysis**: Factoring RSA-2048 requires ~1M physical qubits at 10^{-3} error rate. Current hardware has ~1,200 qubits at ~10^{-3} error rate. The gap is ~1000x in qubit count, with the additional challenge of maintaining error rates at scale. At historical scaling rates (~2x qubits every 1-2 years), the gap closes around 2035-2040. However, architectural breakthroughs (e.g., better error correction codes, hardware with lower native error rates) could accelerate this timeline significantly.
