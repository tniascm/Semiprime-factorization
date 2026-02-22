# Quantum-Inspired Classical Algorithms

Exploring the boundary between quantum and classical computation, with focus on dequantization results and what they imply for the possibility of classical factoring breakthroughs.

---

### Ewin Tang's Dequantization of Quantum Machine Learning
- **Authors**: Ewin Tang
- **Year**: 2018 (initial result), 2019-2025 (extensions)
- **Source**: [https://arxiv.org/abs/1807.04271](https://arxiv.org/abs/1807.04271)
- **Core Idea**: Tang proved that the exponential quantum speedup claimed for the quantum recommendation systems algorithm (Kerenidis & Prakash, 2016) does not exist -- a classical algorithm can achieve comparable performance. The key insight is that when data has low-rank structure and the algorithm has "sample and query" access to the input (ability to sample rows/entries proportional to their norms), classical algorithms based on random sampling and sketching can simulate the quantum algorithm's behavior. This result was subsequently extended to quantum PCA, quantum low-rank linear systems, and quantum supervised clustering, establishing that low-rank structure, not quantumness, is the source of speedup in many quantum ML algorithms. Tang received the Maryam Mirzakhani New Frontiers Prize in 2025 for this work.
- **Mathematical Foundation**: Singular value decomposition and low-rank approximation, importance sampling (sampling matrix entries proportional to squared norms), Frieze-Kannan-Vempala low-rank matrix approximation, stochastic inner product estimation, length-squared sampling access model. The classical algorithm achieves polylogarithmic dependence on matrix dimensions (matching the quantum algorithm) at the cost of polynomial dependence on rank and inverse error -- matching quantum query complexity up to polynomial factors in these secondary parameters.
- **RSA Relevance**: Tang's work establishes a powerful methodology: identify the structural property that a quantum algorithm exploits, then ask whether a classical algorithm can exploit the same structure. For Shor's algorithm, the structural property exploited is the periodicity of modular exponentiation f(x) = a^x mod N. The question becomes: can this periodicity be detected classically with sub-exponential resources, perhaps by exploiting the specific algebraic structure of modular arithmetic? Tang's dequantization does not directly apply (Shor exploits periodicity, not low-rank structure), but the methodology -- identifying the "real" source of speedup and building classical algorithms around it -- is the template.
- **Status**: Proven (the dequantization results are rigorous; application to factoring is speculative)
- **Open Questions**: Is there a classical analog of Shor's quantum Fourier transform that works for periodic functions over cyclic groups? Can the "sample and query" access model be adapted to number-theoretic functions? Does modular exponentiation have any hidden low-rank or low-complexity structure that classical algorithms could exploit? What is the minimal quantum resource (entanglement, superposition depth) needed for Shor's speedup to survive?

---

### The Quantum-Inspired Factoring Question: Can Shor's Insight Be Partially Dequantized?
- **Authors**: Various (ongoing research direction)
- **Year**: 2018-present (post-Tang dequantization era)
- **Source**: Conceptual synthesis of Shor (1994), Tang (2018), and ongoing complexity theory
- **Core Idea**: Shor's algorithm achieves exponential speedup for factoring through two key quantum ingredients: (1) superposition to evaluate f(x) = a^x mod N at all x in {0, ..., N-1} simultaneously, and (2) quantum Fourier transform (QFT) to extract the period of f from the superposition state. The dequantization question asks: can either ingredient be replaced classically? Ingredient (1) -- parallel evaluation -- could potentially be approximated by sampling-based methods if the right samples are informative. Ingredient (2) -- period extraction via QFT -- is the harder part, as classical Fourier analysis of exponentially long periodic sequences requires seeing enough of the sequence to detect the period, which seems to require exponential samples.
- **Mathematical Foundation**: Shor's algorithm: prepare superposition |x>|a^x mod N>, apply QFT to first register, measure to get multiple of N/r (where r is the order of a mod N), use continued fractions to extract r. Classical period-finding: requires O(r) evaluations to detect period r by birthday-type arguments, or O(sqrt(r)) with Pollard's rho. For RSA-sized N, r can be as large as N, making classical period-finding exponential. The gap is that QFT extracts period information from a single coherent superposition, while classical methods need repeated evaluations.
- **RSA Relevance**: This is the central question for classical factoring breakthroughs. If any component of Shor's speedup can be classically replicated, it would transform factoring. Partial dequantization possibilities include: (a) exploiting the specific algebraic structure of modular exponentiation (not just any periodic function) to enable classical period detection, (b) using number-theoretic properties of the group Z/NZ to constrain the period search space, (c) finding a different representation of the factoring problem where classical algorithms can compete. The failure of all known classical approaches to match Shor suggests the speedup may be genuine -- but absence of evidence is not evidence of absence.
- **Status**: Open question / Speculative
- **Open Questions**: Is Shor's speedup fundamentally about quantum interference (destructive interference canceling non-period components) or about parallel evaluation? Could a classical algorithm exploit the multiplicative structure of Z/NZ to constrain period search? Does the success of dequantization in ML suggest that quantum speedups for structured algebraic problems might also be vulnerable? What is the minimum amount of "quantumness" needed for superpolynomial factoring speedup?

---

### QAOA: Quantum Approximate Optimization Algorithm
- **Authors**: Edward Farhi, Jeffrey Goldstone, Sam Gutmann
- **Year**: 2014
- **Source**: [https://arxiv.org/abs/1411.4028](https://arxiv.org/abs/1411.4028)
- **Core Idea**: QAOA is a variational quantum algorithm that applies alternating layers of a "problem Hamiltonian" (encoding the optimization objective) and a "mixer Hamiltonian" (providing exploration) to an initial superposition state. The circuit depth p controls the quality of approximation -- higher p gives better solutions but requires more quantum resources. For p=1, QAOA can be analyzed exactly and often matches or slightly exceeds classical random assignment. For large p, QAOA approaches quantum adiabatic computation. Importantly, low-depth QAOA (small p) can often be efficiently simulated classically, raising questions about where quantum advantage begins.
- **Mathematical Foundation**: Variational principle, parameterized quantum circuits |gamma, beta> = prod U_B(beta_j) U_C(gamma_j) |+>, expectation value optimization E = <gamma, beta| C |gamma, beta>, adiabatic theorem (large p limit). Performance guarantees exist for specific problems: QAOA achieves 0.6924 approximation ratio for MaxCut on 3-regular graphs at p=1 (Farhi et al.), matching the classical Goemans-Williamson bound requires higher p.
- **RSA Relevance**: Factoring can be encoded as an optimization problem: minimize f(x,y) = (N - x*y)^2 over integers x, y in appropriate ranges. QAOA could in principle search for factors by optimizing this objective. However, the landscape of this optimization problem is highly non-convex with exponentially many local minima, and there is no evidence that QAOA provides speedup over classical optimization for this specific structure. The more relevant question is whether QAOA-style variational approaches, when run classically (tensor network simulation), could discover useful factoring heuristics for small instances that generalize.
- **Status**: Active research (quantum algorithm proven, classical simulability for low depth proven, advantage for factoring undemonstrated)
- **Open Questions**: At what circuit depth p does QAOA for factoring become classically intractable to simulate? Does the optimization landscape for factoring-as-QAOA have structure that helps or hinders convergence? Can classical simulation of QAOA circuits reveal useful patterns for factoring small numbers? Is QAOA for factoring fundamentally different from random search for practical instance sizes?

---

### High-Temperature Gibbs States Are Unentangled and Efficiently Preparable
- **Authors**: Ainesh Bakshi, Allen Liu, Ankur Moitra, Ewin Tang
- **Year**: 2024 (FOCS 2024)
- **Source**: [QIP 2025 Plenary Talk](https://arxiv.org/abs/2403.09Surface), [https://arxiv.org/abs/2403.09184](https://arxiv.org/abs/2403.09184)
- **Core Idea**: This work proves that thermal (Gibbs) states of quantum systems at sufficiently high temperature are product states (unentangled) and can be efficiently prepared by classical algorithms. This is significant because Gibbs state preparation was considered a potential source of quantum advantage -- if quantum computers could prepare Gibbs states that classical computers cannot, this would demonstrate useful quantum speedup. The result shows that above a critical temperature, the thermal state has no long-range entanglement, and its local marginals can be computed classically in polynomial time. This shrinks the regime where quantum advantage for Gibbs sampling might exist.
- **Mathematical Foundation**: Quantum Gibbs states rho = e^{-beta H} / Z, cluster expansion techniques, decay of correlations at high temperature, belief propagation on quantum systems, polynomial-time classical algorithms for computing local expectation values. The critical temperature threshold depends on the interaction strength and geometry of the Hamiltonian.
- **RSA Relevance**: This result further constrains the boundary of quantum advantage. If high-temperature quantum states can be classically simulated, the quantum advantage for computation must come from low-temperature (highly entangled) regimes. For factoring, the question is: does Shor's algorithm operate in a "low-temperature" (high-entanglement) regime that resists dequantization, or could there be a "high-temperature" relaxation of the factoring problem that classical algorithms can solve? More broadly, each result that shrinks the domain of quantum advantage increases the probability that classical algorithms for structured problems like factoring can be improved.
- **Status**: Proven (rigorous mathematical result)
- **Open Questions**: What is the critical "temperature" (in a metaphorical sense) below which factoring-related quantum computations become classically intractable? Can the techniques (cluster expansion, belief propagation) be applied to classical factoring subroutines? Does the entanglement structure of Shor's algorithm at the critical step (post-QFT measurement) provide clues about classical simulability?

---

### Improved Classical Singular Value Transformation
- **Authors**: Ainesh Bakshi, Ewin Tang
- **Year**: 2024 (SODA 2024)
- **Source**: [https://arxiv.org/abs/2303.01492](https://arxiv.org/abs/2303.01492)
- **Core Idea**: Quantum singular value transformation (QSVT) is a powerful unifying framework for quantum algorithms that encompasses Grover search, quantum walks, Hamiltonian simulation, and quantum linear algebra. Bakshi and Tang showed that for matrices with bounded stable rank (a softer condition than low rank), classical algorithms can perform many SVT-based tasks with polynomial overhead compared to quantum algorithms. This extends the dequantization program from specific algorithms (recommendation, PCA) to a broad algorithmic framework, showing that the classical-quantum gap for matrix computations is smaller than previously believed.
- **Mathematical Foundation**: Singular value transformation: apply polynomial transformations to singular values of a matrix. Quantum version uses block-encoding and signal processing. Classical version uses sampling-based matrix approximation, stable rank (||A||_F^2 / ||A||^2) as a complexity parameter replacing rank, and stochastic trace estimation. The classical algorithm achieves query complexity polynomial in stable rank and inverse error, polylogarithmic in matrix dimension.
- **RSA Relevance**: SVT-based algorithms encompass quantum linear algebra, which is relevant to lattice-based factoring approaches (where the core computation involves finding short vectors in lattices, requiring linear algebra over large matrices). If classical SVT can approximate quantum SVT for structured matrices arising in lattice problems, this could lead to improved classical lattice algorithms -- which could in turn improve NFS or other lattice-based factoring methods. The stable rank condition is key: do the matrices arising in factoring-related lattice problems have bounded stable rank?
- **Status**: Proven (classical algorithm with rigorous guarantees)
- **Open Questions**: Do lattice matrices arising in NFS/GNFS have low stable rank? Can classical SVT be applied to speed up the linear algebra phase of NFS? Does the dequantization of SVT imply anything about the classical complexity of lattice problems (SVP, CVP) that underlie factoring?

---

### Tensor Network Methods for Classical Simulation of Quantum Circuits
- **Authors**: Various (Vidal, 2003; Verstraete & Cirac, 2004; Orus, 2014 review; Google/IBM teams, 2019-present)
- **Year**: 2003-present
- **Source**: [https://arxiv.org/abs/1306.2164](https://arxiv.org/abs/1306.2164) (Orus review), [https://arxiv.org/abs/quant-ph/0301063](https://arxiv.org/abs/quant-ph/0301063) (Vidal, TEBD)
- **Core Idea**: Tensor network methods represent quantum states as networks of contracted tensors, with the key parameter being the bond dimension chi (controlling the amount of entanglement the representation can capture). Matrix product states (MPS) can represent 1D quantum states with bounded entanglement efficiently, and contraction of the tensor network gives expectation values in time polynomial in chi. For quantum circuits, each gate updates the tensor network, and the bond dimension grows with entanglement -- for circuits with limited entanglement growth (short depth, 1D connectivity, or special structure), classical simulation via tensor networks is efficient. This has been used to challenge quantum supremacy claims (classically simulating Google's Sycamore circuits).
- **Mathematical Foundation**: Tensor decomposition (SVD-based truncation), Schmidt decomposition and entanglement entropy, area laws for entanglement in physical systems, contraction complexity of tensor networks (#P-hard in general but efficient for tree-like or bounded-treewidth networks). The simulation cost is O(chi^3 * n) per time step for MPS, where chi = 2^S for entanglement entropy S. Shor's algorithm generates O(n) entanglement entropy, requiring chi = 2^n -- exponential, preventing efficient tensor network simulation.
- **RSA Relevance**: Tensor network simulation of Shor's algorithm fails because the algorithm generates maximal entanglement during the QFT step. However, tensor networks provide a precise diagnostic: they tell us exactly where and how much entanglement is needed for quantum speedup. If a modified factoring circuit could be designed with lower entanglement (perhaps by sacrificing some success probability or working with structured instances), tensor network methods might enable "partially quantum" classical factoring. Additionally, tensor network techniques have been applied to combinatorial optimization and constraint satisfaction -- could they be applied to factoring formulated as a constraint problem?
- **Status**: Proven (tensor network theory is rigorous; application to factoring simulation is understood to be exponentially costly for Shor's circuit)
- **Open Questions**: Are there quantum factoring circuits with lower entanglement than Shor's that could be classically simulated? Can tensor network optimization (DMRG-style) be applied to factoring-as-optimization with competitive results? What is the minimum entanglement needed for a quantum factoring speedup? Could approximate tensor network simulation of Shor's algorithm (with truncated bond dimension) still extract useful partial information about factors?

---

## Synthesis: What Quantum Speedups Survive Dequantization, and What This Means for Factoring

The dequantization revolution initiated by Ewin Tang has dramatically reshaped our understanding of the quantum-classical boundary. The key findings and their implications for factoring are:

**What Has Been Dequantized (Quantum Advantage Eliminated):**
- Quantum recommendation systems (low-rank structure)
- Quantum PCA and low-rank regression
- Quantum supervised clustering
- Quantum singular value transformation (for bounded stable rank)
- High-temperature Gibbs state preparation

The common thread: when the problem has **low-rank or bounded-complexity structure** and the algorithm has **sampling access** to the input, classical algorithms can match quantum performance up to polynomial factors. The "quantum advantage" in these cases was really a "structure advantage" that classical algorithms can also exploit.

**What Has NOT Been Dequantized (Quantum Advantage Survives):**
- **Shor's factoring algorithm**: Exploits periodicity + quantum Fourier transform, not low-rank structure
- **Grover's search**: Proven quadratic quantum speedup for unstructured search (BBBV lower bound)
- **Quantum simulation of quantum systems**: Genuine exponential advantage for simulating quantum dynamics
- **Low-temperature Gibbs state preparation**: Entanglement is genuinely needed

The critical observation is that **Shor's algorithm exploits a different structural property than the dequantized algorithms**. Dequantized algorithms exploit low-rank structure in data matrices; Shor exploits periodicity in modular arithmetic. The QFT extracts period information from quantum superposition in a way that has no known classical analog for exponentially long periods.

**Implications for Classical Factoring:**

1. **Structure Identification**: Following Tang's methodology, the path forward is to identify exactly what structural property of modular exponentiation Shor's algorithm exploits, and determine whether that structure can be accessed classically. The period r of a^x mod N is typically comparable to N, making classical period detection exponentially costly. But perhaps the *specific* structure of this period (it divides phi(N), which itself has structure) enables shortcuts.

2. **Partial Dequantization**: Even if full dequantization of Shor is impossible, partial dequantization might be valuable. For example, if classical algorithms could narrow the period search space from O(N) to O(N^(1/3)), this would improve classical factoring even without matching Shor's polynomial time.

3. **Entanglement as Diagnostic**: Tensor network analysis confirms that Shor's algorithm requires maximal entanglement (O(n) qubits entangled), suggesting the quantum speedup is genuine. Any classical approach to factoring must therefore find a completely different path -- not by simulating Shor, but by exploiting number-theoretic structure that Shor's approach ignores.

4. **The Optimistic View**: The dequantization results show that quantum advantage is far rarer than once believed. Many "exponential quantum speedups" turned out to be illusory. While Shor's speedup appears genuine, the shrinking domain of quantum advantage should motivate continued search for classical factoring improvements, possibly using entirely different approaches (algebraic, geometric, or information-theoretic) that have not yet been thoroughly explored.

The honest assessment: Shor's factoring speedup is the most robust known quantum advantage for a practical problem. It has resisted dequantization for 30 years, and the structural reasons (maximal entanglement, exploitation of quantum interference for period detection) suggest it may be genuinely quantum. But the lesson of dequantization is humility -- what seems like a fundamental quantum advantage may yet yield to clever classical algorithms that exploit the right structure.
