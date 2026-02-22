# AI-Driven Mathematical Discovery

An annotated bibliography of AI systems that have achieved breakthroughs in mathematics, algorithm design, and formal reasoning, evaluated for potential application to integer factorization and cryptanalysis.

---

### 1. AlphaEvolve: Gemini-Powered Evolutionary Algorithm Discovery
- **Authors**: Google DeepMind (Alexander Novikov, Ngoc-Khanh Nguyen, Marvin Ritter, and team)
- **Year**: 2025
- **Source**: [arXiv:2506.13131](https://arxiv.org/abs/2506.13131)
- **Core Idea**: AlphaEvolve is an evolutionary coding agent powered by Google's Gemini large language models that discovers novel algorithms and mathematical constructions. It operates by maintaining a population of programs, using Gemini to propose mutations (code modifications), evaluating fitness on target problems, and iterating. Its headline result broke a 56-year-old record in fast matrix multiplication: it found a scheme for 4x4 matrix multiplication using only 48 scalar multiplications (previously 49, set by Strassen-derived methods). Beyond this, AlphaEvolve improved on the state of the art for approximately 20% of 50 open mathematical problems it was tested on, spanning combinatorics, geometry, and optimization. Unlike AlphaProof (entry 2), it does not produce formal proofs but rather discovers algorithms that can be verified computationally.
- **Mathematical Foundation**: Evolutionary algorithms (genetic programming, MAP-Elites diversity maintenance), LLM-guided mutation operators, program synthesis, and domain-specific fitness evaluation. For matrix multiplication, the fitness function counts scalar multiplications for a given tensor decomposition. The mathematical connection is to the algebraic complexity of bilinear maps (Strassen's framework), where matrix multiplication of n x n matrices is equivalent to decomposing a specific tensor of rank R, with R being the number of scalar multiplications.
- **RSA Relevance**: Two potential connections: (1) **Algorithm discovery for factoring** — AlphaEvolve could potentially discover novel factoring algorithms or optimize existing ones (e.g., polynomial selection in GNFS, ECM curve parameterization, sieving strategies). The evolutionary framework is well-suited to exploring the space of algorithmic modifications. (2) **Computational number theory** — The matrix multiplication result demonstrates that AI can find algebraic structures (tensor decompositions) that elude human mathematicians. Factoring has a rich algebraic structure (groups, rings, lattices) that might similarly yield to AI-guided search. The key limitation is that AlphaEvolve optimizes for computable fitness functions, and "factors N" is a binary outcome, not a smooth fitness landscape.
- **Status**: Proven (results verified computationally)
- **Open Questions**: Can AlphaEvolve be applied to factoring-related optimization problems (e.g., finding better GNFS polynomials, optimizing lattice reduction strategies)? Can it discover entirely new factoring paradigms, or only optimize known approaches? What happens when the fitness landscape is sparse or deceptive (as it likely is for factoring)?

---

### 2. AlphaProof: Reinforcement Learning for Formal Theorem Proving
- **Authors**: Google DeepMind (Alex Davies, Pushmeet Kohli, and team)
- **Year**: 2024
- **Source**: [Google DeepMind blog (2024)](https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/); [Nature (2025)](https://doi.org/10.1038/s41586-025-08998-4)
- **Core Idea**: AlphaProof combines a pre-trained language model with reinforcement learning (trained via self-play against the Lean 4 formal proof assistant) to prove mathematical theorems. At the 2024 International Mathematical Olympiad, it solved 4 out of 6 problems, achieving a silver medal score of 28/42. The system works by: (1) translating natural language problems into formal Lean statements, (2) generating proof candidates using a language model fine-tuned with RL, (3) verifying proofs using Lean's type checker (providing a ground-truth reward signal), and (4) iterating with self-play to improve. The key insight is that formal verification eliminates hallucination: either the proof checks or it does not.
- **Mathematical Foundation**: Lean 4 dependent type theory, reinforcement learning (MCTS-style search over proof steps), curriculum learning (starting with easier Mathlib lemmas and progressing to harder problems), and the connection between theorem proving and game playing (proofs as strategies against nature). The mathematical coverage spans number theory, algebra, combinatorics, and geometry at olympiad level.
- **RSA Relevance**: AlphaProof could contribute to factoring research by: (1) **Proving or disproving factoring hardness conjectures** — formalizing and attacking statements like "factoring is not in P" or "GNFS is optimal" could either confirm security assumptions or reveal unexpected weaknesses; (2) **Discovering new number-theoretic lemmas** — the RL training process explores proof space and may find useful intermediate results about prime structure, smooth numbers, or group orders; (3) **Verifying claimed factoring breakthroughs** — when papers claim to "destroy RSA" (cf. Schnorr, entry 3 in File 3), formal verification could quickly confirm or refute the claims. The limitation is that AlphaProof currently works at olympiad level, not at the frontier of research mathematics.
- **Status**: Proven (IMO results verified; proofs formally checked in Lean)
- **Open Questions**: Can AlphaProof scale to research-level problems (not just competition mathematics)? Can it prove results in algebraic number theory relevant to factoring? What is the ceiling for RL-based theorem proving — can it discover genuinely novel proof techniques, or only recombine known ones?

---

### 3. AlphaGeometry 2: Neural Geometry Solver
- **Authors**: Google DeepMind (Trieu Trinh, Yuhuai Wu, Quoc Le, He He, Thang Luong, and team)
- **Year**: 2024
- **Source**: [Nature (2024)](https://doi.org/10.1038/s41586-023-06747-5) (AlphaGeometry 1); [Google DeepMind blog](https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/) (AlphaGeometry 2)
- **Core Idea**: AlphaGeometry combines a neural language model with a symbolic deduction engine to solve olympiad-level geometry problems. The neural component proposes auxiliary constructions (new points, lines, circles) that simplify the problem, while the symbolic engine performs deductive reasoning using a fixed set of geometric rules. AlphaGeometry 2, used alongside AlphaProof at IMO 2024, solved the geometry problem (Problem 4) that contributed to the silver medal performance. The system was trained on synthetically generated geometry problems (100M+ synthetic proofs), avoiding the need for human-annotated data.
- **Mathematical Foundation**: Synthetic geometry, the deductive database method (DD) of Chou-Gao-Zhang, algebraic geometry constructions, and the interaction between neural heuristic search (for auxiliary constructions) and sound symbolic deduction (for proof steps). The synthetic data generation uses random geometric constructions with automated theorem verification.
- **RSA Relevance**: Indirect. AlphaGeometry's relevance to factoring is primarily methodological: it demonstrates the power of combining neural heuristics (for creative leaps) with symbolic verification (for correctness). This hybrid architecture could be adapted to number theory, where the "auxiliary construction" analogue might be proposing new algebraic structures, change-of-basis transformations, or lattice constructions that simplify a factoring problem. The geometric reasoning itself is not directly applicable to factoring, but the methodology of "neural creativity + formal verification" is transferable.
- **Status**: Proven (geometry results verified)
- **Open Questions**: Can the neuro-symbolic architecture be adapted to algebraic/number-theoretic domains? What is the equivalent of "auxiliary construction" for factoring problems? Can synthetic data generation work for number theory (generating random factoring-related problems with known solutions)?

---

### 4. DeepSeek R1: Reasoning-Optimized Large Language Model
- **Authors**: DeepSeek AI (Luo Fuli, Zhao Hao, and team)
- **Year**: 2025 (January)
- **Source**: [DeepSeek R1 Technical Report](https://arxiv.org/abs/2501.12948)
- **Core Idea**: DeepSeek R1 is a 671-billion-parameter Mixture-of-Experts (MoE) language model with 37 billion active parameters per token, trained for approximately $6 million — orders of magnitude cheaper than comparable Western models. It achieves 96.3% on AIME (American Invitational Mathematics Examination) and state-of-the-art performance on MATH, GSM8K, and other mathematical benchmarks. Key innovations include: (1) Group Relative Policy Optimization (GRPO) — a reinforcement learning method that uses group-level baselines instead of a separate value network; (2) Multi-head Latent Attention (MLA) — compressing KV cache for efficient inference; (3) extensive RL training on mathematical reasoning tasks with verifiable outputs.
- **Mathematical Foundation**: Transformer architecture with MoE routing, reinforcement learning from verifiable rewards (not just human preferences), chain-of-thought reasoning, and the insight that mathematical reasoning can be improved through RL because correctness is verifiable. The GRPO method avoids training a critic model by comparing outputs within a group, reducing training cost.
- **RSA Relevance**: DeepSeek R1's mathematical reasoning capabilities could be applied to: (1) **Generating factoring conjectures** — the model can reason about number-theoretic structures and propose hypotheses; (2) **Code generation for factoring** — writing and optimizing factoring code (ECM parameters, GNFS polynomial selection, lattice reduction); (3) **Analyzing mathematical literature** — processing the vast literature on factoring, Langlands, and related topics to identify unexplored connections. The limitation is that LLMs reason by analogy and pattern-matching, not by genuine mathematical insight; they can organize and suggest but not prove.
- **Status**: Proven (benchmark results verified; open-weight model available)
- **Open Questions**: Can R1-class models genuinely contribute to mathematical research (beyond benchmarks)? Can RL training on factoring-specific tasks (e.g., optimizing ECM parameters) produce useful results? How does performance scale with increased compute on number-theory-specific training?

---

### 5. DeepSeek V3: Agent Capabilities and RL-Enhanced Reasoning
- **Authors**: DeepSeek AI
- **Year**: 2025
- **Source**: [DeepSeek V3 Technical Report](https://arxiv.org/abs/2412.19437)
- **Core Idea**: DeepSeek V3 is the foundation model underlying R1, with enhanced agent capabilities: tool use, multi-step planning, code execution, and web interaction. While R1 focuses on pure reasoning (chain-of-thought), V3 is optimized for agentic workflows where the model must interact with external tools (calculators, code interpreters, databases, search engines) to solve problems. V3 uses the same MoE architecture as R1 (671B total, 37B active) but with additional training for tool-augmented reasoning.
- **Mathematical Foundation**: Same MoE/MLA architecture as R1, plus function calling, code generation and execution, retrieval-augmented generation (RAG), and multi-turn planning. The agent framework allows V3 to decompose a mathematical problem, write code to explore it computationally, verify intermediate results, and iterate.
- **RSA Relevance**: V3's agent capabilities are directly relevant to computational number theory research: (1) **Automated experimentation** — V3 could autonomously run factoring experiments (e.g., testing ECM curves, GNFS polynomials) and analyze results; (2) **Literature mining** — processing papers on factoring, Langlands, and quantum computing to identify connections; (3) **Hybrid human-AI research** — acting as a research assistant that handles computational exploration while humans guide the theoretical direction. The agent paradigm is particularly suited to the iterative, experimental nature of algorithmic number theory.
- **Status**: Proven (agent capabilities demonstrated)
- **Open Questions**: Can agentic LLMs autonomously conduct meaningful mathematical research? What is the optimal human-AI collaboration model for attacking hard problems like factoring? Can V3's code generation produce factoring implementations that are competitive with hand-optimized code (e.g., CADO-NFS)?

---

### 6. Deep Learning Approaches to Integer Factorization
- **Authors**: Various (emerging research area)
- **Year**: 2020 -- present
- **Source**: Various; see [Charfeddine et al. (2023)](https://arxiv.org/abs/2310.09405) for a survey
- **Core Idea**: Several research groups have explored using neural networks (RNNs, Transformers, and hybrid architectures) to learn the factoring function directly: given N as input, predict p and q. Approaches include: (1) **Sequence-to-sequence models** — treating factoring as translating the binary representation of N to the binary representations of its factors; (2) **Classifier approaches** — predicting individual bits of the smaller factor; (3) **Hybrid architectures** — combining neural networks with classical number-theoretic preprocessing (e.g., providing residues modulo small primes as features). Results to date show that neural networks can learn to factor numbers up to ~20-30 bits with reasonable accuracy but fail dramatically beyond this range.
- **Mathematical Foundation**: Supervised learning on (N, p, q) triples, recurrent neural networks (LSTMs, GRUs), Transformer attention mechanisms, binary/integer representation learning, and curriculum learning (training on progressively larger numbers). The fundamental challenge is that the factoring function has no known smooth, local structure — the factors of N and N+2 can be completely unrelated — making generalization extremely difficult for gradient-based learning.
- **RSA Relevance**: Currently negligible. Neural networks cannot learn to factor beyond ~30 bits, while RSA-2048 requires factoring 2048-bit numbers. The gap is not incremental; it reflects a fundamental obstacle: factoring has no exploitable local structure for gradient descent. The only plausible path to relevance would be if neural networks could learn useful intermediate representations (e.g., predicting smoothness probabilities, optimal sieving parameters) that accelerate classical algorithms, rather than learning the factoring function end-to-end.
- **Status**: Active Research (but no results approaching cryptographic relevance)
- **Open Questions**: Is there a fundamental impossibility result for neural network factoring (i.e., can we prove that no polynomial-size network can factor)? Can neural networks learn useful auxiliary functions for factoring (smoothness detection, polynomial selection)? Could a sufficiently large Transformer model learn factoring patterns that elude smaller models?

---

### 7. Ising Machine Factorization via QUBO Formulation
- **Authors**: Various (Ising machine community)
- **Year**: 2025
- **Source**: [EPJ Quantum Technology (2025)](https://link.springer.com/article/10.1140/epjqt/s40507-025-00449-9)
- **Core Idea**: This work formulates integer factorization as a QUBO (Quadratic Unconstrained Binary Optimization) problem and solves it on Ising machines — specialized hardware (optical, electronic, or quantum) designed to find ground states of Ising Hamiltonians. The factoring QUBO encodes N = p * q by representing p and q as binary strings, expanding the multiplication, and penalizing configurations where the product does not equal N. The Ising machine then searches for the ground state (zero energy = correct factorization) using simulated annealing, coherent Ising machine dynamics, or quantum annealing. The paper explores various QUBO formulations (direct binary, carry-bit, logarithmic encoding) and benchmarks them on different Ising hardware platforms.
- **Mathematical Foundation**: QUBO/Ising model equivalence (QUBO: minimize x^T Q x; Ising: minimize sum J_{ij} s_i s_j + sum h_i s_i), binary multiplication circuits as polynomial constraints, penalty methods for equality constraints, and the physics of Ising machines (optical parametric oscillators for coherent Ising machines, superconducting flux qubits for quantum annealers). The key challenge is that the number of QUBO variables scales as O(n) for an n-bit number, but the constraint density and precision requirements grow, creating an increasingly rugged energy landscape.
- **RSA Relevance**: Limited by fundamental scaling issues. While Ising machines can factor small numbers (demonstrations up to ~30 bits), the energy landscape for factoring grows exponentially rugged with number size, meaning the annealing time (or number of optical pulses) grows exponentially. There is no theoretical or empirical evidence that Ising machines can factor numbers faster than classical algorithms. The approach is primarily interesting as a benchmarking problem for Ising hardware, not as a practical attack on RSA. Some niche value exists for factoring numbers with known special structure (e.g., known bit patterns in the factors).
- **Status**: Active Research (hardware demonstrations; no scaling advantage shown)
- **Open Questions**: Can any Ising machine architecture avoid the exponential scaling of annealing time for factoring? Can problem-specific QUBO reformulations (exploiting number-theoretic structure) improve performance? Is there a hybrid Ising-classical approach that outperforms pure classical factoring for any problem size?

---

### 8. AI for Mathematics Initiative: Unified AlphaProof + AlphaGeometry + AlphaEvolve
- **Authors**: Google DeepMind (Demis Hassabis, Pushmeet Kohli, and the broader AI for Math team)
- **Year**: 2025
- **Source**: [Google DeepMind AI for Mathematics](https://deepmind.google/discover/blog/ai-driven-discovery-in-mathematics/); various publications (see entries 1-3)
- **Core Idea**: Google DeepMind's AI for Mathematics initiative aims to create a unified pipeline combining three complementary systems: (1) **AlphaProof** — formal theorem proving via RL, providing rigorous verification; (2) **AlphaGeometry** — neuro-symbolic reasoning for geometric and structural problems; (3) **AlphaEvolve** — evolutionary algorithm discovery for computational optimization. The vision is an AI mathematician that can conjecture (AlphaEvolve), prove (AlphaProof), and reason structurally (AlphaGeometry). The initiative is actively expanding beyond olympiad mathematics toward research-level problems in number theory, combinatorics, and algebra. Early results include FunSearch (2023, discovering new cap set constructions) and the matrix multiplication breakthroughs.
- **Mathematical Foundation**: The unified system combines formal verification (dependent type theory in Lean 4), reinforcement learning (self-play against proof assistants), evolutionary search (population-based optimization with LLM-guided mutations), and neural heuristic evaluation (learned value functions for proof search). The mathematical coverage is expanding from competition math toward open research questions, with a focus on problems where correctness is computationally verifiable.
- **RSA Relevance**: The unified pipeline represents the most promising AI approach to mathematical discovery relevant to factoring: (1) **AlphaEvolve** could discover novel factoring algorithms or optimize existing ones; (2) **AlphaProof** could prove properties of these algorithms (correctness, complexity bounds) or prove hardness results ruling out certain attacks; (3) **The combined system** could explore the space of Langlands-inspired approaches to factoring, formalizing conjectures and testing them computationally. The key advantage over individual systems is the feedback loop: AlphaEvolve proposes, AlphaProof verifies, and the cycle iterates. This mirrors how human mathematicians work (conjecture, verify, refine) but at much higher throughput.
- **Status**: Active Research (individual components proven; unified pipeline in development)
- **Open Questions**: Can the unified pipeline scale to research-level number theory? How long before AI systems can formalize and attack problems like "is factoring in P?" Can the evolutionary component discover factoring-relevant structures that humans have missed? What role will formal verification play in cryptanalysis — could an AI prove that a specific attack works before anyone implements it?

---

## Synthesis: AI Paths to Factoring

The AI-driven mathematical discovery landscape suggests three distinct paths by which AI could contribute to breaking RSA:

**Path 1: Algorithm Discovery (AlphaEvolve-style)**
Evolutionary search guided by LLMs could discover novel factoring algorithms or optimizations to GNFS/ECM. This is the most direct path but faces the challenge of defining a smooth fitness landscape for factoring. Current results (matrix multiplication, cap sets) suggest that AI can find structures in algebraic problems that elude humans, and factoring has rich algebraic structure. Timeline: 5-15 years for meaningful factoring-specific results.

**Path 2: Formal Mathematics (AlphaProof-style)**
RL-trained theorem provers could prove new results in number theory relevant to factoring — either positive (new structural theorems exploitable by algorithms) or negative (hardness results). The Langlands program, with its web of conjectures amenable to formal verification, is a natural target. Timeline: 10-20 years for research-level number theory results.

**Path 3: Computational Optimization (DeepSeek/Agent-style)**
LLM agents could optimize existing factoring infrastructure: better GNFS polynomials, better ECM curves, better lattice reduction strategies, better parallelization of CADO-NFS. This is the least glamorous but most immediately practical path. Timeline: 1-5 years for constant-factor improvements.

**Assessment**: No AI system currently threatens RSA. The fundamental obstacle is that factoring's hardness appears to be structural (related to deep number-theoretic properties), not merely a matter of search efficiency. AI excels at search but has not yet demonstrated the ability to overcome structural barriers in mathematics. However, the rapid progress of 2024-2025 (AlphaProof, AlphaEvolve, DeepSeek R1) suggests that AI's mathematical capabilities are improving faster than expected, and the intersection of AI with computational number theory deserves careful monitoring.
