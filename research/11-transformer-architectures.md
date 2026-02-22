# Transformer Architectures & Computational Duality

This document surveys the rapidly evolving landscape of sequence modeling architectures -- transformers, state-space models (SSMs), and hybrids -- with particular attention to the recently proven mathematical duality between attention and recurrence. We examine what these architectural insights might imply for computational problems like integer factorization.

---

### State Space Duality (SSD) / Mamba-2
- **Authors**: Tri Dao and Albert Gu
- **Year**: 2024
- **Source**: [arXiv:2312.00752](https://arxiv.org/abs/2312.00752) (Mamba-1); Mamba-2 paper on State Space Duality (SSD)
- **Core Idea**: Dao and Gu proved a remarkable mathematical equivalence: structured state-space models (SSMs) with scalar-valued, identity-structured state matrices are exactly equivalent to a form of masked self-attention with a 1-semiseparable causal mask matrix. Specifically, the output of an SSM with diagonal state matrix A, input projection B, and output projection C can be written as y = M * (x), where M is a lower-triangular matrix whose (i,j) entry is C_i * A^{i-j} * B_j -- and this matrix M is exactly a 1-semiseparable matrix, which is the class of matrices that can be computed using structured matrix multiplication algorithms. This duality means that the sequential recurrence of an SSM (process tokens one at a time, maintaining a hidden state) and the parallel matrix operation of attention (process all tokens simultaneously via matrix multiplication) are not different computations but the SAME computation expressed in dual forms.
- **Mathematical Foundation**: Semiseparable matrices and their algebraic properties. A matrix M is (r,s)-semiseparable if every submatrix strictly below the diagonal has rank at most r and every submatrix strictly above has rank at most s. The 1-semiseparable causal (lower-triangular) matrices form an algebra closed under multiplication and inversion. The SSM recurrence h_t = A * h_{t-1} + B_t * x_t, y_t = C_t * h_t generates exactly this matrix structure. The duality extends to a computational duality: the recurrence is O(Tn) sequential work (T = sequence length, n = state dimension), while the matrix form is O(T^2 * n) parallel work but can be chunked into O(T/c * c^2 * n) hybrid computation that exploits both sequential and parallel hardware.
- **RSA Relevance**: The duality theorem has a profound conceptual implication for factorization. It proves that certain computations that appear fundamentally sequential (processing one token at a time through a recurrence) are actually equivalent to parallel matrix operations. Factoring by trial division is sequential: test each candidate divisor one at a time. Is there a "dual" parallel form of factoring that computes the same result via a matrix operation? The SSD framework suggests we should look for factoring algorithms that can be expressed as both a recurrence and a matrix multiplication, because the matrix form might be more efficiently parallelizable.
- **Status**: Proven
- **Open Questions**: Can the SSD framework be extended beyond linear recurrences to the nonlinear computations involved in modular arithmetic? Is there a semiseparable matrix structure hidden in the modular exponentiation operation a^x mod n? Could we formulate trial division as a recurrence and then find its "dual" parallel form using the SSD machinery?

---

### Gated Attention
- **Authors**: Various (NeurIPS 2025 Best Paper)
- **Year**: 2025
- **Source**: NeurIPS 2025 proceedings
- **Core Idea**: Gated attention introduces a sigmoid gate after each attention head, allowing the model to dynamically modulate the contribution of attention outputs. The gate g = sigmoid(W_g * x + b_g) is applied element-wise to the attention output: y = g * Attention(Q, K, V). This seemingly simple modification has outsized effects. It reduces "attention sinks" -- the phenomenon where attention concentrates on the first token or special tokens regardless of relevance -- by allowing the gate to suppress uninformative attention patterns. This improves training stability, reduces loss spikes, and enables more uniform gradient flow. The gated architecture achieves better perplexity than standard transformers at the same parameter count, with particular improvements on long-context tasks.
- **Mathematical Foundation**: The gate can be understood as a learned soft mixture between "attend" and "pass-through" modes. Without the gate, each layer must route all information through attention, even when the input already contains the needed information. The gate allows y = g * Attn(x) + (1-g) * 0, effectively letting the layer "opt out" of attention when it isn't helpful. This relates to the theory of residual learning (He et al., 2016) and highway networks (Srivastava et al., 2015): the gate ensures that the identity function is always easy to learn, preventing gradient degradation.
- **RSA Relevance**: The gating mechanism is relevant for any neural approach to factorization. A model trying to learn factoring patterns would need to selectively attend to relevant digit patterns while ignoring irrelevant ones. The attention sink problem is particularly acute for number-theoretic data: a model might fixate on the most significant digits (analogous to a "first token" sink) when the factor-relevant information is distributed across all digit positions. Gated attention could help a factoring model learn to attend to the right number-theoretic features.
- **Status**: Proven
- **Open Questions**: Would gated attention improve the performance of transformers trained on factoring tasks? Can the learned gate patterns reveal which features of a number's representation are most informative for factoring?

---

### Jamba: Transformer + Mamba + MoE Hybrid
- **Authors**: AI21 Labs
- **Year**: 2024
- **Source**: AI21 Labs technical report
- **Core Idea**: Jamba is a production hybrid architecture that interleaves transformer attention layers, Mamba SSM layers, and Mixture of Experts (MoE) layers. The 52B parameter model (12B active per token) achieves 256K context length with 3x throughput compared to similarly-sized pure transformers. The key design insight is that different types of information processing benefit from different architectural components: attention excels at precise content-based retrieval (finding a specific fact in context), SSM excels at aggregating sequential patterns (understanding temporal dynamics), and MoE excels at routing different types of inputs to specialized processors. By interleaving all three, Jamba gets the benefits of each without the costs of applying any single mechanism uniformly.
- **Mathematical Foundation**: The architecture can be analyzed through the lens of operator composition. An attention layer computes a data-dependent linear combination (the attention matrix is a function of the input). An SSM layer computes a fixed linear recurrence with input-dependent modulation. An MoE layer computes a sparse mixture of nonlinear transformations. The composition of these operators creates a function class that is strictly more expressive than any single component repeated -- this follows from the universality theorems for deep networks combined with the structural biases of each component.
- **RSA Relevance**: A Jamba-style hybrid architecture could be ideal for a factoring model. Attention layers could perform content-based retrieval: "what modular residues are consistent with each other?" SSM layers could track sequential patterns: "as we examine larger candidate factors, how does the residue pattern evolve?" MoE layers could route different types of semiprimes to specialized experts: "this number has factors close to sqrt(N), use the Fermat expert." The hybrid approach acknowledges that factoring likely requires multiple types of computation, not just one.
- **Status**: Proven
- **Open Questions**: What is the optimal interleaving pattern for a factoring-focused hybrid model? Should SSM layers dominate (since modular arithmetic is inherently sequential) or attention layers (since factor search requires content-based matching)? Would the MoE routing learn meaningful classifications of semiprimes?

---

### Falcon Mamba-7B: Attention-Free Language Modeling
- **Authors**: Technology Innovation Institute (TII)
- **Year**: 2024
- **Source**: TII technical report; Hugging Face release
- **Core Idea**: Falcon Mamba-7B is the first pure SSM model (no attention layers whatsoever) to match or exceed the performance of same-sized transformer models on standard benchmarks. This is a landmark result because it proves that attention is not necessary for strong language modeling -- sequential state-space computation alone is sufficient. The model uses the Mamba-1 architecture with optimized training recipes. On benchmarks like MMLU, HellaSwag, and WinoGrande, Falcon Mamba-7B matches Llama-2-7B and Mistral-7B despite using fundamentally different computation. The model has O(1) memory per token during inference (constant state size) compared to O(T) for transformers (growing KV cache), making it inherently more efficient for long sequences.
- **Mathematical Foundation**: The result can be understood through the lens of computational complexity. Transformers compute with O(T^2) attention, which in principle allows arbitrary pairwise comparisons between all positions. SSMs compute with O(T) sequential operations and a fixed-size hidden state, which in principle limits them to patterns that can be captured by finite-state dynamics. The fact that SSMs match transformers on practical tasks suggests that the O(T^2) capacity of attention is rarely needed -- most useful computations can be captured by O(T) sequential processing with sufficient state dimension. This connects to the theory of finite automata and regular languages vs. context-free/context-sensitive languages.
- **RSA Relevance**: If attention-free models can learn complex patterns, they might learn number-theoretic patterns too. An SSM model could process the digits of a semiprime sequentially, maintaining a hidden state that accumulates information about the factor structure. The fixed state size is both a limitation (can the state capture enough about a 2048-bit number?) and a feature (the compression from 2048 bits to a fixed state might FORCE the model to learn compact factor representations). This connects directly to the MLA compression idea from the DeepSeek research.
- **Status**: Proven
- **Open Questions**: Can pure SSMs learn modular arithmetic? (Recent work suggests transformers struggle with this -- do SSMs fare better or worse?) Is the sequential nature of SSMs an advantage for factoring, since modular reduction is naturally sequential? What state dimension would be needed to represent factoring-relevant information about n-bit numbers?

---

### Sub-Quadratic Attention Alternatives: The Broader Landscape
- **Authors**: Various (Katharopoulos et al. for linear attention; Peng et al. for RWKV; Sun et al. for RetNet)
- **Year**: 2020-2025
- **Source**: Various; survey by Tay et al., "Efficient Transformers: A Survey," ACM Computing Surveys, 2022
- **Core Idea**: The quadratic cost of standard attention (O(T^2) in sequence length) has driven a wave of sub-quadratic alternatives. Linear attention replaces softmax(QK^T)V with phi(Q)(phi(K)^T V), which can be computed in O(T * d^2) using the associative property of matrix multiplication. RWKV combines the parallelism of transformers during training with the efficiency of RNNs during inference, using a learned exponential decay mechanism. RetNet uses multi-scale retention with explicit exponential decay, achieving O(T) inference with a recurrence formulation and O(T * log T) training with a chunk-wise parallel formulation. Investment in non-transformer architectures grew approximately 400% between 2023 and 2025, reflecting both commercial interest and genuine scientific progress.
- **Mathematical Foundation**: Linear attention's key insight: softmax attention computes a_ij = exp(q_i^T k_j) / sum_j exp(q_i^T k_j), which requires materializing the T x T attention matrix. If we replace exp with a feature map phi, then phi(q_i)^T phi(k_j) can be rewritten using the kernel trick: a_ij = phi(q_i)^T (sum of phi(k_j) * v_j^T) / phi(q_i)^T (sum of phi(k_j)). The inner quantity sum of phi(k_j) * v_j^T is a d x d matrix that can be maintained as a running sum, giving O(1) per-token computation. RetNet's retention mechanism: r_{i,j} = gamma^{i-j} * (q_i^T k_j) for explicit exponential decay, which is a structured (Toeplitz-like) attention pattern.
- **RSA Relevance**: The proliferation of efficient attention alternatives matters for any neural factoring approach because factoring large numbers requires processing long sequences (a 2048-bit RSA key is a 617-digit number). Standard attention would scale poorly. Linear attention and RetNet provide O(T) alternatives that could handle long number representations. Moreover, the exponential decay in RetNet and RWKV has a natural number-theoretic interpretation: more significant digits should have more influence on factor predictions, and exponential decay from MSB to LSB is a reasonable inductive bias.
- **Status**: Proven (architectures work; 400% investment claim is industry estimate)
- **Open Questions**: Which sub-quadratic architecture is best suited for number-theoretic computation? Does the kernel trick in linear attention have a number-theoretic interpretation? Could the "decay rate" in RetNet be set based on number-theoretic principles rather than learned?

---

### Mamba-2 SSD Algorithm: Efficient Matmul-Based Computation
- **Authors**: Tri Dao and Albert Gu
- **Year**: 2024
- **Source**: Mamba-2 paper (SSD framework)
- **Core Idea**: The SSD (Structured State Space Duality) algorithm is the efficient computational realization of the SSM-attention duality. Rather than computing the SSM recurrence sequentially (O(T*n) time, no parallelism) or the full attention matrix (O(T^2) time, full parallelism), SSD chunks the sequence into blocks of size c and uses a hybrid strategy. Within each chunk, it computes the equivalent attention matrix (c x c, small enough to fit in fast memory). Across chunks, it uses the SSM recurrence to propagate state. This gives O(T/c * c^2) = O(T*c) total work, with the parallelism of matmul within chunks and the efficiency of recurrence across chunks. By tuning c to match hardware characteristics (GPU SRAM size), SSD achieves 2-8x speedups over Mamba-1 while computing the same function.
- **Mathematical Foundation**: The SSD algorithm exploits the semiseparable structure of the SSM output matrix. A block-decomposition of a semiseparable matrix yields: M = block_diag(M_11, M_22, ...) + low_rank_correction. The block-diagonal terms correspond to within-chunk attention (computed via matmul), and the low-rank correction corresponds to cross-chunk state propagation (computed via recurrence). This decomposition is exact, not an approximation. The chunk size c controls the tradeoff between parallelism (larger c = more within-chunk matmul) and memory (larger c = larger attention matrices to store).
- **RSA Relevance**: The chunked computation strategy maps naturally to factoring. A long number can be decomposed into digit-chunks, with within-chunk patterns captured by parallel attention and cross-chunk patterns captured by sequential state propagation. For factoring, "within-chunk" patterns might correspond to local digit correlations (which are weak but exist for semiprimes), while "cross-chunk" patterns correspond to carry propagation in multiplication (which creates long-range dependencies). The SSD framework provides a principled way to handle both.
- **Status**: Proven
- **Open Questions**: Can the SSD chunk-based computation be adapted to modular arithmetic, where the "state" propagated across chunks is a modular residue? Would the optimal chunk size for number-theoretic data differ from natural language (where c = 64-256 is typical)? Could the semiseparable matrix structure of SSMs be related to the structure of multiplication tables in Z/nZ?

---

## Synthesis: Architectural Duality and Factorization

The entries above converge on three key insights for integer factorization:

### (a) Training Transformers on Factoring

The most direct application: train a sequence model (transformer, SSM, or hybrid) on the mapping (semiprime -> factors). Recent work has shown that transformers can learn modular arithmetic and even simple factoring for small numbers. The architectural innovations surveyed here -- gated attention for better feature selection, MoE for routing different number types to specialized experts, efficient sub-quadratic attention for handling long digit sequences -- all improve the prospects for this direct approach.

The key question is whether such models can generalize: if trained on 32-bit semiprimes, can they factor 64-bit semiprimes? Current evidence suggests limited generalization, but the architectures are improving rapidly.

### (b) Sequential-Parallel Duality and Factoring

The Mamba-2 SSD theorem is not just an engineering optimization -- it is a deep mathematical statement about computational duality. It proves that certain sequential computations (recurrences) have exact parallel equivalents (matrix multiplications). This raises a tantalizing question: **is factoring a computation that has such a dual form?**

Trial division is inherently sequential: test d = 2, 3, 5, 7, ... and check if d | n. But what if this sequential search can be reformulated as a matrix operation? The SSD framework shows that this is possible when the sequential computation has the right algebraic structure (specifically, when it corresponds to a linear recurrence with scalar state). Trial division is not a linear recurrence -- but could a reformulation of factoring (e.g., via lattice methods or the quadratic sieve) have the right structure?

The quadratic sieve, in fact, already has a parallel component: the sieving step is embarrassingly parallel, and the final step is a linear algebra problem (finding null vectors of a matrix over GF(2)). The SSD lens suggests looking for MORE parallelism in the supposedly sequential parts.

### (c) MoE Sparse Activation and Factorization Decomposition

The MoE pattern -- having many "experts" but activating only a few per input -- mirrors a key aspect of factoring: different numbers require different approaches. The Number Field Sieve, for all its generality, has many special cases and parameter choices that depend on the specific number being factored. An MoE model could learn this meta-level optimization: which algebraic structures to search, which sieve parameters to use, which lattice dimensions to try.

More speculatively, MoE sparse activation parallels the sparse factor structure of semiprimes themselves. A semiprime n = pq can be "decomposed" into two factors, just as an MoE decomposes computation into a few active experts. If the routing function of an MoE learned something about the factor structure of its input numbers, the routing decision itself would contain information about the factors.

### Toward a Unified Architecture for Factoring

A hypothetical "factoring-optimal" architecture might combine:
- **SSM backbone** for efficient sequential processing of long digit sequences
- **Sparse attention layers** at key positions for content-based matching of digit patterns
- **MoE routing** to specialize different model components for different number structures
- **Gated outputs** to suppress uninformative computations
- **SSD chunking** to process large numbers in hardware-efficient blocks

This architecture would leverage the proven engineering of modern sequence models while targeting the specific computational structure of integer factorization. Whether it could learn to factor numbers beyond the reach of direct training data remains the central open question.
