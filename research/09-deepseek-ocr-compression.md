# DeepSeek OCR, MLA & Visual Compression

This document surveys DeepSeek's family of compression innovations -- from visual token compression in OCR to KV-cache compression via Multi-Head Latent Attention (MLA) -- and examines whether the underlying mathematical principles could inspire novel approaches to integer factorization.

---

### DeepSeek-OCR: Contexts Optical Compression
- **Authors**: DeepSeek AI Research Team
- **Year**: 2025 (October)
- **Source**: [arXiv:2510.18234](https://arxiv.org/abs/2510.18234), [GitHub](https://github.com/deepseek-ai/DeepSeek-OCR)
- **Core Idea**: DeepSeek-OCR achieves 96%+ OCR precision while compressing visual tokens by a 9-10x ratio. The key insight is that most visual tokens in document images are redundant for text extraction. By aggressively compressing the visual representation before feeding it to the LLM decoder, the system dramatically reduces inference cost without sacrificing accuracy. The architecture uses a DeepEncoder that combines window-based local attention (built on SAM-base) with global attention (built on CLIP-large) to capture both fine-grained character details and document-level layout structure.
- **Mathematical Foundation**: Information-theoretic compression -- the system learns to identify and discard visually redundant tokens while preserving the information-theoretically necessary content. Window attention partitions the image into local receptive fields (analogous to wavelet decomposition), while global attention captures long-range dependencies (analogous to low-frequency components in Fourier analysis). The dual-scale architecture mirrors multi-resolution analysis from signal processing.
- **RSA Relevance**: The core principle -- that high-dimensional data often lives on a much lower-dimensional manifold -- is directly relevant. A semiprime N = p * q has enormous apparent complexity in its digit representation but is fully determined by just two numbers. If we could learn the right "compression" of number-theoretic representations of N that preserves factor-relevant structure while discarding noise, we might expose the factors.
- **Status**: Proven
- **Open Questions**: Can the dual-scale attention approach (local + global) be adapted to number-theoretic data where "local" means digit-level patterns and "global" means modular arithmetic properties? What is the theoretical minimum compression ratio for number-theoretic data that preserves factor information?

---

### DeepSeek-OCR2: Advanced Visual Encoding
- **Authors**: DeepSeek AI Research Team
- **Year**: 2026 (January)
- **Source**: DeepSeek technical reports
- **Core Idea**: DeepSeek-OCR2 introduces a new visual encoding architecture that achieves further compression gains over the original. Complex document pages are encoded in 256-1,120 tokens depending on visual complexity, compared to thousands of tokens in standard vision-language models. The architecture adapts token count dynamically based on content complexity -- simple text pages use fewer tokens while complex tables or mixed-media pages use more.
- **Mathematical Foundation**: Adaptive-rate compression. Rather than a fixed compression ratio, OCR2 learns a content-dependent encoding length. This connects to rate-distortion theory in information theory: for a given acceptable distortion level, there exists an optimal encoding rate, and that rate varies with source complexity. The system approximates the rate-distortion function for document images.
- **RSA Relevance**: The adaptive compression idea is intriguing for factorization. Different semiprimes may have different "compressibility" depending on the relationship between their factors. Semiprimes where p and q are close together (Fermat-vulnerable) might compress differently than semiprimes with very different-sized factors. An adaptive compression system might learn to detect these structural differences.
- **Status**: Proven
- **Open Questions**: What determines the compression "difficulty" of a semiprime? Is there a meaningful rate-distortion curve for number-theoretic representations? Could an adaptive encoder learn to allocate more "tokens" to harder-to-factor numbers, and would analyzing where it allocates attention reveal factor-correlated structure?

---

### DeepSeek MLA (Multi-Head Latent Attention)
- **Authors**: DeepSeek AI (introduced in DeepSeek-V2)
- **Year**: 2024
- **Source**: [arXiv:2405.04434](https://arxiv.org/abs/2405.04434)
- **Core Idea**: MLA compresses the KV cache to low-dimensional latent vectors via learned low-rank projections. Instead of caching full key and value vectors for each attention head, MLA projects them into a shared low-rank latent space and reconstructs them on-the-fly during attention computation. This achieves a 32x compression ratio on the KV cache and a 20x inference speedup, using only ~4% of the KV cache memory that standard Multi-Head Attention (MHA) would require for large models. The critical insight is that keys and values across different heads are highly correlated and can be jointly compressed without meaningful information loss.
- **Mathematical Foundation**: Low-rank matrix approximation and SVD-like compression. Given a KV cache matrix M of rank r, MLA finds matrices U (d x k) and V (k x n) where k << r such that M ~ UV. The projection W_down: R^d -> R^k compresses, and W_up: R^k -> R^d reconstructs. This is mathematically related to the Eckart-Young-Mirsky theorem, which proves that the best rank-k approximation to a matrix (in Frobenius or spectral norm) is given by truncated SVD. MLA learns a task-specific variant of this via gradient descent.
- **RSA Relevance**: This is perhaps the most directly relevant technique. Lattice-based factoring (Schnorr, Regev) already works by finding short vectors in high-dimensional lattice spaces -- which is fundamentally a problem of finding low-rank structure in high-dimensional geometry. MLA demonstrates that neural networks can learn highly effective low-rank projections for complex, high-dimensional data. If we could train a model to compress modular arithmetic data (residues of N modulo various bases, multiplicative group structure, etc.) into a latent space, the learned projection might implicitly encode factor structure. The key-value duality in attention even mirrors the multiplicative duality of p and q in N = pq.
- **Status**: Proven
- **Open Questions**: Can MLA-style compression be applied to number-theoretic data? What would the "queries," "keys," and "values" be in a factorization context? If we train MLA on a dataset of (semiprime, modular residues) pairs, does the learned latent space cluster by factor properties? What is the theoretical minimum latent dimension needed to preserve factor information for n-bit semiprimes?

---

### DeepSeek V2/V3 MoE Architecture
- **Authors**: DeepSeek AI
- **Year**: 2024-2025
- **Source**: DeepSeek-V2 and V3 technical reports
- **Core Idea**: DeepSeek's V2 and V3 models use a Mixture of Experts (MoE) architecture with 671B total parameters but only 37B active per token. This "sparse activation" pattern means the full model has enormous capacity for learning diverse patterns, but any individual input only activates a small, relevant subset of that capacity. Training uses Group Relative Policy Optimization (GRPO) for reinforcement learning, which is more sample-efficient than PPO for language model alignment.
- **Mathematical Foundation**: Mixture of Experts is grounded in ensemble methods and gating networks. The router function g(x) = softmax(W_g * x) assigns each input to the top-k experts. The output is y = sum_i g_i(x) * E_i(x) where E_i are expert networks. GRPO extends REINFORCE with group-normalized baselines: the advantage of a response is computed relative to the mean reward of a group of sampled responses, eliminating the need for a separate value model.
- **RSA Relevance**: The MoE paradigm mirrors a key aspect of factorization: different semiprimes may require fundamentally different approaches depending on their structure. A semiprime with smooth factors (Pollard p-1 vulnerable), a semiprime with factors near sqrt(N) (Fermat vulnerable), and a "hard" semiprime with random factors each benefit from different algorithms. An MoE model could learn specialized "expert" subnets for different factor structures, with the router learning to detect which structure is present -- effectively learning a meta-algorithm that selects the right factoring approach.
- **Status**: Proven
- **Open Questions**: Could an MoE model learn to route semiprimes to different expert networks that specialize in different factor structures? Would the learned routing function itself reveal information about what makes certain numbers easier to factor? Can GRPO be used to train a factoring model where the reward is inversely proportional to computation time?

---

### MLA Compression Principle: Low-Rank Latent Space Projection
- **Authors**: Conceptual synthesis from DeepSeek MLA
- **Year**: 2024-2026
- **Source**: Derived from MLA architecture analysis
- **Core Idea**: The fundamental MLA mechanism projects queries Q, keys K, and values V into a shared low-rank latent space via a compression matrix W_down, then reconstructs K and V from the same compressed representation via separate up-projection matrices W_K_up and W_V_up. The key mathematical insight is that K and V share a common compressed representation c = W_down * x, and then K = W_K_up * c, V = W_V_up * c. This means the information needed for both matching (K) and retrieval (V) can be jointly encoded in a much smaller space.
- **Mathematical Foundation**: Joint compression of correlated signals. If K and V are drawn from distributions with shared structure (high mutual information), they can be jointly compressed more efficiently than separately. This connects to the Slepian-Wolf theorem in distributed source coding: correlated sources can be compressed at their joint entropy rate, which is less than the sum of their individual entropy rates. In the factorization context, the "key" (identifying which modular residue pattern a number has) and "value" (what the factors are) share the deep structure of N = pq.
- **RSA Relevance**: This is exactly what we want to do with number-theoretic data. Consider representing a semiprime N by its residues modulo many small primes: r_2, r_3, r_5, r_7, ... By the Chinese Remainder Theorem, this representation fully determines N (up to a bound). These residues live in a high-dimensional space, but the fact that N = pq means they all share a hidden low-rank structure determined by just two numbers. MLA-style compression should, in principle, be able to recover this low-rank structure. Train a model where the "input" is the residue vector, the latent compression captures the essential structure, and the "reconstruction" attempts to predict some factor-correlated quantity. If the compression works well, the latent space has learned something about factors.
- **Status**: Speculative (as applied to factorization)
- **Open Questions**: What is the right input representation for semiprimes? CRT residues? Binary digits? A combination? How many latent dimensions would the compression need? (Naively, 2 -- one for each factor -- but the encoding is nonlinear so it could be more.) Would gradient-based learning of the compression matrices converge, or is the loss landscape too rough? Could we verify the approach on small semiprimes and scale up?

---

## Synthesis: MLA Compression as Inspiration for Factorization

The thread connecting all five entries above is a single powerful idea: **high-dimensional data with hidden low-rank structure can be efficiently compressed, and the compression itself reveals the structure**.

DeepSeek's innovations demonstrate this at industrial scale:
- OCR compresses visual tokens 9-10x because document images are highly structured
- MLA compresses KV caches 32x because attention heads are highly correlated
- MoE activates only 5.5% of parameters because inputs are diverse but each individual input only needs a small subset of model capacity

Integer factorization is, at its core, a problem of hidden low-rank structure. A semiprime N = pq lives in a space of n-bit integers (dimension 2^n), but it is fully determined by two numbers of roughly n/2 bits each. The challenge is that the multiplication function scrambles the relationship between {p, q} and the digit representation of N in a way that appears random.

**The MLA-inspired approach to factorization would work as follows:**

1. **Representation**: Encode N as a high-dimensional vector using multiple number-theoretic features: CRT residues, digit patterns in multiple bases, quadratic residue symbols, continued fraction coefficients, etc.

2. **Compression**: Train a neural network to compress this high-dimensional representation into a low-dimensional latent space, where the training objective is to predict some factor-correlated quantity (e.g., residues of p modulo small primes, or whether N is divisible by a given number).

3. **Extraction**: If the compression succeeds, the latent representation has captured information about the factors. Analyze the learned compression matrices to understand what number-theoretic features are most informative.

4. **Scaling**: Test whether the approach generalizes from small semiprimes (where we can verify) to larger ones (where factoring is hard).

The key open question is whether neural network optimization (gradient descent on a loss function) can navigate the loss landscape of this problem. The multiplication function is a polynomial map, so its "inverse" should be representable by a sufficiently deep network -- but representability does not guarantee learnability. The MLA experience suggests that low-rank structure, when present, is learnable -- but the factorization problem's low-rank structure may be qualitatively harder to learn than the correlations between attention heads.

This remains one of the most promising speculative directions because it leverages proven engineering (MLA works at scale) and connects to established mathematics (low-rank approximation, CRT, lattice methods), while targeting a concrete computational goal (factor extraction from latent representations).
