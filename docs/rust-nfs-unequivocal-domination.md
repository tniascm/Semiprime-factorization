# Rust-NFS: The Path to Unequivocal Domination

To achieve a 100x to 1000x constant-factor speedup over CADO-NFS and secure an insurmountable asymptotic advantage as bit sizes scale ($N \ge 130$ bits), incremental software engineering is insufficient. We must fundamentally alter the Number Field Sieve (NFS) architecture to align with post-2020 hardware realities (Massive SIMT compute, High-Bandwidth Memory, RDMA) and leverage cutting-edge 2024 mathematical breakthroughs.

This roadmap outlines the multi-pronged strategy for absolute dominance over legacy implementations.

---

## Pillar 1: Algorithmic Supremacy (Breaking the Asymptotic Wall)

The primary bottleneck of classic NFS is the Canfield-Erdős-Pomerance theorem: the probability of a random number being $B$-smooth decays as $u^{-u}$ (where $u = \frac{\log x}{\log B}$). CADO-NFS randomly sieves massive arrays hoping to stumble upon smooth numbers. As $N$ grows, this probability plummets, creating an asymptotic wall.

### 1. Constructive Smoothness via LLL/BKZ Lattice Sampling (Pilatte's Breakthrough)
**The 1000x Shift:** Instead of *searching* for smooth numbers, we *construct* them. Based on Pilatte's 2024 proofs using L-function zero-density estimates, we can guarantee that elements of $(\mathbb{Z}/N\mathbb{Z})^*$ can be written as short products of the first $d = O(\sqrt{n})$ primes.
*   **The Mechanism:** Construct a short product lattice: $\mathcal{L} = \{ \vec{e} \in \mathbb{Z}^d : \prod p_i^{e_i} \equiv 1 \pmod N \}$. Run highly optimized LLL/BKZ lattice reduction algorithms on this lattice. The shortest vectors directly yield the smoothest relations.
*   **Scaling Benefit:** The time to find a relation shifts from exponential search (sieving) to polynomial-time lattice reduction. As bit sizes increase, this constructive approach fundamentally outpaces random sieving.

### 2. Quantum-Inspired Tensor Network Sampling
**The 100x Shift:** Emulate Regev's quantum sampling step using classical Tensor Networks (TTN/MPS). Instead of brute-force random walks, we use tensor contractions to sample the smooth probability distribution. This heavily biases the search space toward the densest pockets of smooth relations, skipping the "empty" spaces that CADO-NFS blindly sieves through.

---

## Pillar 2: Hardware Dominance (The Era of Massive SIMT)

CADO-NFS was built for x86 CPU clusters. Modern compute is dominated by GPUs with thousands of cores and HBM (High-Bandwidth Memory).

### 3. GPU-Native Bucket Sieving & Metal/CUDA Kernels
**The 100x Shift:** Sieving is an aggressively memory-bound operation.
*   **The Mechanism:** Move the entire bucket sieve to the GPU. Instead of iterating through lines, we use parallel prefix sums and atomic scatter additions in GPU memory. An Apple Silicon M3 Max has ~400 GB/s of unified memory bandwidth; an NVIDIA H100 has over 3 TB/s. CADO-NFS on a fast CPU tops out around 50-80 GB/s. A fully GPU-native sieve operates at 50x to 100x the raw throughput of a CPU implementation.

### 4. Tensor Core GF(2) Matrix Multiplication for Block Wiedemann
**The 1000x Shift:** The Linear Algebra (LA) stage requires finding the nullspace of massive, sparse, binary matrices.
*   **The Mechanism:** Block Wiedemann requires repeatedly multiplying the sparse matrix by blocks of dense vectors. Modern GPUs possess Tensor Cores specifically designed for rapid, low-precision matrix multiplication. We can encode GF(2) arithmetic into standard FP16/INT8 Tensor Core operations, achieving tera-operations per second (TOPS) that no CPU cluster can match. This guarantees that LA never becomes the bottleneck as $N$ scales to 200+ bits.

### 5. GPU-Batch Elliptic Curve Method (ECM)
**The 10x Shift:** Cofactorization (verifying if the unsieved remainder is smooth) is a massive time sink.
*   **The Mechanism:** GPU-ECM. We batch 100,000 survivors at a time and launch thousands of concurrent ECM curves per survivor on the GPU. This reduces the cofactorization phase to virtually zero wall-clock time.

---

## Pillar 3: Architectural Innovation (Zero-I/O and ML Sparsification)

### 6. Zero-Copy RDMA Clustering
**The Constant Speedup:** CADO-NFS spends an exorbitant amount of time serializing relations to gigabytes of text files, transferring them via SSH/rsync, and parsing them back into memory.
*   **The Mechanism:** As $N$ grows, the relation dataset exceeds single-node RAM. We implement an RDMA (Remote Direct Memory Access) cluster architecture. Worker nodes directly write relation structs into the memory space of the master node via PCIe/Infiniband without involving the CPU or OS network stack. The pipeline (Sieve $\rightarrow$ Filter $\rightarrow$ LA) becomes a continuous, lock-free, zero-copy stream.

### 7. Graph Neural Network (GNN) Driven Matrix Sparsification
**The Asymptotic Shift:** CADO-NFS uses deterministic clique-finding (Cavallar's algorithm) to merge Large Primes (LP) and reduce the matrix size. However, finding the optimal cycle basis is NP-hard.
*   **The Mechanism:** Train a lightweight Graph Neural Network (GNN) to predict the minimal-weight cycle basis in the relation graph. By dropping edges (relations) that the GNN predicts will result in dense rows, we construct an exponentially sparser matrix than CADO-NFS. A sparser matrix drastically accelerates the Block Wiedemann sequence generation phase.

---

## Summary of the Scaling Trajectory

| Optimization | CADO-NFS Approach | Rust-NFS Dominance Strategy | Impact (Constant) | Impact (Asymptotic) |
| :--- | :--- | :--- | :--- | :--- |
| **Relation Finding** | Random array sieving | Constructive LLL Lattice Sampling | 10x | **Exponential Shift** |
| **Sieve Execution** | CPU cache-blocking | Massive SIMT GPU Scatter/Gather | 100x | Linear scaling with HBM |
| **Cofactorization** | Single-thread CPU ECM | Batched GPU-ECM | 10x | Flat wall-time |
| **Data Pipeline** | File I/O (Gigabytes of text) | Zero-Copy RDMA Memory Streaming | 50x | Eliminates I/O wall |
| **Matrix Filtering** | Deterministic Clique Finding | GNN-guided Graph Sparsification | 5x | Exponentially smaller LA |
| **Linear Algebra** | CPU/Basic GPU Block Wiedemann | Tensor Core GF(2) Multiplication | 100x - 1000x | $O(n^2)$ with microscopic constants |

By pursuing this roadmap, `rust-nfs` transitions from a "fast Rust port" to a fundamentally distinct, next-generation factoring engine capable of breaking cryptographic records.