# Rust-NFS vs CADO-NFS: Inefficiencies Analysis and Improvement Plan

This document outlines the key inefficiencies in `rust-nfs` across the major stages of the Number Field Sieve (NFS) algorithm compared to the highly optimized CADO-NFS implementation. The findings are based on an analysis of the current source code and are ranked by their potential impact (reward) relative to their implementation complexity.

## Stage 1: Polynomial Selection (`gnfs/src/polyselect.rs`)

**Current State in `rust-nfs`:**
- Implements a very naive base-m method: `m = floor(N^(1/d))`, expanding `N` in base `m`.
- Attempts to find variants simply by subtracting `k` from `m` (`m = m - k`).
- No use of optimization metrics like Murphy E, Murphy Alpha, or root sieving.

**CADO-NFS State:**
- Employs sophisticated algorithms (e.g., Kleinjung's algorithm) to find polynomials with exceptionally good size (small coefficients) and high number of roots modulo small primes (root optimization).
- Uses Murphy E score and Murphy Alpha to rank and select the best polynomials, massively reducing the required sieving time.

**Inefficiency:** Critical. A poor polynomial drastically increases the sieving volume and time required to find smooth relations. This is likely the single biggest performance gap.

## Stage 2: Sieve (`rust-nfs/src/sieve/`)

**Current State in `rust-nfs`:**
- Has a reasonable bucket sieve architecture with cache-friendly 3-byte updates (`bucket.rs`).
- The small sieve (`small.rs`) uses simple loops over the sieve array.
- Norm initialization (`norm.rs`) computes values using `f64` arithmetic inside the inner loop, converting to `u8` log-norms.
- Franke-Kleinjung lattice reduction is implemented but scattering updates could be more optimized.
- Root finding for special-qs uses a fallback `find_roots_mod_p` if not in the factor base cache.

**CADO-NFS State:**
- Heavily optimized assembly/SIMD for norm initialization and bucket application.
- Uses exact, integer-based or highly tuned fixed-point math for norms instead of `f64`.
- Sophisticated Franke-Kleinjung lattice walker to minimize branching during update scattering.
- Block-based or line-based small sieve optimizations.

**Inefficiency:** High. While the structural design of the bucket sieve is correct, the inner-loop micro-optimizations (like avoiding `f64` for norms) are missing.

## Stage 3: Filtering (`rust-nfs/src/filter.rs` & `rust-nfs/src/partial_merge.rs`)

**Current State in `rust-nfs`:**
- Basic deduplication and sparse singleton removal (`filter.rs`).
- `partial_merge.rs` implements a 2-Large Prime (2LP) merge using a simple Union-Find algorithm to find cycles.
- Merged relations are simply represented as sets with zero parity for LP keys.

**CADO-NFS State:**
- Multi-stage filtering pipeline: `dup1`/`dup2`, `purge`, `merge`.
- Performs $n$-way merges (handling 3LP, 4LP, etc., not just 2LP).
- Uses sophisticated clique-finding and spanning-tree algorithms (e.g., Cavallar's weight minimization) to produce a smaller, sparser matrix for linear algebra.

**Inefficiency:** Medium to High. CADO's filtering produces a significantly smaller and sparser matrix, which directly speeds up the linear algebra stage.

## Stage 4: Linear Algebra (`gnfs/src/linalg.rs`)

**Current State in `rust-nfs`:**
- A basic pre-elimination step (`find_dependencies_with_preelim`) removes weight-1 and merges weight-2 columns.
- Uses dense Gaussian Elimination (GE) via XORing `BitRow` vectors, which is an $O(n^3)$ operation.
- The matrix columns are ordered to prioritize sparse columns, but it's still fundamentally GE.

**CADO-NFS State:**
- Uses the Block Wiedemann algorithm, an iterative $O(n^2)$ method ideal for large, sparse matrices over GF(2).
- Highly parallelized and distributed.

**Inefficiency:** High for $N > 100$ bits. GE is acceptable for very small matrices but scales terribly. For the target 96-128 bit range, the matrix is small enough that GE might work, but Block Wiedemann would still be significantly faster.

---

## Improvement Plan

The improvements are prioritized based on a "High Reward / Low Complexity" to "Low Reward / High Complexity" scale.

### Phase 1: High Reward / Low-to-Medium Complexity

1.  **Sieve: Eliminate `f64` in Norm Initialization**
    *   **Action:** Rewrite `init_norm_rat` and `init_norm_alg` in `rust-nfs/src/sieve/norm.rs` to use integer arithmetic or fixed-point approximations instead of `f64`.
    *   **Why:** Floating-point math in the innermost loop is slow. Replacing it with integer math/shifts will provide an immediate, noticeable speedup to the sieve stage.
2.  **Sieve: Optimize Bucket Application Loop**
    *   **Action:** Investigate vectorizing or unrolling the `apply_bucket_updates` loop in `rust-nfs/src/sieve/region.rs`. Ensure `saturating_sub` compiles to efficient branchless instructions (e.g., x86 `psubusb`).
    *   **Why:** This loop executes billions of times. Small micro-optimizations here yield high returns.

### Phase 2: High Reward / High Complexity

3.  **Polyselect: Implement Murphy Alpha and Root Optimization**
    *   **Action:** Replace the naive base-m polynomial selection in `gnfs/src/polyselect.rs` with an algorithm that generates polynomials with good root properties (measured by Murphy $\alpha$) and scores them using Murphy E.
    *   **Why:** A better polynomial drastically reduces the sieving time by making relations easier to find. This is complex to implement correctly but essential for beating CADO-NFS.
4.  **Filtering: Implement $n$-way Merging (Cavallar's Algorithm)**
    *   **Action:** Upgrade `partial_merge.rs` from a simple 2LP Union-Find to a full clique-finding algorithm capable of merging relations with 3 or more large primes, aiming to minimize the weight of the resulting matrix rows.
    *   **Why:** Reduces matrix size and density, speeding up Linear Algebra.

### Phase 3: Medium Reward / High Complexity

5.  **Linear Algebra: Implement Block Wiedemann**
    *   **Action:** Replace Gaussian Elimination in `gnfs/src/linalg.rs` with the Block Wiedemann algorithm for finding nullspace vectors of sparse GF(2) matrices.
    *   **Why:** GE is $O(n^3)$ while Wiedemann is $O(n^2)$. As bit sizes increase towards 128, the matrix size makes GE a major bottleneck.

### Phase 4: Pre-Commit and Finalization

6.  **Pre-Commit Checks**
    *   **Action:** Execute pre-commit instructions to ensure proper testing, verification, review, and reflection are done before finalizing changes. This involves checking formatting, running tests, and verifying the build.
    *   **Why:** Ensures the codebase remains stable and maintainable.
7.  **Submit Changes**
    *   **Action:** Commit the analysis document and any implemented code changes with a descriptive branch name and commit message.


## Beyond CADO-NFS: Next-Generation Optimizations

If the goal is not just to match CADO-NFS but to significantly outpace it—reducing both the constant factor and asymptotic execution time—`rust-nfs` must adopt architectures and mathematical breakthroughs that CADO-NFS (designed over a decade ago for x86 CPU clusters) lacks.

### 1. Zero-Copy In-Memory Pipeline (Architectural)
**CADO-NFS Limitation:** CADO-NFS uses a multi-process, client-server architecture written in Python that orchestrates C binaries. It serializes hundreds of gigabytes (or terabytes) of relations to disk as plain text between the Sieving, Filtering, and Linear Algebra phases.
**The "Beyond" Solution:** Implement a zero-copy, fully in-memory pipeline. Rust's concurrency model allows spawning a sieve thread pool that directly feeds relations into a concurrent lock-free hash map or channel for filtering. This eliminates massive disk I/O bottlenecks and parsing overhead, drastically reducing the constant time factor.

### 2. Apple Silicon / Modern ARM Cache Tuning (Hardware)
**CADO-NFS Limitation:** CADO's bucket sieve was painstakingly tuned for older x86 processors with 32KB to 64KB L1 caches (`LOG_BUCKET_REGION = 16`).
**The "Beyond" Solution:** Modern processors, particularly Apple Silicon (M-series), possess 128KB to 192KB L1 data caches and massive unified memory bandwidth. Expanding the bucket region size (`LOG_BUCKET_REGION = 17` or `18`) allows the inner loop to process exponentially larger continuous blocks of the sieve array without stalling on L2/L3 cache misses.

### 3. GPU-Native Sieving and Cofactorization (Hardware)
**CADO-NFS Limitation:** While CADO has experimental GPU support for Linear Algebra, its core sieve and cofactorization (ECM/P-1/P+1) are strictly CPU-bound.
**The "Beyond" Solution:**
*   **GPU Cofactorization:** The Elliptic Curve Method (ECM) used to factor the remaining cofactors after sieving is perfectly suited for SIMT (Single Instruction, Multiple Thread) execution on GPUs. Batching tens of thousands of survivors and running GPU-ECM can reduce cofactorization time to near zero.
*   **GPU Sieving:** Implementing the bucket sieve directly in CUDA/Metal, taking advantage of massive memory bandwidth.

### 4. Pilatte's Smooth Relations via LLL (Algorithmic Breakthrough)
**CADO-NFS Limitation:** CADO relies entirely on brute-force sieving to find smooth relations (the Canfield-Erdős-Pomerance theorem).
**The "Beyond" Solution:** As detailed in the `smooth-pilatte` crate concept (based on Pilatte's 2024 proof), we can exploit L-function zero-density estimates. Instead of sieving an array, we can construct a short product lattice of small primes and use LLL/BKZ reduction to directly *sample* smooth relations. This bypasses the traditional sieve entirely and could fundamentally alter the asymptotic complexity of relation generation.

### 5. Machine Learning Guided Polynomial Selection and Filtering (Algorithmic)
**CADO-NFS Limitation:** Uses hardcoded heuristics for Murphy E/Alpha and combinatorial clique-finding for filtering.
**The "Beyond" Solution:**
*   **Filtering:** Train a lightweight neural network or decision tree to predict which large prime (LP) relations are most likely to form useful cycles in the merge phase. Drop low-probability relations early to save memory and processing time.
*   **Polyselect:** Use gradient descent or RL to navigate the polynomial coefficient space rather than exhaustive searching.
