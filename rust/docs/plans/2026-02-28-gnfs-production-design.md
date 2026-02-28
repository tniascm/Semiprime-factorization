# Production GNFS Implementation — Design Document

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a production-grade General Number Field Sieve in Rust that matches or beats CADO-NFS on 130+ digit numbers, exploiting M4 unified memory, GPU cofactorization, large L1 cache, and Neural Engine for ML-guided decisions.

**Architecture:** Clean-sheet Rust crate (`gnfs`) with no dependencies on existing workspace crates. Five-stage pipeline (polyselect → sieve → filter → linalg → sqrt) with checkpoint/restart at every stage boundary. Performance-first design: NEON intrinsics in sieve inner loop, Metal compute shaders for ECM cofactorization and Block Wiedemann SpMV, CoreML for ML-guided polynomial screening and cofactor prediction.

**Tech Stack:** Rust (stable, aarch64-apple-darwin), `rug` (GMP bindings for BigInt — same assembly backend as CADO-NFS), `rayon`, `clap`, `serde`, `metal-rs`, `core-ml` (via objc2 bindings), NEON intrinsics via `std::arch::aarch64`.

---

## 1. Structural Advantages Over CADO-NFS

CADO-NFS is a 200K LOC C codebase designed for discrete CPU + RAM on x86 clusters. Five structural advantages make a faster implementation possible on M4:

### 1.1 Unified Memory — Zero-Copy CPU/GPU Data Sharing

M4's CPU and GPU share the same physical memory. CADO-NFS does all cofactorization on CPU because its target hardware (x86 + discrete GPU) requires PCIe transfers. Our architecture runs ECM cofactorization on the GPU with zero-copy: sieve candidates produced by CPU are immediately visible to GPU compute shaders without any data transfer. GPU results are immediately visible to CPU relation-collection threads. This eliminates cofactorization as a CPU bottleneck.

### 1.2 M4 P-Core L1 Cache: 192KB vs x86 32-64KB

CADO-NFS tunes its bucket sieve for 32-64KB L1 cache (Intel/AMD). M4 P-cores have 192KB L1 data cache — 3-6x larger. Larger buckets mean each factor base prime hits fewer buckets, producing fewer (bucket_id, offset, log_p) triples in the scatter phase and fewer cache misses in the gather phase. Expected: 25-30% reduction in memory traffic for the medium-prime sieve tier, which dominates sieve time.

### 1.3 GPU ECM Cofactorization

Cofactorization (testing sieve survivors for smoothness via P-1, P+1, ECM) consumes 30-50% of CADO-NFS sieve time on CPU. Our architecture offloads ECM to the M4 GPU (10 cores, ~1280 parallel ALUs). A dedicated GPU dispatch thread batches candidates from a lock-free ring buffer and launches Metal compute shaders processing ~4096 ECM curves simultaneously. The CPU never stalls on cofactorization. Expected: cofactorization drops from 30-50% to <10% of sieve wall-clock time.

### 1.4 Neural Engine for ML-Guided Decisions

M4 Neural Engine: 16 TOPS at INT8. Not for arithmetic — for prediction:

- **Polynomial screening**: Train small MLP to predict Murphy's E-value from polynomial coefficients. Screen 10M+ candidates/second on Neural Engine. Only top 0.1% proceed to expensive CPU root optimization. 100x fewer root optimization calls.
- **Cofactor smoothness prediction**: Predict P(smooth | cofactor_size, partial_factorization) before committing ECM cycles. Skip low-probability candidates early.
- **Sieve yield prediction**: Predict relations/special-q for upcoming ranges. Prioritize high-yield special-q values.

### 1.5 Rust Zero-Cost Parallelism

- Fat LTO + monomorphization inlines across all module boundaries. CADO-NFS's separate compilation units prevent cross-module inlining.
- Rayon work-stealing for embarrassingly parallel stages (polynomial search, sieving, filtering).
- Ownership model enables lock-free concurrent data structures without data races.
- `unsafe` blocks scoped to NEON intrinsics in the sieve hot path — everything else is memory-safe.

---

## 2. Crate Structure

```
gnfs/
  Cargo.toml
  build.rs              # Metal shader compilation, CoreML model bundling
  src/
    main.rs             # CLI entry, pipeline orchestrator
    lib.rs              # Public API
    types.rs            # Core types: Relation, FactorBase, Polynomial, SpecialQ
    arith.rs            # Montgomery multiplication, batch modular ops, NEON helpers
    polyselect.rs       # Kleinjung polynomial selection + root optimization
    polyselect_ml.rs    # CoreML polynomial screening model
    sieve.rs            # Special-q lattice sieving + bucket sort
    sieve_neon.rs       # NEON intrinsics for sieve inner loop
    cofactor.rs         # ECM, P-1, P+1 cofactorization (CPU path)
    cofactor_gpu.rs     # Metal compute ECM pipeline
    filter.rs           # Singleton/clique removal, parallel SGE
    linalg.rs           # Block Wiedemann over GF(2)
    linalg_gpu.rs       # Metal SpMV kernel
    sqrt.rs             # Algebraic + rational square root, factor extraction
    pipeline.rs         # Stage orchestrator with checkpoints
    params.rs           # Parameter tables (c60 through c200)
    log.rs              # StageLogger: timestamped structured logging
    bench.rs            # Head-to-head benchmark harness vs CADO-NFS
  shaders/
    ecm_stage1.metal    # ECM Stage 1 kernel
    ecm_stage2.metal    # ECM Stage 2 kernel
    spmv_gf2.metal      # Sparse matrix-vector multiply over GF(2)
  models/
    poly_screen.mlmodel # Polynomial E-value predictor (trained offline)
    cofactor_pred.mlmodel # Cofactor smoothness predictor
  tests/
    integration.rs
    correctness.rs      # Cross-validate against known factorizations
    bench_comparison.rs # Automated CADO-NFS comparison
```

### External Dependencies

| Crate | Purpose | Why |
|-------|---------|-----|
| `rug` 1.x | GMP-backed BigInt | Same assembly as CADO-NFS uses. Non-negotiable for parity. |
| `rayon` 1.x | Work-stealing parallelism | Sieving, filtering, polynomial search |
| `clap` 4.x | CLI | Stage selection, parameter override, resume |
| `serde` / `serde_json` | Checkpoint serialization | Stage boundary persistence |
| `rand` / `rand_chacha` | Reproducible RNG | Deterministic test suites |
| `metal` 0.x | Metal compute shaders | GPU ECM, GPU SpMV |
| `objc2` / `core-ml` bindings | Neural Engine access | ML-guided polynomial/cofactor screening |
| `crossbeam` | Lock-free ring buffers | CPU→GPU cofactor pipeline |
| `memmap2` | Memory-mapped relation files | Efficient large-file I/O for filtering |

### Build Configuration

```toml
[profile.release]
lto = "fat"
codegen-units = 1
opt-level = 3
panic = "abort"          # No unwinding overhead in hot loops

[target.aarch64-apple-darwin]
rustflags = ["-C", "target-cpu=native"]   # NEON, LSE atomics, ARMv8.4-A
```

---

## 3. Core Types (`types.rs`)

```rust
/// A polynomial pair (f, g) where f(m) ≡ 0 mod N and g is linear.
pub struct PolynomialPair {
    pub f: Vec<rug::Integer>,     // Algebraic polynomial coefficients [a0, a1, ..., ad]
    pub g: [rug::Integer; 2],     // Rational polynomial [g0, g1] where g(x) = g0 + g1*x
    pub m: rug::Integer,          // Common root: f(m) ≡ g(m) ≡ 0 mod N
    pub skewness: f64,            // Optimal skewness for sieve region
    pub murphy_e: f64,            // Murphy's E-value (quality metric)
}

/// A smooth relation: (a, b) with complete factorization on both sides.
pub struct Relation {
    pub a: i64,
    pub b: u64,
    pub rational_factors: Vec<(u32, u8)>,    // (prime, exponent) on rational side
    pub algebraic_factors: Vec<(u32, u8)>,   // (prime, exponent) on algebraic side
    pub rational_large_primes: Vec<u64>,     // Large primes on rational side
    pub algebraic_large_primes: Vec<u64>,    // Large primes on algebraic side
}

/// Factor base entry with precomputed data for sieving.
pub struct FBEntry {
    pub p: u32,                    // Prime
    pub log_p: u8,                 // floor(log2(p) * scale_factor)
    pub roots: Vec<u32>,           // Roots of f(x) mod p
    pub reciprocal: u64,           // Precomputed floor(2^64 / p) for division-free modular reduction
    pub projective: bool,          // p divides leading coefficient
}

/// Special-q descriptor for lattice sieving.
pub struct SpecialQ {
    pub q: u64,                    // Special-q prime
    pub r: u64,                    // Root: f(r) ≡ 0 mod q
    pub basis: [[i64; 2]; 2],     // Reduced lattice basis for the q-lattice
}

/// Full GNFS parameter set for a given digit size.
pub struct GnfsParams {
    pub degree: u32,               // Polynomial degree (4, 5, or 6)
    pub lim0: u64,                 // Rational factor base bound
    pub lim1: u64,                 // Algebraic factor base bound
    pub lpb0: u32,                 // Large prime bits, rational side
    pub lpb1: u32,                 // Large prime bits, algebraic side
    pub mfb0: u32,                 // Max cofactor bits, rational side
    pub mfb1: u32,                 // Max cofactor bits, algebraic side
    pub sieve_i: u32,              // Sieve region width = 2^I
    pub qmin: u64,                 // Minimum special-q
    pub qmax: u64,                 // Maximum special-q
    pub rels_wanted: u64,          // Target relation count
    pub ncurves0: u32,             // ECM curves per side (rational)
    pub ncurves1: u32,             // ECM curves per side (algebraic)
    pub bw_block: u32,             // Block Wiedemann blocking factor (64)
    pub bw_interval: u32,          // Checkpoint interval for Krylov
    pub nchar: u32,                // Schirokauer map characters
}
```

---

## 4. Stage Designs

### 4.1 Polynomial Selection (`polyselect.rs` + `polyselect_ml.rs`)

**Algorithm**: Kleinjung (2008) ad-search + Bai-Bouvier-Kruppa root optimization.

**Phase 1 — Ad-search with Neural Engine pre-screening:**

1. Enumerate leading coefficients ad in [admin, admax] with step `incr` (typically 60 or 210, ensuring many small prime factors).
2. For each ad, compute m = floor(N^(1/(d+1))), construct base-m polynomial.
3. Extract feature vector (ad, m, skewness_estimate, first 3 coefficients).
4. **Neural Engine batch**: Submit batch of ~10K feature vectors to CoreML poly_screen model. Model returns predicted Murphy's E-value in <1ms for the batch.
5. Only candidates with predicted E-value in top 0.1% proceed to Phase 2.
6. Parallelism: ad ranges split across Rayon threads. Neural Engine calls batched from a single coordination thread.

**Phase 2 — Root optimization (CPU):**

1. For each surviving candidate, apply root sieve: perturb polynomial coefficients to optimize alpha-value (expected divisibility by small primes 2,3,5,...,200).
2. Score final polynomial with full Murphy's E-value computation.
3. Keep top `nrkeep` polynomials (typically 10-50).
4. Select the best for sieving.

**Output**: `poly.json` — serialized PolynomialPair with full metadata.

**Logging**: Every candidate scored, Phase 1 throughput (candidates/sec), Phase 2 optimization trajectory, final Murphy's E-value.

### 4.2 Sieving (`sieve.rs` + `sieve_neon.rs` + `cofactor.rs` + `cofactor_gpu.rs`)

**Algorithm**: Special-q lattice sieving with tiered bucket sort.

**Architecture — three concurrent thread groups:**

```
[Sieve threads: 4 P-cores]  →  lock-free ring buffer  →  [GPU dispatch: 1 thread]
                                                              ↓
                                                        [Metal GPU: ECM batches]
                                                              ↓
                                                        [Relation writer: 1 thread]
```

**Sieve thread (one per P-core):**

1. Claim next special-q from atomic work queue.
2. Compute reduced lattice basis via extended GCD.
3. Initialize sieve array (2^I × 2^(I-1) bytes, fits in L2 cache for I ≤ 14).
4. **Bucket scatter phase**: For each factor base prime p:
   - **Small primes** (p < 192KB = L1 size): Direct sieve. Step through array by p, subtract log_p using NEON `vqsubq_u8` (16 entries/instruction). This is the hottest loop — `unsafe` + NEON intrinsics.
   - **Medium primes** (192KB < p < 2^I): Franke-Kleinjung 16-row block sieve. Process 16 adjacent j-lines per pass, fitting working set in L1.
   - **Large primes** (p > 2^I): Bucket sort. Store (bucket_id, offset_within_bucket, log_p) into per-bucket vectors. No sieve array access in this phase.
5. **Bucket gather phase**: For each L1-sized bucket, load all pending large-prime updates, apply them to the cache-resident sieve array.
6. **Survivor identification**: Scan sieve array for entries below threshold (norm < expected_smooth_bound). For each survivor (i, j), recover (a, b), compute exact norms, trial-divide by small primes.
7. **Cofactor dispatch**: For survivors with remaining cofactor, push (a, b, cofactor_rat, cofactor_alg) into the lock-free ring buffer.

**GPU dispatch thread:**

1. Drain ring buffer into batch (target batch size: 4096 candidates).
2. Allocate Metal buffer in unified memory (zero copy — just pass pointer).
3. Dispatch ECM Stage 1 compute shader: each GPU thread runs one Edwards curve on one candidate. B1 chain: 100 → 600 → 3500.
4. Read results (zero copy). For successful factorizations, construct full Relation and push to writer.
5. For remaining unfactored cofactors within mfb bound, dispatch ECM Stage 2.
6. Candidates exceeding mfb after all attempts: discard.

**Relation writer thread:**

1. Drain completed relations from a second ring buffer.
2. Compress and write to relation file (one per special-q range).
3. Update global relation counter (atomic).
4. Log: relations/second, yield per special-q, cofactorization hit rate.

**NEON sieve inner loop** (`sieve_neon.rs`):

```rust
use std::arch::aarch64::*;

/// Sieve a small prime p across the sieve array.
/// Processes 16 bytes per iteration using NEON saturating subtract.
#[inline(always)]
pub unsafe fn sieve_small_prime(
    sieve: &mut [u8],
    start: usize,
    stride: usize,
    log_p: u8,
    len: usize,
) {
    let log_vec = vdupq_n_u8(log_p);
    let mut pos = start;
    while pos + 16 <= len {
        let ptr = sieve.as_mut_ptr().add(pos);
        let current = vld1q_u8(ptr);
        let updated = vqsubq_u8(current, log_vec);
        vst1q_u8(ptr, updated);
        pos += stride;
    }
    // Scalar cleanup for tail
    while pos < len {
        sieve[pos] = sieve[pos].saturating_sub(log_p);
        pos += stride;
    }
}
```

**Metal ECM kernel** (`shaders/ecm_stage1.metal`):

Each GPU thread processes one (candidate, curve_seed) pair. Montgomery multiplication in 256-bit using uint4 (128-bit) pairs. The kernel performs ECM Stage 1: iterate over primes up to B1, compute scalar multiplication on an Edwards curve modulo the cofactor. If GCD with cofactor yields a factor, write it to the output buffer.

**Output**: Compressed relation files in `rels/` directory, one per special-q range.

**Logging**: Per special-q: yield, norms, cofactor hit rate. Per batch: GPU dispatch latency, ECM throughput. Global: total relations, relations/second, estimated time to target.

### 4.3 Filtering (`filter.rs`)

**Algorithm**: Duplicate removal → singleton purge → clique removal → parallel SGE.

**Phase 1 — Duplicate removal:**
- Memory-map all relation files.
- Hash each (a, b) pair (normalized: gcd(a,b)=1, b>0) into a concurrent hash set.
- Deduplicate. Expected ~25% duplicates.

**Phase 2 — Singleton/clique purge:**
- Build ideal-to-relation incidence map (concurrent HashMap<ideal_index, Vec<relation_index>>).
- Iteratively remove singletons (ideals appearing in exactly one relation) and their containing relations.
- Remove weight-2 cliques: merge two relations sharing a unique ideal into one combined relation.
- Repeat until convergence. Typically removes 60-70% of relations.
- Rayon parallel iteration over ideals for each pass.

**Phase 3 — Structured Gaussian Elimination (merge):**
- For ideals of increasing weight (3, 4, ..., up to 25):
  - Select pivot relation (Markowitz criterion: minimize fill-in).
  - XOR pivot row into all other rows containing this ideal.
  - Track fill-in. Abort merge for this ideal if fill-in exceeds threshold.
- Parallel merge: independent ideals (no shared relations) processed concurrently via Rayon.
- Target: reduce matrix to ~1/5 of post-purge size, average row weight ~110.

**Output**: Sparse binary matrix in CSR format (`matrix.bin`), column-to-ideal mapping.

**Logging**: Per phase: input/output sizes, time. Merge phase: fill-in trajectory, pivot quality.

### 4.4 Linear Algebra (`linalg.rs` + `linalg_gpu.rs`)

**Algorithm**: Block Wiedemann over GF(2), blocking factor n = m = 64.

**Three computational phases:**

**Krylov step** (90% of time):
- Compute sequence x^T * M^i * y for i = 0, 1, ..., 2*ceil(N_rows/64).
- Each iteration: one sparse matrix-vector multiply (SpMV) of M × v where v is a dense 64-column block vector.
- **GPU path**: Metal compute shader for SpMV. Each GPU thread processes one matrix row: load row's column indices from CSR, XOR corresponding 64-bit entries from source vector, write to destination. At c130 scale (~5M rows), the GPU launches 5M threads across its 10 cores.
- **Unified memory**: Matrix and vectors live in shared memory. No CPU↔GPU transfer.
- Checkpoint every `bw_interval` iterations: serialize Krylov vectors to disk.

**Lingen step** (Berlekamp-Massey):
- Thome's sub-quadratic algorithm on the 64×64 matrix sequence.
- CPU-only. O(N * log(N)^2) — fast, not a bottleneck.

**Mksol step:**
- Reconstruct null-space vectors from minimal polynomial.
- Similar SpMV cost to Krylov but fewer iterations (~N/64).
- GPU SpMV reused.

**Metal SpMV kernel** (`shaders/spmv_gf2.metal`):

```metal
kernel void spmv_gf2(
    device const uint* row_ptr    [[buffer(0)]],
    device const uint* col_idx    [[buffer(1)]],
    device const ulong* src       [[buffer(2)]],  // 64-bit packed source vector
    device ulong* dst             [[buffer(3)]],   // 64-bit packed destination vector
    uint tid                      [[thread_position_in_grid]]
) {
    ulong acc = 0;
    uint start = row_ptr[tid];
    uint end = row_ptr[tid + 1];
    for (uint k = start; k < end; k++) {
        acc ^= src[col_idx[k]];
    }
    dst[tid] = acc;
}
```

**Output**: Dependency vectors (`deps.bin`) — each is a bitmask over relations.

**Logging**: Iteration count, iterations/second, checkpoint events, estimated time remaining.

### 4.5 Square Root (`sqrt.rs`)

**Algorithm**: Given a dependency (subset S of relations whose product squares on both sides):

1. **Rational square root**: Compute product of all G(a,b) for (a,b) in S using `rug::Integer`. Take integer square root. Verify square. Straightforward.

2. **Algebraic square root**: Compute product of (a - b*alpha) in the number field Q(alpha) = Q[x]/f(x). Represent elements as polynomials in alpha with `rug::Integer` coefficients, reduce modulo f(x) after each multiplication. Take square root via Hensel lifting:
   - Start from square root modulo a small prime p.
   - Lift to modulo p^2, p^4, ..., p^(2^k) until precision exceeds N.
   - Schirokauer map characters (from linalg) ensure the algebraic square root exists.

3. **Factor extraction**: Map rational_sqrt and algebraic_sqrt to Z/NZ via the homomorphism alpha → m. Compute gcd(rational_sqrt - algebraic_sqrt, N). If non-trivial: done. If trivial: try gcd(rational_sqrt + algebraic_sqrt, N). If still trivial: try next dependency.

**Output**: `factors.json` with the two prime factors and verification (p * q = N).

**Logging**: Each dependency attempted, Hensel lift progress, GCD results.

---

## 5. Pipeline Orchestration (`pipeline.rs`)

### CLI

```
gnfs factor <N>                           # Factor N using auto-detected params
gnfs factor <N> --params c130             # Use c130 parameter set
gnfs factor --input number.txt            # Read N from file
gnfs factor <N> --stage sieve             # Run only sieve stage (requires poly.json)
gnfs resume <run_dir>                     # Resume from last checkpoint
gnfs resume <run_dir> --stage linalg      # Resume specific stage
gnfs bench --range 60-130 --step 10       # Head-to-head benchmark vs CADO-NFS
gnfs bench --target c100 --compare cado   # Single comparison at c100
```

**Flags:**
- `--seed <u64>`: Deterministic RNG seed (default: 42)
- `--threads <N>`: CPU thread count (default: 4 for P-cores)
- `--gpu / --no-gpu`: Enable/disable Metal GPU (default: enabled)
- `--ml / --no-ml`: Enable/disable Neural Engine ML (default: enabled)
- `--output-dir <path>`: Run output directory
- `--verbose`: Per-special-q sieve logging
- `--checkpoint-interval <secs>`: How often to write checkpoints

### Run Directory Structure

```
runs/
  c130_seed42_20260228_143000/
    run_config.json                      # Full parameter dump + hardware info
    stage_polyselect/
      candidates.jsonl                   # All scored polynomial candidates
      poly.json                          # Selected polynomial pair
      log.jsonl                          # Timestamped event log
    stage_sieve/
      rels/
        rels_q0750000_q0760000.gz        # Compressed relation files
        rels_q0760000_q0770000.gz
        ...
      checkpoint.json                    # Last completed special-q, relation count
      log.jsonl
    stage_filter/
      matrix.bin                         # Sparse CSR matrix
      col_map.bin                        # Column-to-ideal mapping
      stats.json                         # Filtering statistics
      log.jsonl
    stage_linalg/
      deps.bin                           # Dependency vectors
      checkpoint_iter_3000.bin           # Krylov checkpoint
      log.jsonl
    stage_sqrt/
      factors.json                       # Final result
      log.jsonl
    summary.json                         # End-to-end timing and metrics
```

### Stage Transitions

Each stage reads its predecessor's output and validates it before proceeding:

```
polyselect → poly.json
    ↓ validate: degree matches params, f(m) ≡ 0 mod N, Murphy's E > threshold
sieve → rels/*.gz
    ↓ validate: relation count ≥ rels_wanted, sample relations verify
filter → matrix.bin
    ↓ validate: matrix has excess (rows > cols), density in expected range
linalg → deps.bin
    ↓ validate: at least one dependency, M * dep = 0 over GF(2)
sqrt → factors.json
    ↓ validate: p * q = N, p > 1, q > 1
```

If any stage fails validation, it logs the failure with full diagnostic context and stops. No silent data corruption propagates to the next stage.

---

## 6. Structured Logging (`log.rs`)

Every operation gets a timestamped structured log entry:

```rust
pub struct StageLogger {
    stage: String,
    start: Instant,
    log_file: BufWriter<File>,
}

impl StageLogger {
    /// Log an event with automatic UTC timestamp and elapsed time.
    pub fn event(&mut self, msg: &str, metrics: &serde_json::Value) { ... }

    /// Log stage start with full parameter dump.
    pub fn start(&mut self, params: &serde_json::Value) { ... }

    /// Log stage completion with summary metrics.
    pub fn finish(&mut self, summary: &serde_json::Value) { ... }

    /// Write checkpoint and log it.
    pub fn checkpoint(&mut self, state: &impl Serialize) { ... }
}
```

**Console output format:**
```
[14:30:05 UTC] [polyselect] START — degree=5, admin=1260, admax=38000
[14:30:05 UTC] [polyselect] Phase 1: screening 175K candidates via Neural Engine
[14:30:06 UTC] [polyselect] Phase 1: 175K screened in 0.8s (219K/s), 175 survivors
[14:30:12 UTC] [polyselect] Phase 2: root-optimized 175 candidates in 6.2s
[14:30:12 UTC] [polyselect] DONE — best E=2.34e-14, skew=48291.3 — 7.0s
[14:30:12 UTC] [sieve] START — lim0=6M, lim1=7.5M, lpb0=28, lpb1=29, I=13
[14:30:12 UTC] [sieve] Factor base: 412K rational + 501K algebraic primes
[14:30:12 UTC] [sieve] GPU ECM: enabled (Metal, batch=4096, B1=[100,600,3500])
[14:32:12 UTC] [sieve] progress: 15K/200K sq | 1.2M/30M rels | 80 rels/sq | 10K rels/s | ETA 47min
```

**JSON log format** (one line per event in `log.jsonl`):
```json
{"ts":"2026-02-28T14:32:12Z","elapsed_s":120.3,"stage":"sieve","event":"progress","sq_done":15000,"sq_total":200000,"rels":1200000,"rels_target":30000000,"yield_per_sq":80.0,"rels_per_sec":10000,"eta_sec":2820}
```

---

## 7. Parameter Tables (`params.rs`)

Matching CADO-NFS parameter files for direct comparison:

| Param | c60 | c80 | c100 | c120 | c130 | c140 |
|-------|-----|-----|------|------|------|------|
| degree | 4 | 5 | 5 | 5 | 5 | 5 |
| lim0 | 50K | 200K | 1M | 4M | 6M | 10M |
| lim1 | 50K | 200K | 1.5M | 5M | 7.5M | 13M |
| lpb0 | 20 | 23 | 26 | 27 | 28 | 29 |
| lpb1 | 20 | 23 | 26 | 28 | 29 | 30 |
| mfb0 | 40 | 46 | 52 | 54 | 54 | 58 |
| mfb1 | 40 | 46 | 52 | 56 | 57 | 60 |
| I | 11 | 12 | 13 | 13 | 13 | 14 |
| rels_wanted | 50K | 500K | 5M | 20M | 30M | 60M |
| qmin | 5K | 50K | 200K | 500K | 750K | 1.5M |
| ncurves | 3 | 6 | 10 | 12 | 13 | 15 |

These are derived from CADO-NFS's published params.c{N} files and can be overridden via CLI.

---

## 8. Milestones

### M1: 60-digit (Foundation)

Build the complete 5-stage pipeline with correct but unoptimized implementations:
- Base-m polynomial selection (not Kleinjung yet)
- Line sieving (not lattice sieving yet)
- Trial division cofactorization (not ECM yet)
- Dense GF(2) Gaussian elimination (not Block Wiedemann yet)
- Rational + algebraic square root
- Full logging and checkpoint infrastructure
- CLI with stage selection and resume

**Validation**: Factor 10 known 60-digit semiprimes. Cross-check factors are correct. Benchmark vs CADO-NFS on same inputs — record baseline gap.

### M2: 80-digit (Sieve Upgrade)

Replace the sieve with production-grade implementation:
- Special-q lattice sieving with bucket sort
- NEON intrinsics for small-prime inner loop
- L1-cache-tuned bucket size (192KB)
- ECM/P-1/P+1 cofactorization on CPU
- Large prime variation (2 large primes per side)

**Validation**: Factor known 80-digit semiprimes. Measure sieve yield per special-q — must match CADO-NFS within 5% on identical parameters. Sieve throughput (rels/core-sec) must match or exceed CADO-NFS.

### M3: 100-digit (Linalg + Filtering Upgrade)

Replace linear algebra and add filtering:
- Block Wiedemann over GF(2) with GPU SpMV
- Singleton/clique purge + parallel SGE
- Memory-mapped relation I/O for filtering

**Validation**: Factor RSA-100 (known factors). Benchmark Block Wiedemann iterations/second vs CADO-NFS. Total core-hours must be within 1.5x of CADO-NFS.

### M4: 120-digit (Polynomial Selection + GPU ECM)

Add the final performance pieces:
- Kleinjung polynomial selection with Neural Engine screening
- Metal GPU ECM cofactorization pipeline
- Full parameter tuning for c120

**Validation**: Factor known 120-digit semiprimes. Must match CADO-NFS total time. GPU ECM throughput measured independently.

### M5: 130+ digit (Production Tuning)

Optimization and hardening:
- Profile-guided optimization of sieve inner loop
- Metal shader tuning for M4 GPU occupancy
- Neural Engine cofactor prediction model
- Full parameter sweep for c130, c140
- Distributed work queue (optional, for multi-machine)

**Validation**: Factor numbers using CADO-NFS's published c130 parameters. Total time must be ≤ CADO-NFS. Produce detailed per-stage timing comparison.

---

## 9. Performance Verification Protocol

### Automated Benchmark Suite

```
gnfs bench --target c80 --compare cado-nfs --trials 10
```

Runs both implementations on 10 deterministically-generated semiprimes per digit size. For each trial, compares:

| Metric | Requirement |
|--------|-------------|
| Polynomial Murphy's E-value | ≥ CADO-NFS best |
| Sieve: relations/core-second | ≥ CADO-NFS |
| Sieve: yield per special-q | within 5% of CADO-NFS |
| Filter: matrix reduction ratio | within 10% of CADO-NFS |
| Linalg: iterations/second | ≥ CADO-NFS |
| Total: core-seconds to factor | ≤ CADO-NFS |

Results written to `bench_results.json` with full timing breakdown per stage. The benchmark fails (exit code 1) if any metric regresses below parity, preventing performance regressions from being committed.

### Hardware Counter Profiling

For the sieve inner loop, use `perf` (or Instruments on macOS) to verify:
- L1 cache miss rate < 1% during bucket gather phase
- NEON utilization > 80% during small-prime sieve
- No false sharing on lock-free ring buffer cache lines

### Correctness Verification

Every factorization verified by: p * q = N, p > 1, q > 1, p and q prime (Miller-Rabin with 25 rounds). Cross-checked against CADO-NFS output on the same input when available.

---

## 10. Data Volumes and Resource Estimates at c130

| Resource | Estimate |
|----------|----------|
| Factor base (both sides) | ~900K primes, ~20MB |
| Raw relations (with duplicates) | ~40M, ~4-8 GB compressed |
| Post-filter matrix | ~1-5M rows, ~500MB |
| Block Wiedemann vectors | ~100MB |
| Peak memory (filtering) | ~8-12 GB |
| Peak memory (sieving, 4 threads) | ~4 GB |
| Disk total | ~10-20 GB |
| Estimated wall-clock (M4, 4 P-cores + GPU) | 12-48 hours |
| Estimated core-hours | 50-200 |

All estimates based on CADO-NFS published parameters for c130 and known relation counts from RSA factorization records.
