# Rust-NFS vs CADO-NFS: Parity Achieved (c30)

## Results (post-optimization, 2026-03-12)

| Metric | Rust-NFS | CADO-NFS | Ratio |
|--------|----------|----------|-------|
| Single-threaded median | **~940ms** | 2149ms | **0.44x** (2.3x faster) |
| Multi-threaded mean | **~350ms** | ~5780ms | **0.06x** (16.5x faster) |

Benchmark: 50 random 100-bit semiprimes (5 seeds x 10 runs), Apple M-series.

### Previous milestone (2026-03-10)

| Metric | Rust-NFS | CADO-NFS | Ratio |
|--------|----------|----------|-------|
| Single-threaded CPU | 2071ms | 2149ms | 0.96x (3.6% faster) |
| Multi-threaded wall | 773ms | ~5780ms | 0.13x (7.5x faster) |

## CADO-NFS Reference (single-threaded c30)

CADO-NFS single-threaded timings for the same semiprime:

| Stage | CPU time | Wall time |
|-------|----------|-----------|
| Sieve | ~390ms | — |
| LA (Block Wiedemann) | ~200ms | — |
| Filter | ~50ms | — |
| Sqrt | ~30ms | — |
| **Algorithm total** | **~2149ms** | — |
| Python orchestration + IPC | — | ~5630ms |
| **Total wall clock** | — | **~7780ms** |

CADO's wall clock is ~3.6x its CPU time due to Python orchestration overhead
(process spawning, parameter file I/O, inter-stage serialization). The CPU
time of 2149ms reflects pure C algorithm performance. CADO's sieve is much
faster (390ms vs 1566ms) thanks to Kleinjung polynomial selection (Y1 > 1)
producing ~308 rels/SQ vs our base-m ~12 rels/SQ, but their LA and filter
are comparable.

## Optimization History

Full journey from initial implementation to CADO parity:

| Milestone | Total (1T) | vs CADO | Key change |
|-----------|-----------|---------|------------|
| Initial baseline | ~6917ms | 3.2x slower | Naive sieve, dense GE, no pre-elim |
| + FK lattice walk | ~4800ms | 2.2x slower | Partial-GCD scatter replaces row-by-row |
| + SIMD survivor scan | ~4600ms | 2.1x slower | NEON 16-byte vectorized threshold scan |
| + Pre-elimination (w1-w2) | ~3800ms | 1.8x slower | Singleton + weight-2 merges before GE |
| + 2LP + sieve_mfb tuning | ~2800ms | 1.3x slower | Two large primes, tight sieve threshold |
| + Two-phase polyselect | ~2500ms | 1.2x slower | Screen by lognorm+alpha, then Murphy E |
| + Queue-based filter | ~2400ms | 1.1x slower | O(n) singleton removal |
| + Weight-3 Markowitz | ~2313ms | 1.08x slower | Phase3 pre-elim, centered coefficients |
| + Weight-K merges (K=3..10) | 2071ms | 0.96x (faster) | Generalized pre-elim, col-rows index |
| + Adaptive termination | ~1350ms | 1.6x faster | Stop sieve when relations suffice |
| + Polyselect parallel+cached | ~1250ms | 1.7x faster | Rayon Phase1/2, Dickman cache |
| + Early-exit GE + dep_limit | ~1100ms | 2.0x faster | Cap deps at 64, lazy map-back |
| + c30 param retuning | **~940ms** | **2.3x faster** | lim=20k, lpb=18, qmin=30k |

## Optimizations (55 commits on `feat/cado-parity-phase2`)

### Polynomial Selection
- **Murphy E-value scoring** with Dickman rho approximation for smoothness weighting
- **Two-phase screening**: Phase 1 scores by (lognorm + alpha) cheaply, Phase 2 does rotation + full Murphy E on top 100 candidates only
- **Centered base-m coefficients**: c_i in [-m/2, m/2) for smaller norms
- **Prime caching** in murphy_alpha via OnceLock (eliminates repeated sieve_primes allocation)
- **Rayon parallelization** of Phase 1 ad sweep and Phase 2 rotation
- **Cached Dickman table** (fixed 36.7s cold-start bottleneck)
- **Two-phase rotation search** (~6x speedup)

### Sieve
- **FK (Franke-Kleinjung) lattice walk** with partial-GCD reduced basis vectors
- **Single-comparison a_ok/b_ok** using opposite-sign u-vector invariant
- **NEON SIMD survivor scan** (16 bytes/cycle, 99.9% fast-path exits)
- **Sieve threshold separation**: sieve_mfb = 2/3 * cofactoring_mfb (reduces false positives by 90%)
- **2LP (two large primes)** support with separate sieve/cofactoring thresholds
- **Early-exit cofactoring**: skip algebraic side when rational fails

### Linear Algebra
- **Generalized weight-K Markowitz merges** (K=3..10) in pre-elimination
  - Phase 1: queue-based singleton removal (O(nnz))
  - Phase 2: batched weight-2 merges
  - Phase 3: weight-3 merges with Markowitz pivot selection
  - Phase 4: weight-K merges with flat column-to-rows index
  - Result: 7121 -> 3573 rows, dense GE 183ms -> 75ms
- **BitRow-based dep extraction** (word-level XOR instead of HashSet)
- **Early-exit GE**: stop after max_deps (default 64) dependencies found
- **dep_limit cap** with lazy map-back
- **split_at_mut** for zero-clone GE pivot elimination
- **Parallel map-back** and dependency randomization (Rayon)
- **Sequential GE** in single-threaded mode (avoids Rayon overhead)
- **NEON xor_with** for BitRow operations on aarch64
- **Lower Rayon threshold** (512 -> 128)

### Filter
- **Queue-based singleton removal**: O(n) instead of O(n*k) iterative rescan

### Square Root
- **Non-monic polynomial support** via scaling trick
- **Parallel first-batch attempts** in MT mode (rayon::scope + AtomicBool)
- Better dependency structure from deeper pre-elimination -> fewer sqrt attempts

## Key Parameters (c30, current)

```
degree=3, lim0/lim1=20000, lpb0/lpb1=18
mfb0/mfb1=20 (2LP), sieve_mfb0/mfb1=18 (tuned)
log_i=8, sieve_width=512, max_j=256
bucket_thresh=256, qmin=30000, qrange=500
```

## Scaling: c45 and c60 Readiness

| Aspect | c30 | c45 | c60 |
|--------|-----|-----|-----|
| Parameters defined | Yes | Yes | No (falls back to c45) |
| Sieve tested | Yes | Yes | No |
| LA (dense GE) | Yes | Yes | Yes (up to ~20k rows) |
| LA (Block Wiedemann) | Yes | Yes | Yes (auto above 20k rows) |
| Cofactoring (u128) | N/A | Yes | Yes |
| Full pipeline success | High | Partial | Untested |

### c45: Works, polynomial quality issue

Parameters tuned (degree=4, lim0=55k, lim1=65k, lpb0=18, lpb1=19, log_i=10).
Benchmark on 148-bit semiprime: ~61s total (56.9s sieve, 3.5s LA, 1.2s sqrt).
All subsystems functional. Known issue: some polynomial variants hit 200
trivial GCDs in sqrt — a poly selection quality problem, not infrastructure.

### c60: Infrastructure ready, needs setup

- No `c60()` parameter set — defaults to c45 (degree 4 vs needed 5-6)
- Block Wiedemann implemented and auto-activates above 20k rows
- u128 cofactoring handles larger algebraic norms
- Would run but with wrong polynomial degree and untuned sieve area

**To enable c60:**
1. Add `c60()` to `params.rs` (degree 5-6, lim ~200k+, lpb 22-24, log_i 11-12)
2. Validate polynomial selection for degree 5-6
3. Test and benchmark full pipeline

### Optimizations are size-independent

All key optimizations scale to any input size — none are c30-specific:

- FK lattice walk: parameterized by sieve_width/half_i
- Pre-elimination (phases 1-4): driven by matrix structure, not input size
- 2LP merge + sieve_mfb tuning: ratio-based (2/3 of mfb)
- Block Wiedemann: sparse SpMV, O(n²) for large matrices
- SIMD survivor scan: operates on any sieve region size
- Queue-based filter: O(n) regardless of relation count

### Pipeline
- **Adaptive sieve termination** with fine-grained stopping (est_dense_cols)
- **Parallel remap_hybrid** and matrix build
- **Cantor-Zassenhaus** for quadratic character selection (replaces O(p) scan)
- **Pipeline observability** framework with per-stage JSON timing

## Remaining Gaps (c30)

- **Polynomial quality**: Base-m with Y1=1 yields ~14 rels/SQ vs CADO's Kleinjung (Y1>1) ~308 rels/SQ. Compensated by 2LP but produces larger matrices.
- **Scatter phase**: ~400ms (65% of sieve) is still the dominant cost. FK walk is efficient but per-prime partial-GCD overhead adds up for ~2300 large primes per SQ.
- **Retry variance**: ~1-in-20 runs hits matrix deficit, triggering 700-1200ms retry. Further reducing this is the main path to lower p95 latency.

See also: [Optimization Findings](2026-03-12-rust-nfs-optimization-findings.md) for detailed lessons learned.
