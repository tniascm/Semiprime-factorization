# Rust-NFS Optimization: Findings, Reflections, and Lessons Learned

**Date**: 2026-03-12
**Branch**: `feat/cado-parity-phase2` (55 commits, merged to main)
**Goal**: Sub-1000ms single-threaded, sub-500ms multi-threaded c30 factorization

## Final Results

| Metric | Baseline | Final | Target | Status |
|--------|----------|-------|--------|--------|
| ST median | 2149ms | ~940ms | <1000ms | Achieved |
| MT mean | 772ms | ~350ms | <500ms | Achieved |
| Reliability | ~95% | 100% | 100% | Achieved |

Benchmarked on 100-bit balanced semiprimes, Apple M-series, 50 runs across 5 seeds.

### Per-Stage Breakdown (ST, seed 42 average)

| Stage | Baseline | Final | Speedup |
|-------|----------|-------|---------|
| Polyselect | 177ms | ~60ms | 2.9x |
| Sieve | 1611ms | ~590ms | 2.7x |
| Filter | 23ms | ~15ms | 1.5x |
| LA | 270ms | ~95ms | 2.8x |
| Sqrt | 68ms | ~50ms | 1.4x |
| **Total** | **2149ms** | **~810ms** | **2.7x** |

Note: totals include overhead (timing framework, remap, matrix build) not shown
in stage breakdown.

## What Worked (ranked by impact)

### 1. Parameter Retuning (~500ms ST saved)

The single biggest win was not code — it was changing c30 parameters:

```
lim:  30000 → 20000  (33% smaller factor base)
lpb:  17 → 18        (2x larger large-prime bound)
mfb:  18 → 20        (2LP tolerance raised)
qmin: 50000 → 30000  (lower special-q start)
```

**Why it works**: Smaller factor base (lim=20000) means fewer primes in scatter
(~2300 vs ~3400), which directly reduces the 75%-of-time FK walk. Higher lpb=18
compensates by catching more cofactoring successes per special-q (~14 rels/SQ vs
~12), so fewer SQs are needed overall. The matrix is also ~30% smaller, cutting
LA from ~195ms to ~100ms.

**Lesson**: algorithm-level parameter tuning dominates micro-optimization. One
parameter change saved more than all SIMD optimizations combined.

### 2. Adaptive Sieve Termination (~400ms ST saved)

Instead of always iterating over the full qrange, stop sieving once enough
relations are collected. The target is `est_dense_cols * adaptive_rows_ratio`
where `est_dense_cols` estimates the matrix column count from factor base sizes.

Key finding: `adaptive_rows_ratio=1.10` is optimal. Lower values (0.90-1.00)
cause matrix deficit retries that cost 700-1200ms each — far worse than the
few hundred ms of "excess" sieving at 1.10.

**The est_dense_cols overestimate**: The column estimator overestimates by ~37%
(8272 raw vs ~5695 after compact+prune). This is partially compensated by ~21%
relation loss during remap and ~10% during filter prune, making ratio=1.10 close
to optimal when accounting for all three correction factors.

### 3. Polyselect Parallelization (~120ms MT saved)

Rayon-parallelized both Phase 1 (ad candidate sweep) and Phase 2 (rotation
optimization). In ST mode, the cached Dickman table and fast rotation scoring
were the main wins (fixing a 36.7s cold-start bottleneck from repeated prime
sieve allocation).

### 4. LA Pipeline Optimization (~170ms ST saved)

Multiple orthogonal improvements:
- **Weight-K Markowitz merges** (K=3..10): pre-elimination reduces matrix from
  ~7100 to ~3500 rows before dense GE
- **Early-exit GE**: stop after finding `max_deps` dependencies (default 64)
  instead of processing all pivot columns
- **dep_limit cap**: extract only needed dependencies, lazy map-back
- **Parallel map-back**: Rayon for history-to-BitRow conversion
- **Sequential GE in ST mode**: avoid Rayon overhead when single-threaded

### 5. Parallel Sqrt Attempts (~20ms MT saved)

Try 2-4 dependencies simultaneously with `rayon::scope` in MT mode. First
success cancels remaining via `AtomicBool`. Dependencies sorted by length
(ascending) so shortest are tried first.

## What Didn't Work

### NEON SIMD for GE xor_with (0ms impact)

Manually wrote aarch64 NEON intrinsics (`veorq_u64`) for the BitRow XOR hot
loop. Confirmed the compiler was NOT auto-vectorizing the scalar path. The NEON
code IS emitted (verified via assembly: `eor.16b`, `ldp q0, q1`). But: zero
measurable speedup. The bottleneck is cache misses (random-access row XOR
pattern), not XOR throughput. The data doesn't fit in L1 and each pivot row
elimination touches a random subset of other rows.

**Lesson**: profile before optimizing. "Hot loop" doesn't mean "optimization
opportunity" when the bottleneck is memory access pattern, not compute.

### Matrix Probing (negative impact, +100ms)

Tried periodically building the actual matrix mid-sieve to check if we have
enough relations. Each probe costs ~25ms (remap + compact + prune). With 4
probes per run, added ~100ms overhead that exceeded any early-stopping benefit.

**Lesson**: cheap estimates beat expensive exact checks for stopping criteria.
The `est_dense_cols` approximation, despite 37% overestimate, is far better
than spending 100ms on precision.

### Lower adaptive_rows_ratio (reliability regression)

Tried ratios of 0.90, 0.95, 1.00. All caused some semiprimes to hit matrix
deficit after remap+prune, triggering polynomial variant retries at 700-1200ms
each. Mean time was WORSE despite faster per-attempt sieve.

| Ratio | Mean ST | Retries |
|-------|---------|---------|
| 1.10 | 948ms | ~0.5/10 |
| 1.00 | 1050ms | ~2/10 |
| 0.95 | 1100ms | ~3/10 |
| 0.90 | FAIL | ~4/10 |

**Lesson**: reliability trumps speed. A 5% faster sieve that fails 20% of the
time is much worse than a slightly conservative approach.

### Single-Bucket Fast Path (no activation)

Built `SingleBucketVec` and `BucketStorage` enum to skip bucket-index overhead
when `n_buckets=1`. For c30 with `BUCKET_REGION=65536` and sieve area=131072,
`n_buckets=2`, so the optimization never activates. Would need
`LOG_BUCKET_REGION=17` (128KB) to trigger — possible on Apple Silicon's 192KB
L1 but untested.

**Lesson**: verify assumptions about runtime behavior before building
optimizations. Check `n_buckets` with actual parameters first.

### Higher mfb (no effect)

Tried `mfb=22` (from 20). No measurable difference because `sieve_mfb`
(unchanged at 18) controls survivor detection. The cofactoring mfb only affects
post-survivor processing, which handles <1% of positions. The sieve threshold
is the gate, not the cofactoring bound.

### Smaller admax for Polyselect (worse polynomials)

Reduced ad search range from 5000 to 3000 for 30ms polyselect savings. But
lower-quality polynomials cost ~60ms more in sieve per run. Net negative.

**Lesson**: polynomial quality has outsized downstream effects. Time spent
in polyselect is an investment, not overhead.

## Architecture Decisions

### Observability Framework

Created `timing.rs` with `StageTimer`, `StageResult`, and `PipelineTimings`.
Per-stage JSON output (via `RUST_NFS_TIMING_JSON=1`) with sub-stage metrics:

```json
{"stage": "sieve", "wall_ms": 256, "special_qs": 580, "ms_per_sq": 0.44,
 "relations_raw": 7200, "rels_per_sq": 12.4, "survivors_per_sq": 39}
```

This was critical for diagnosis. Without per-stage timing, all optimization
would have been guesswork. Configurable timeouts via environment variables
(`RUST_NFS_SIEVE_TIMEOUT_MS`, etc.) provide graceful degradation.

### Adaptive vs Fixed Sieve

The adaptive approach (stop when relations suffice) beats fixed qrange by
~40% in sieve time. The key insight is that different semiprimes have wildly
different relation yields per SQ (8-20), so a fixed qrange either undershoots
or wastes 400+ SQs.

### Dense GE vs Block Wiedemann

Dense GE with pre-elimination is faster for c30 matrices (~3500 rows).
Block Wiedemann auto-activates above 20k rows but has higher constant factor.
The pre-elimination pipeline (phases 1-4) is the key: it reduces the matrix
enough that dense GE dominates BW for all c30/c35/c40 inputs.

### 2LP (Two Large Primes)

Separate sieve_mfb (tight threshold for survivor detection) from cofactoring
mfb (looser bound for actual factoring). sieve_mfb=18 filters out 90% of
false positives that mfb=20 would pass through, while cofactoring at mfb=20
catches the extra smooth relations among survivors.

## Variance Analysis

The remaining performance variance is dominated by:

1. **Polynomial quality variance** (~200ms): some base-m polynomials produce
   better norms than others, affecting rels/SQ yield
2. **Retry overhead** (~700-1200ms when triggered): matrix deficit after
   remap causes a polynomial variant retry, approximately 1-in-20 runs
3. **Sqrt attempt count** (~30ms per dep): ranges from 1-8 dependencies
   tried; dependent on algebraic structure of the dependency

The retry mechanism is the main source of outliers. A run that retries once
takes ~1700ms vs ~900ms for a single-attempt success.

## Scaling Considerations

All optimizations are parameterized, not c30-specific:

- FK scatter: scales with factor base size (lim parameter)
- Pre-elimination: driven by matrix structure, not input size
- Adaptive termination: uses est_dense_cols which scales with lim
- SIMD survivor scan: operates on any sieve region size
- Parallel sqrt: works with any dependency count

For c45 (148-bit): pipeline works but takes ~61s (56.9s sieve). The
bottleneck shifts entirely to sieve scatter at larger sizes. Polynomial
quality becomes critical — Kleinjung polynomial selection (Y1 > 1) would
yield ~25x more rels/SQ than base-m.

## Commit History (55 commits)

### Infrastructure (3 commits)
- Pipeline observability framework with per-stage timing
- Documentation and benchmark results
- Scaling assessment for c45/c60

### Polynomial Selection (11 commits)
- Murphy E-value scoring with Dickman rho
- Two-phase candidate screening (size+alpha then rotation)
- Centered base-m coefficients for smaller norms
- Rayon parallelization of Phase 1 and Phase 2
- Cached Dickman table (fixed 36.7s cold-start)
- Two-phase rotation search (~6x speedup)

### Sieve (8 commits)
- FK walk with single-comparison a_ok/b_ok
- NEON SIMD survivor scan
- Sieve/cofactoring mfb separation (2LP)
- Small sieve SIMD for p=2,3,5,7
- Early-exit cofactoring
- Trial division p^2>n early exit

### Linear Algebra (11 commits)
- Weight-K Markowitz merges (K=3..10)
- Column-to-rows index for O(1) merge lookup
- Early-exit GE with max_deps
- dep_limit cap with lazy map-back
- split_at_mut for zero-clone GE
- Parallel map-back and dependency randomization
- Sequential GE in single-threaded mode
- NEON xor_with for BitRow
- Lower Rayon threshold (512 → 128)

### Filter (1 commit)
- Queue-based singleton removal (O(n) vs O(n*k))

### Sqrt (2 commits)
- Non-monic polynomial support via scaling trick
- Parallel first-batch attempts in MT mode

### Pipeline (6 commits)
- Adaptive sieve termination with fine-grained stopping
- Parallel remap_hybrid and matrix build
- Cantor-Zassenhaus for quadratic character selection
- Increased initial relation target for fewer adaptive windows

### Parameters (2 commits)
- sieve_mfb0/sieve_mfb1 fields
- c30 parameter retuning (lim, lpb, mfb, qmin)

### Reverted (2 commits)
- Single-bucket fast path (n_buckets=2, not 1)
- phase2_keep=50 regression

## Key Takeaways

1. **Parameters > Code**: The single biggest speedup came from changing four
   numbers in params.rs, not from any algorithmic or SIMD optimization.

2. **Reliability > Speed**: Aggressive early stopping saves time per attempt
   but increases retry rate. The retry cost (700-1200ms) dominates any per-
   attempt savings. Conservative ratios win on expected value.

3. **Profile First**: NEON SIMD for GE XOR was a technically correct optimization
   (compiler wasn't vectorizing, hand-written intrinsics were faster in isolation)
   that had zero end-to-end impact because the bottleneck was cache misses.

4. **Cheap Estimates > Expensive Probes**: est_dense_cols overestimates by 37%
   but costs 0ms. Matrix probing is exact but costs 100ms. The estimate wins.

5. **Polynomial Quality Compounds**: Time spent in polyselect pays back 2-3x
   in sieve efficiency. Cutting polyselect corners (smaller admax, fewer Phase 2
   candidates) consistently backfired.

6. **Observability Enables Everything**: The timing framework paid for itself
   immediately. Every diagnosis, every parameter experiment, every regression
   check used per-stage timing. Build instrumentation first.

7. **Pre-elimination is Transformative**: Weight-K Markowitz merges reduced the
   dense GE matrix by 50%, cutting LA time by ~65%. This is a multiplicative
   speedup that compounds with all other LA improvements.

8. **Adaptive > Fixed**: Different semiprimes have 2x+ variance in relation
   yield. Fixed qrange wastes 30-40% of sieve time on easy inputs. Adaptive
   termination captures this variance automatically.
