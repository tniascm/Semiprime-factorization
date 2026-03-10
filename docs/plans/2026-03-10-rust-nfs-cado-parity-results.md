# Rust-NFS vs CADO-NFS: Parity Achieved (c30)

## Results

| Metric | Rust-NFS | CADO-NFS | Ratio |
|--------|----------|----------|-------|
| Single-threaded CPU | **2071ms** | 2149ms | **0.96x** (3.6% faster) |
| Multi-threaded wall | **773ms** | ~5780ms | **0.13x** (7.5x faster) |

Benchmark: `684217602914977371691118975023` (100-bit semiprime, c30 params).
5-run averages, Apple M-series, `RAYON_NUM_THREADS=1` for single-threaded.

## Component Breakdown (single-threaded)

| Stage | Before (baseline) | After | Saved |
|-------|-------------------|-------|-------|
| Sieve (FK scatter) | 1580ms | 1566ms | 14ms |
| Linear Algebra | 341ms | 242ms | 99ms |
| Square Root | 194ms | 60ms | 134ms |
| Filter | 22ms | 22ms | 0ms |
| Polyselect + overhead | 178ms | 181ms | -3ms |
| **Total** | **2313ms** | **2071ms** | **242ms** |

## Optimizations (29 commits on `feat/cado-parity-phase2`)

### Polynomial Selection
- **Murphy E-value scoring** with Dickman rho approximation for smoothness weighting
- **Two-phase screening**: Phase 1 scores by (lognorm + alpha) cheaply, Phase 2 does rotation + full Murphy E on top 100 candidates only
- **Centered base-m coefficients**: c_i in [-m/2, m/2) for smaller norms
- **Prime caching** in murphy_alpha via OnceLock (eliminates repeated sieve_primes allocation)

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

### Filter
- **Queue-based singleton removal**: O(n) instead of O(n*k) iterative rescan

### Square Root
- **Non-monic polynomial support** via scaling trick
- Better dependency structure from deeper pre-elimination -> fewer sqrt attempts

## Key Parameters (c30)

```
degree=3, lim0/lim1=30000, lpb0/lpb1=17
mfb0/mfb1=36 (2LP), sieve_mfb0/mfb1=24 (tuned)
log_i=8, sieve_width=512, max_j=256
bucket_thresh=256, qmin=50000, qrange=1000
```

## Remaining Gaps

- **Polynomial quality**: Base-m with Y1=1 yields ~12 rels/SQ vs CADO's Kleinjung (Y1>1) ~308 rels/SQ. Compensated by 2LP but produces larger matrices.
- **Scatter phase**: 1566ms (75% of sieve) is still the dominant cost. FK walk is efficient but per-prime partial-GCD overhead adds up for ~6000 large primes per SQ.
- **Small sieve**: ~228ms could be reduced with SIMD pattern tiling for primes 2,3,5,7.
