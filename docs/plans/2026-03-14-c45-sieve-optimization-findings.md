# c45 Sieve Optimization Findings & Next Steps

Date: 2026-03-14
Branch: `feat/kleinjung-polyselect` (commit `2442406`)

## Current c45 Performance (post-Kleinjung polyselect, 148-bit, seed 42, ST)

```
polyselect: 173ms (1057 candidates -> 5 kept, best E=2.49e-11)
sieve:      19009ms (34350 raw rels, 3835 SQs, 103K survivors)
  FK scatter: 13001ms (68%) — alg ~5.0s, rat ~4.6s, misc ~3.4s
  region_scan: 5796ms (31%) — small_sieve ~3.4s, norm ~0.7s, bucket+scan ~0.6s
  cofactor:   164ms (<1%)
filter:     85ms
LA:         676ms  (12757×11576 matrix, 64 deps)
sqrt:       277ms  (factor found at dep #3)
total:      20071ms
```

**Success: 100% (1/1 factored)**, good matrix quality (1181 surplus rows), fast sqrt.

## Sieve Bottleneck Analysis

### FK Scatter Setup (13.0s, 68% of sieve)

The FK scatter processes ~8400 (prime, root) pairs per SQ, 3835 SQs = **31.8M entries total**.

**Per-entry cost breakdown (~410ns each):**
| Component | Time | Notes |
|-----------|------|-------|
| `transform_root` | ~150ns | 2 muls + `mod_inverse` (extended GCD) |
| Partial-GCD | ~80ns | ~5 iterations, each with 1 division |
| FK walk | ~50ns | ~12 steps with bucket push |
| Branches/overhead | ~130ns | Fallback paths, vector push, function calls |

The `mod_inverse` (extended GCD) alone accounts for ~100ns per call, or ~3.2s total across all entries (33% of FK scatter time).

### Small Sieve (3.4s, 18% of sieve)

- ~97 small primes (< 512), sieve_width=1024, max_j=512
- ~2.8M subtract ops per SQ, ~1ns each → ~2.8ms per SQ
- Scales linearly with number of primes and sieve area

### Region Scan (2.4s, 13% of sieve)

- Norm initialization: 0.7s (block-16 approximation, degree-4 polynomial)
- Bucket apply + scan: 0.6s
- Remaining: 1.1s overhead (block iteration, memory)

## Experiments Conducted

### 1. Polynomial Skewness in Q-Lattice (REVERTED)

**Change:** Pass `optimal_polynomial_skewness(f_coeffs)` ≈ 99.7 to `reduce_qlattice()` instead of hardcoded `1.0`.

**Result:**
- Sieve speedup: 19.0s → 13.4s (29% faster)
- BUT: 100% trivial GCD in sqrt across ALL 5 polynomial variants (20315 total sqrt attempts)
- Dependency lengths dropped dramatically: p50≈170 vs p50≈5658 (normal)
- sq_cols dropped: 1202 vs 3311 (normal)

**Root cause:** The skewed q-lattice changes the (a,b) distribution, concentrating relations in a narrow region. This produces a degenerate matrix with short dependencies that systematically give γ(m) ≡ ±rational_product (mod N).

**Decision:** Reverted. Would require fundamental changes to the matrix/sqrt pipeline to accommodate skewed distributions.

### 2. log_i Sweep (Sieve Area Scaling)

| log_i | Area | Sieve Time | SQs | Rels/SQ | Factor? |
|-------|------|-----------|-----|---------|---------|
| 9 (baseline) | 512K | 19.0s | 3835 | 9.0 | YES |
| 10 | 2M | 24.2s¹/71.5s² | 2231/5686 | ~6.3 | 1/2 variants |
| 11 | 8M | 101.0s | 2761 | 12.3 | YES (389s total) |

¹ First variant only. ² Variant that collected enough rels.

**log_i=10 with larger FB** (lim=55K/65K): sieve=56.1s, factor=none.
All larger log_i configurations are strictly worse than log_i=9.

### 3. FB Size Sweep

| lim0/lim1 | Sieve Time | SQs | Factor? | Notes |
|-----------|-----------|-----|---------|-------|
| 40K/45K (baseline) | 19.0s | 3835 | YES | Current params |
| 25K/30K | 85.0s+ | 20546+ | NO | Too few rels/SQ, needs too many SQs |
| 20K/20K | 3535s+ | 980840+ | NO | Extreme SQ count, useless |

Current FB size (lim=40K/45K) is confirmed near-optimal. Smaller FBs produce fewer rels/SQ,
requiring vastly more SQs that overwhelm any per-SQ savings.

### 4. log_i=10 (original finding, detailed)

**Change:** `RUST_NFS_OVR_LOG_I=10` → sieve area 4x larger (2M vs 512K positions).

**Result:**
- Variant 1: sieve=24.2s (setup=13.0s, scan=11.0s) — 27% SLOWER
- FK scatter setup: similar (~13s) because it's O(N_fb × N_sq), not O(area)
- Only 1.7x more rels/SQ (not 4x as expected), needing 2231 SQs (vs 3835)
- Region scan: 2x more expensive per SQ
- Net: worse because setup is amortized over fewer SQs but scan is proportionally more

**Decision:** log_i=9 remains optimal for scatter sieve architecture.

## Planned Optimization: Batch Modular Inversion

### Concept

Replace per-(entry, SQ) `mod_inverse` calls with Montgomery's batch trick:
- For each FB prime p, compute all denominators across all SQs in the batch
- 1 `mod_inverse` per prime (instead of 1 per entry per SQ)
- Cost: 1 ext_GCD + 3(n-1) multiplications for n SQs

### Expected Impact

With batch_size=64 SQs:
- Per-entry cost: 410ns → ~300ns (eliminate 100ns mod_inverse overhead)
- FK scatter: 9.6s → ~6.1s (36% speedup)
- Total sieve: 19.0s → ~15.4s

### Implementation Status

- `batch_mod_inverse()` function added to `arith.rs` (uncommitted)
- Inline hints added to `extended_gcd` and `mod_inverse`
- Integration into FK scatter NOT YET DONE — requires restructuring the precomputation loop

### Integration Plan

1. Before per-SQ `par_iter` loop: reduce all q-lattices for the batch
2. For each FB prime: batch-compute denominators, batch-invert, compute all r' values
3. Store precomputed `r'` in a 2D array `[n_entries × n_sq]` (~2.4MB per batch)
4. Per-SQ processing reads precomputed `r'`, skips `transform_root`, does partial-GCD + FK walk
5. Sentinel `u64::MAX` for projective/failed roots → row-by-row fallback

## Out of Scope (Requires Fundamentally Different Architecture)

### 1. Sub-1s Sieve (Phase 3 Target: 1000ms ST)

Even with batch inversion, the minimum possible sieve time is:
- FK setup (partial-GCD only, no mod_inverse): ~2.5s
- FK walk: ~1.6s
- Small sieve: ~3.4s
- Norms + scan: ~1.8s
- **Minimum: ~9.3s with current scatter sieve**

**Reaching <1s requires a lattice sieve (Task 10)** — the scatter sieve's O(N_fb × N_sq) scaling is fundamentally too slow.

### 2. Polynomial Skewness

The q-lattice skewness optimization gives 29% sieve speedup but breaks sqrt with 100% trivial GCD. Fixing this requires either:
- A new sqrt algorithm that handles skewed (a,b) distributions
- Adding diversity constraints to the matrix builder for skewed distributions
- Or a completely different approach to exploiting polynomial skewness (e.g., at the norm computation level rather than q-lattice level)

### 3. Partial-GCD Elimination

The partial-GCD (2.5s total) cannot be easily eliminated — it converts O(max_j) row-by-row walks into O(area/p) FK walks, saving ~1.9μs per entry. The 80ns partial-GCD cost is well worth it.

### 4. SIMD-ized Small Sieve

The small sieve (3.4s) could potentially be SIMD-optimized using NEON/SSE for the `saturating_sub` inner loop. Currently processes 1 byte at a time. With NEON, could process 16 bytes simultaneously → up to 16x speedup theoretically, but alignment and loop overhead would limit it to ~4-8x.

## Summary of Current Position

| Metric | Current | Phase 1 Target | Phase 3 Target |
|--------|---------|----------------|----------------|
| c45 ST total | 20.1s | <8s | <1s |
| c45 sieve | 19.0s | <7s | <0.5s |
| c45 success rate | 100% | ≥80% | ≥90% |
| c30 ST median | ~940ms | <1000ms | <500ms |
| Architecture | Scatter sieve | Scatter + batch inv | Lattice sieve |

**Next concrete steps:**
1. Complete batch inversion integration into FK scatter (expected: sieve 19s → 15s)
2. SIMD-ize small sieve inner loop (expected: 3.4s → ~1s)
3. Profile and optimize region scan overhead (expected: ~1s savings)
4. Investigate lattice sieve for Phase 3 (required for <1s)
