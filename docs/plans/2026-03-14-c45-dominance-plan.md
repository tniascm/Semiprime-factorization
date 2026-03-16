# c45 Dominance Plan: 1000ms ST / 500ms MT

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Achieve <1000ms ST / <500ms MT for c45 (148-bit) factorization with 100% reliability, scalable to c50+.

**Architecture:** Replace FK scatter sieve with CADO-style row-by-row enumeration (eliminates per-prime partial-GCD), add smart norm initialization (lg2 approximation replacing Horner), fix hd_residual remap for 100% reliability, and recalibrate parameters for the new sieve architecture.

**Tech Stack:** Rust, `rug` (GMP bindings), `rayon` (parallelism), NEON SIMD (aarch64)

---

## Current State

| Metric | Potapov-NFS | CADO-NFS | Gap |
|--------|----------|----------|-----|
| c45 total (ST) | 20s (67% success) | 8.92s (100%) | 2.2x slower |
| c45 sieve | 19s | 2.16s | 8.8x slower |
| c45 LA | 0.7s | 1.1s | 1.6x faster |
| c45 sqrt | 0.25s | 0.14s | 1.8x slower |
| c30 total (ST) | 648ms | 5.6s | 8.6x faster |

**Sieve breakdown (19s per successful c45):**
- FK scatter setup (partial-GCD per prime): 13s (65%)
- Region scan (norm init + small sieve + bucket apply + cofactor): 6s (35%)

**Root cause:** FK scatter does partial-GCD reduction per (prime, root) pair per special-q. For c45 with ~5000 large FB primes, that's ~5000 partial-GCD calls per SQ. CADO does zero per-prime pre-computation — it walks each sieve row with simple modular arithmetic.

## Impact Ranking

| Task | Expected Speedup | Reliability Impact | Risk |
|------|------------------|-------------------|------|
| 1: Row-by-row sieve | 5-10x sieve | None | Medium (new code) |
| 2: Smart norm init | 2-3x norm | None | Low |
| 3: hd_residual fix | None | 67% → 100% | Low |
| 4: Parameter recalibration | 1.5-3x overall | +reliability | Low |
| 5: Cofactor fast-path | 1.2-1.5x cofactor | None | Low |

---

### Task 1: Row-by-Row Sieve (Highest Impact)

**Files:**
- Create: `rust/potapov-nfs/src/sieve/rowsieve.rs`
- Modify: `rust/potapov-nfs/src/sieve/mod.rs`

**Context:** The FK scatter sieve (lines 1730-1944 of `sieve/mod.rs`) computes partial-GCD lattice reduction per (prime, root) per special-q. This is 65% of sieve time. CADO's approach (in `sieve/las-plattice.h` and `sieve/las-smallsieve.cpp`) instead processes one row at a time: for each row j, for each prime p with root r, compute starting offset `i0 = -(r*j) mod p` (one modular multiply) and walk with stride p. No partial-GCD needed.

**Step 1: Write the row-by-row scatter function.**

The new function replaces `scatter_bucket_updates_fk_batch()` for large primes:

```rust
/// Row-by-row bucket sieve: for each row j, for each large prime p,
/// compute starting position and walk with stride p.
///
/// This replaces the FK scatter approach which does per-prime partial-GCD.
/// Complexity: O(rows * n_primes / p_avg) vs FK's O(n_primes * partial_gcd_cost).
fn row_sieve_bucket_updates(
    entries: &[(u64, u64, u8)],    // (prime, root_in_qlattice, logp)
    sieve_width: usize,
    max_j: usize,
    half_i: i64,
    buckets: &mut BucketArray,
) {
    let sw = sieve_width as u64;

    for j in 0..max_j {
        let row_base = j * sieve_width;

        for &(p, r, logp) in entries {
            // Starting offset in row: i0 such that (i0 + half_i + r*j) ≡ 0 (mod p)
            // i.e., i0 = (-(half_i + r*j)) mod p
            let rj = ((r as u128 * j as u128) % p as u128) as u64;
            let offset = p - ((half_i as u64 % p + rj) % p);
            let mut i_pos = if offset >= p { 0 } else { offset };

            while i_pos < sw {
                let gpos = row_base + i_pos as usize;
                buckets.push(
                    gpos >> LOG_BUCKET_REGION,
                    BucketUpdate {
                        pos: (gpos & (BUCKET_REGION - 1)) as u16,
                        logp,
                    },
                );
                i_pos += p;
            }
        }
    }
}
```

**Key insight:** The root `r` in q-lattice coordinates is the transformed root, same as what `transform_root()` already computes. We reuse that transform but skip partial-GCD entirely.

**Step 2: Wire into `sieve_specialq()` as the default for large primes.**

Replace the calls to `scatter_bucket_updates_fk_batch()` at lines 322 and 344 with calls to `row_sieve_bucket_updates()`. Keep the FK path behind an env var `POTAPOV_NFS_USE_FK=1` for A/B testing.

```rust
let use_fk = std::env::var("POTAPOV_NFS_USE_FK")
    .map(|v| v == "1")
    .unwrap_or(false);

if use_fk {
    scatter_bucket_updates_fk_batch(...);
} else {
    // Transform roots once per SQ
    let transformed: Vec<(u64, u64, u8)> = entries.iter()
        .filter_map(|&(p, root, logp)| {
            transform_root(p, root, &qlat).ok().map(|rp| (p, rp, logp))
        })
        .collect();
    row_sieve_bucket_updates(&transformed, sieve_width, max_j, half_i, buckets);
}
```

**Step 3: Run c45 benchmark (3 semiprimes, seed 42).**

```bash
cargo run --release -- --bits 148 --semiprimes 3 --seed 42 --threads 1
```

Expected: sieve time drops from 19s to ~2-4s (5-10x improvement). The FK partial-GCD overhead (13s) is eliminated. The row walk cost is O(J * sum(W/p)) ≈ O(J * ln(lim/thresh)) which is similar to what CADO does.

**Step 4: Run c30 regression.**

```bash
cargo run --release -- --bits 100 --semiprimes 10 --seed 42 --threads 1
```

Expected: c30 median < 700ms (no regression — the change only affects large primes, which dominate c45 but not c30).

**Step 5: Commit.**

```bash
git add rust/potapov-nfs/src/sieve/rowsieve.rs rust/potapov-nfs/src/sieve/mod.rs
git commit -m "sieve: replace FK scatter with row-by-row enumeration for large primes"
```

---

### Task 2: Smart Norm Initialization

**Files:**
- Modify: `rust/potapov-nfs/src/sieve/mod.rs` (lines 396-451, norm init block)
- Modify: `rust/potapov-nfs/src/sieve/norm.rs`

**Context:** Current algebraic norm initialization uses full Horner evaluation per cell (or per block of 16). For degree-4 polynomials, this is ~30 cycles/cell: 4 multiplies + 4 adds + log2 + clamp. CADO uses `lg2_raw()` — a 3-cycle integer bit-scan that estimates log2 from the exponent field. The actual norm doesn't need to be precise — we only need log2 ± 1 bit, since the sieve subtracts log contributions and only cares about whether the residual is below threshold.

**Step 1: Add `lg2_approx_f64` helper.**

In `norm.rs`:
```rust
/// Fast approximate log2 using IEEE 754 exponent extraction.
/// Accuracy: ±1 bit (sufficient for sieve initialization).
#[inline(always)]
fn lg2_approx_f64(x: f64) -> f64 {
    // Extract bits, shift out mantissa, subtract bias
    let bits = x.to_bits();
    let exp = ((bits >> 52) & 0x7FF) as i64 - 1023;
    exp as f64
}
```

**Step 2: Replace per-cell Horner with block approximation using endpoint interpolation.**

For degree-d algebraic norm along a row, the norm is a degree-d polynomial in `i`. Instead of evaluating at every cell, evaluate at row endpoints and interpolate log2 linearly:

```rust
// Endpoints of this row segment
let a_lo = a0f * (i_start as f64) + a1f * j_f;
let b_lo = b0f * (i_start as f64) + b1f * j_f;
let norm_lo = eval_homogeneous_norm_f64(f_coeffs, a_lo, b_lo, d);
let log_lo = lg2_approx_f64(norm_lo);

let a_hi = a0f * (i_end as f64) + a1f * j_f;
let b_hi = b0f * (i_end as f64) + b1f * j_f;
let norm_hi = eval_homogeneous_norm_f64(f_coeffs, a_hi, b_hi, d);
let log_hi = lg2_approx_f64(norm_hi);

// Linear interpolation of log2 across the segment
let inv_len = 1.0 / (overlap_len as f64);
for k in 0..overlap_len {
    let t = k as f64 * inv_len;
    let log_val = log_lo + (log_hi - log_lo) * t;
    alg_sieve[local_start + k] = (log_val * scale).clamp(0.0, 255.0) as u8;
}
```

This reduces per-cell cost from ~30 cycles (Horner) to ~3 cycles (lerp + clamp).

**Step 3: Run c45 benchmark, compare norm init time.**

Expected: norm init drops from ~1.5s to ~0.3s for c45.

**Step 4: Run c30 regression.**

**Step 5: Commit.**

```bash
git add rust/potapov-nfs/src/sieve/mod.rs rust/potapov-nfs/src/sieve/norm.rs
git commit -m "sieve: fast log2 norm init via endpoint interpolation"
```

---

### Task 3: Fix hd_residual Remap for 100% Reliability

**Files:**
- Modify: `rust/potapov-nfs/src/pipeline.rs` (remap_hybrid function, ~line 3762-4070)

**Context:** On ~33% of c45 polynomial variants, `remap_hybrid` drops 38-40% of relations due to `hd_residual` failures. This happens when a prime p has k < degree roots (a "higher-degree ideal"), and the algebraic norm's factorization doesn't account for the residual ideal contribution. The check at line 3889 requires `residual_divisible` — that the unfactored part of the norm is divisible by p^(d-k). For some polynomials (especially non-monic Kleinjung ones), the trial-division factorization misses the HD contribution because the norm is expressed differently.

The fix is to make the HD check more lenient: instead of requiring exact p^(d-k) divisibility of the residual, track the HD ideal contribution as a single column flip (odd/even exponent of the HD ideal) regardless of the exact residual value. This matches how CADO-NFS handles inert and ramified primes.

**Step 1: Modify the hd_residual check to be exponent-parity based.**

Currently (line 3887-3916):
```rust
if hd_degree == 0 || !residual_divisible {
    invalid_reason = Some("hd_residual");
    break;
}
```

Change to: compute the total p-adic valuation of the algebraic norm, subtract the known root contributions, and use the remainder's parity for the HD column. Don't reject the relation.

```rust
// Instead of rejecting, compute HD exponent from total p-valuation
let total_v = p_valuation(alg_norm_abs, p);
let known_v: u32 = roots.iter().map(|&r| /* matching root exponent */).sum();
let hd_v = total_v.saturating_sub(known_v);
// Flip HD column if hd_v is odd
if hd_v % 2 == 1 {
    if let Some(hd_col) = gnfs_fb.hd_offset(prime_idx, poly_degree) {
        row.flip(base_hd_col + hd_col);
    }
}
```

**Step 2: Add `p_valuation` helper if not already present.**

```rust
fn p_valuation(mut n: u128, p: u64) -> u32 {
    let mut v = 0;
    let p128 = p as u128;
    while n > 0 && n % p128 == 0 {
        n /= p128;
        v += 1;
    }
    v
}
```

**Step 3: Run c45 benchmark with the fix.**

```bash
POTAPOV_NFS_HD_RESIDUAL_SAMPLE_LIMIT=10 cargo run --release -- --bits 148 --semiprimes 5 --seed 42 --threads 1
```

Expected: remap keep ratio goes from ~60% to ~98%+. Success rate goes from 67% to 100%.

**Step 4: Run c30 regression (10-run).**

**Step 5: Commit.**

```bash
git add rust/potapov-nfs/src/pipeline.rs
git commit -m "remap: fix hd_residual check to use p-valuation parity instead of residual divisibility"
```

---

### Task 4: Parameter Recalibration for New Sieve

**Files:**
- Modify: `rust/potapov-nfs/src/params.rs` (c45 parameters only)

**Context:** After Tasks 1-3, the sieve architecture is fundamentally different. The row-by-row approach has different cost characteristics: per-SQ cost is O(J * sum(W/p)) instead of O(n_primes * partial_gcd_cost). This means:
- Larger sieve regions (higher log_i) are now cheaper (FK per-prime cost was O(n_primes), now it's O(J*n_primes/p_avg))
- Higher lim values are now tolerable (more primes → more sieve hits, previously more primes → more partial-GCD cost)

**Step 1: Sweep log_i.**

With row-by-row sieve, increasing log_i from 9 to 10 doubles J but also doubles hits per prime. Net effect: more relations per SQ, fewer SQs needed.

```bash
# Current: log_i=9
cargo run --release -- --bits 148 --semiprimes 1 --seed 42 --threads 1
# Try log_i=10
POTAPOV_NFS_OVR_LOG_I=10 cargo run --release -- --bits 148 --semiprimes 1 --seed 42 --threads 1
```

**Step 2: Sweep lim (FB size).**

```bash
POTAPOV_NFS_OVR_LIM0=55000 POTAPOV_NFS_OVR_LIM1=65000 cargo run --release -- --bits 148 --semiprimes 1 --seed 42 --threads 1
POTAPOV_NFS_OVR_LIM0=80000 POTAPOV_NFS_OVR_LIM1=100000 cargo run --release -- --bits 148 --semiprimes 1 --seed 42 --threads 1
```

**Step 3: Sweep lpb and qmin.**

```bash
POTAPOV_NFS_OVR_LPB0=21 POTAPOV_NFS_OVR_LPB1=22 cargo run --release -- --bits 148 --semiprimes 1 --seed 42 --threads 1
POTAPOV_NFS_OVR_QMIN=50000 cargo run --release -- --bits 148 --semiprimes 1 --seed 42 --threads 1
```

**Step 4: Update `c45()` in `params.rs` with best combination.**

**Step 5: Run 5-semiprime reliability check + c30 regression.**

**Step 6: Commit.**

```bash
git add rust/potapov-nfs/src/params.rs
git commit -m "params: recalibrate c45 for row-by-row sieve architecture"
```

---

### Task 5: Cofactor u64 Fast-Path

**Files:**
- Modify: `rust/potapov-nfs/src/sieve/mod.rs` (cofactoring section, ~lines 522-610)

**Context:** After FB trial division, many c45 cofactors fit in u64 (cofactor < 2^64). Currently all cofactors go through u128 arithmetic. Adding a u64 fast-path for the common case avoids the overhead of 128-bit division and modular arithmetic.

**Step 1: Add dispatch after trial division.**

In the cofactoring section of `sieve_specialq()`, after computing `rat_cofactor` and `alg_cofactor`:

```rust
let (rat_smooth, rat_lp) = if rat_cofactor <= u64::MAX as u128 {
    cofactor::cofactorize(rat_cofactor as u64, &cofact_config_rat)
} else {
    cofactor::cofactorize_u128(rat_cofactor, &cofact_config_rat)
};
```

**Step 2: Run c45 benchmark.**

Expected: 1.2-1.5x cofactoring speedup (cofactoring is ~10-15% of sieve time after Tasks 1-2).

**Step 3: c30 regression.**

**Step 4: Commit.**

```bash
git add rust/potapov-nfs/src/sieve/mod.rs
git commit -m "sieve: add u64 fast-path for cofactoring when cofactor fits in 64 bits"
```

---

## Verification Plan

1. **Per-task:** `cargo test --release --lib` (all tests must pass)
2. **Per-task:** c30 regression — 10-run seed 42, median < 700ms
3. **After Task 3:** c45 5-run reliability check (target: 100% success)
4. **After Task 4:** c45 10-run ST benchmark — target < 2s mean
5. **After all tasks:** c45 20-run ST + MT benchmark — targets: < 1000ms ST, < 500ms MT
6. **Cross-size:** c30/c35/c40 regression after each task

## Critical Files

- `rust/potapov-nfs/src/sieve/mod.rs` — FK scatter (to be replaced), norm init, cofactoring
- `rust/potapov-nfs/src/sieve/rowsieve.rs` — new row-by-row sieve (Task 1)
- `rust/potapov-nfs/src/sieve/norm.rs` — norm initialization helpers
- `rust/potapov-nfs/src/pipeline.rs` — remap_hybrid hd_residual fix (Task 3)
- `rust/potapov-nfs/src/params.rs` — c45 parameter set (Task 4)
- `rust/potapov-nfs/src/sieve/lattice.rs` — q-lattice reduction, transform_root
- `rust/potapov-nfs/src/sieve/bucket.rs` — BucketArray, BucketUpdate
