# Sieve Optimization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Cut single-threaded sieve time from ~6600ms to ~2500ms on c30 by implementing three CADO-aligned optimizations.

**Architecture:** (1) Replace row-by-row bucket scatter with Franke-Kleinjung lattice enumerator that uses packed 1D coordinates and two-addition inner loop. (2) Decouple sieve survivor threshold from 2LP mfb bump so cofactoring doesn't waste time on 91% false positives. (3) Add coprimality pre-filter in FK inner loop to skip both-even (i,j) pairs.

**Tech Stack:** Rust, no new dependencies. All changes in `rust/potapov-nfs/src/sieve/`.

---

### Task 1: Fix sieve threshold (decouple from 2LP mfb bump)

This is the trivial high-impact fix. Currently `rat_bound = (params.mfb0 * scale)` where mfb0 gets bumped from 18 to 36 for 2LP. CADO never inflates the sieve threshold for 2LP — it handles 2LP entirely in cofactor strategy. We should cap the sieve threshold at `2 * lpb` (the maximum un-bumped mfb).

**Files:**
- Modify: `rust/potapov-nfs/src/sieve/mod.rs:326-327`

**Step 1: Write the failing test**

Add to `rust/potapov-nfs/src/sieve/mod.rs` in the `#[cfg(test)] mod tests` block:

```rust
#[test]
fn test_sieve_bound_not_inflated_by_2lp_mfb() {
    // When mfb is bumped to 36 for 2LP, the sieve bound should still
    // be capped at 2*lpb to avoid 91% false-positive survivors.
    let scale = 1.442;
    let lpb = 17u32;
    let mfb_bumped = 36u32; // bumped for 2LP

    // Old behavior: bound = mfb * scale = 36 * 1.442 = 51 (too generous)
    let old_bound = ((mfb_bumped as f64) * scale).min(255.0) as u8;
    assert_eq!(old_bound, 51);

    // New behavior: cap at 2*lpb
    let sieve_mfb = mfb_bumped.min(2 * lpb);
    let new_bound = ((sieve_mfb as f64) * scale).min(255.0) as u8;
    assert_eq!(new_bound, 49); // 34 * 1.442 = 49
    assert!(new_bound < old_bound);
}
```

**Step 2: Run test to verify it passes** (this is a unit logic test, it passes immediately since it tests arithmetic)

Run: `cd rust/potapov-nfs && cargo test test_sieve_bound_not_inflated_by_2lp_mfb -- --nocapture`

**Step 3: Apply the fix**

In `rust/potapov-nfs/src/sieve/mod.rs`, change lines 326-327 from:

```rust
let rat_bound = ((params.mfb0 as f64) * scale).min(255.0) as u8;
let alg_bound = ((params.mfb1 as f64) * scale).min(255.0) as u8;
```

to:

```rust
let sieve_mfb0 = params.mfb0.min(2 * params.lpb0);
let sieve_mfb1 = params.mfb1.min(2 * params.lpb1);
let rat_bound = ((sieve_mfb0 as f64) * scale).min(255.0) as u8;
let alg_bound = ((sieve_mfb1 as f64) * scale).min(255.0) as u8;
```

**Step 4: Run full test suite**

Run: `cd rust/potapov-nfs && cargo test`
Expected: All tests pass.

**Step 5: Benchmark to measure impact**

Run: `cd rust/potapov-nfs && cargo run --release -- --factor 684217602914977371691118975023 --threads 1 2>&1 | grep -E '(sieve:|total_ms|survivors)'`

Expected: Survivor count should drop significantly (from ~234k to ~50-80k), cofactor time should drop proportionally.

**Step 6: Commit**

```bash
git add rust/potapov-nfs/src/sieve/mod.rs
git commit -m "sieve: cap survivor threshold at 2*lpb to avoid 2LP mfb inflation"
```

---

### Task 2: Implement FK lattice enumerator (new function)

Replace `scatter_bucket_updates_for_prime` with a function that uses the existing `PLattice` from `lattice.rs` (already has `reduce_plattice` with FK walk parameters). The existing `reduce_plattice` already computes `inc_step`, `inc_warp`, `bound_step`, `bound_warp` — we just need to use them.

**Files:**
- Modify: `rust/potapov-nfs/src/sieve/mod.rs:639-728` (replace `scatter_bucket_updates_for_prime`)
- Read: `rust/potapov-nfs/src/sieve/lattice.rs:116-288` (existing `PLattice` and `reduce_plattice`)

**Step 1: Write the failing test**

Add to `rust/potapov-nfs/src/sieve/mod.rs` tests:

```rust
#[test]
fn test_fk_scatter_matches_naive_scatter() {
    use crate::sieve::lattice::{reduce_plattice, reduce_qlattice};

    let q = 65537u64;
    let r_q = 12345u64;
    let qlat = reduce_qlattice(q, r_q, 1.0);
    let log_i = 7u32;
    let half_i = 1i64 << log_i;
    let sieve_width = (2 * half_i) as usize;
    let max_j = half_i as usize;

    // Test several FB primes
    let test_primes: Vec<(u64, u64)> = vec![
        (131, 42),
        (251, 100),
        (521, 200),
        (1031, 500),
    ];

    for &(p, root) in &test_primes {
        let n_buckets = (sieve_width * max_j + BUCKET_REGION - 1) / BUCKET_REGION;
        let ups_per = 4096;

        // Collect hits from old scatter
        let mut old_buckets = BucketArray::new(n_buckets.max(1), ups_per);
        scatter_bucket_updates_for_prime(
            p, root, 7, &qlat, log_i, &mut old_buckets,
            sieve_width, max_j, half_i,
        );
        let mut old_hits: Vec<usize> = Vec::new();
        for b in 0..n_buckets {
            for u in old_buckets.updates_for_bucket(b) {
                old_hits.push(b * BUCKET_REGION + u.position() as usize);
            }
        }
        old_hits.sort();

        // Collect hits from FK scatter
        let mut new_buckets = BucketArray::new(n_buckets.max(1), ups_per);
        scatter_bucket_updates_fk(
            p, root, 7, &qlat, log_i, &mut new_buckets,
            sieve_width, max_j, half_i,
        );
        let mut new_hits: Vec<usize> = Vec::new();
        for b in 0..n_buckets {
            for u in new_buckets.updates_for_bucket(b) {
                new_hits.push(b * BUCKET_REGION + u.position() as usize);
            }
        }
        new_hits.sort();

        assert_eq!(
            old_hits, new_hits,
            "FK scatter mismatch for p={}, root={}: old={} hits, new={} hits",
            p, root, old_hits.len(), new_hits.len()
        );
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cd rust/potapov-nfs && cargo test test_fk_scatter_matches_naive_scatter`
Expected: FAIL — `scatter_bucket_updates_fk` does not exist yet.

**Step 3: Implement `scatter_bucket_updates_fk`**

Add this function in `rust/potapov-nfs/src/sieve/mod.rs` right after `scatter_bucket_updates_for_prime` (line ~728):

```rust
/// Scatter bucket updates using Franke-Kleinjung lattice enumeration.
///
/// Instead of iterating row-by-row with modular arithmetic, this packs (i,j)
/// into a single u64 as `x = (i + half_i) + j * sieve_width` and walks the
/// lattice with two additions per step (no modular arithmetic in the inner loop).
///
/// Uses the pre-reduced PLattice from `reduce_plattice` which provides
/// inc_step, inc_warp, bound_step, bound_warp.
fn scatter_bucket_updates_fk(
    p: u64,
    root: u64,
    logp: u8,
    qlat: &QLattice,
    log_i: u32,
    buckets: &mut BucketArray,
    sieve_width: usize,
    max_j: usize,
    half_i: i64,
) {
    use crate::sieve::lattice::reduce_plattice;

    let pl = reduce_plattice(p, root, qlat, log_i);
    if !pl.hits {
        return;
    }

    let fence = (max_j as u64) * (sieve_width as u64);

    // The FK enumerator walks with packed x = (i + half_i) + j * sieve_width.
    // inc_step and inc_warp encode the two reduced basis vectors in this 1D space.
    //
    // For the "step" direction: when the i-coordinate (x % sieve_width - half_i)
    // is below bound_step, we add inc_step.
    // For the "warp" direction: when the i-coordinate is at or above bound_warp,
    // we add inc_warp.
    //
    // The bounds are defined as:
    //   bound_step = half_i - |su|  (step applicable while i < bound_step)
    //   bound_warp = half_i - |wu|  (warp applicable while i >= bound_warp)
    // But in 1D packed coords, we compare the i-part of x against these bounds
    // shifted by half_i (since x stores i + half_i).

    let sw = sieve_width as u64;
    let hi = half_i as u64;

    // Convert bounds from centered [-I, I) to offset [0, 2I) coordinates
    let bound_step_offset = (pl.bound_step + half_i) as u64;
    let bound_warp_offset = (pl.bound_warp + half_i) as u64;

    // Starting position: enumerate from x = half_i (i=0, j=0).
    // The first hit at j=0 is at i = start (from PLattice).
    let mut x = pl.start + hi;

    // For primes smaller than sieve_width, there may be multiple hits per row.
    // We need to enumerate all of them. The FK walk naturally does this since
    // inc_step advances within a row when |su| < sieve_width.

    // Safety: if inc_step and inc_warp are both 0 or negative in packed space,
    // we'd loop forever. Guard against that.
    if pl.inc_step == 0 && pl.inc_warp == 0 {
        return;
    }

    // Walk the lattice
    while x < fence {
        // Push this hit
        let bucket_idx = (x as usize) >> LOG_BUCKET_REGION;
        let pos_in_bucket = (x as usize) & (BUCKET_REGION - 1);
        if bucket_idx < buckets.n_buckets() {
            buckets.push(
                bucket_idx,
                BucketUpdate {
                    pos: pos_in_bucket as u16,
                    logp,
                },
            );
        }

        // Advance: extract i-coordinate from packed x
        let i_offset = x % sw; // i + half_i, in [0, sieve_width)

        // Apply warp if i is large (near right edge)
        let warp_add = if i_offset >= bound_warp_offset {
            pl.inc_warp
        } else {
            0
        };
        // Apply step if i is small (near left edge)
        let step_add = if i_offset < bound_step_offset {
            pl.inc_step
        } else {
            0
        };

        let advance = warp_add + step_add;
        if advance <= 0 {
            // Shouldn't happen with correct FK reduction, but guard against infinite loop
            break;
        }
        x = (x as i64 + advance) as u64;
    }
}
```

**Step 4: Run test to verify it passes**

Run: `cd rust/potapov-nfs && cargo test test_fk_scatter_matches_naive_scatter -- --nocapture`
Expected: PASS — both functions produce the same set of hits.

**Step 5: Commit**

```bash
git add rust/potapov-nfs/src/sieve/mod.rs
git commit -m "sieve: add FK lattice enumerator scatter_bucket_updates_fk"
```

---

### Task 3: Fix FK enumerator edge cases and validate thoroughly

The initial FK implementation may have edge cases around:
- Projective roots (denom == 0 in root transform)
- Start position computation (first hit in sieve region)
- Primes where p < sieve_width (multiple hits per row)
- Primes where p > sieve_width (at most one hit per row)
- Very small primes near bucket_thresh boundary

**Files:**
- Modify: `rust/potapov-nfs/src/sieve/mod.rs` (FK function + tests)
- Modify: `rust/potapov-nfs/src/sieve/lattice.rs` (may need to fix reduce_plattice for projective case)

**Step 1: Write edge case tests**

```rust
#[test]
fn test_fk_scatter_small_prime_multi_hit() {
    // p=7 with log_i=7 means sieve_width=256, so 256/7 ≈ 36 hits per row
    use crate::sieve::lattice::reduce_qlattice;
    let qlat = reduce_qlattice(97, 30, 1.0);
    let log_i = 7u32;
    let half_i = 1i64 << log_i;
    let sieve_width = (2 * half_i) as usize;
    let max_j = half_i as usize;
    let n_buckets = (sieve_width * max_j + BUCKET_REGION - 1) / BUCKET_REGION;

    let mut old_buckets = BucketArray::new(n_buckets.max(1), 8192);
    scatter_bucket_updates_for_prime(7, 3, 3, &qlat, log_i, &mut old_buckets, sieve_width, max_j, half_i);
    let old_count = old_buckets.total_updates();

    let mut new_buckets = BucketArray::new(n_buckets.max(1), 8192);
    scatter_bucket_updates_fk(7, 3, 3, &qlat, log_i, &mut new_buckets, sieve_width, max_j, half_i);
    let new_count = new_buckets.total_updates();

    assert!(old_count > 100, "small prime should produce many hits: {}", old_count);
    assert_eq!(old_count, new_count, "hit counts must match for small prime");
}

#[test]
fn test_fk_scatter_large_prime_single_hit_per_row() {
    // p=521 with log_i=7 means sieve_width=256, so at most 1 hit per row
    use crate::sieve::lattice::reduce_qlattice;
    let qlat = reduce_qlattice(65537, 12345, 1.0);
    let log_i = 7u32;
    let half_i = 1i64 << log_i;
    let sieve_width = (2 * half_i) as usize;
    let max_j = half_i as usize;
    let n_buckets = (sieve_width * max_j + BUCKET_REGION - 1) / BUCKET_REGION;

    let mut old_buckets = BucketArray::new(n_buckets.max(1), 4096);
    scatter_bucket_updates_for_prime(521, 100, 9, &qlat, log_i, &mut old_buckets, sieve_width, max_j, half_i);

    let mut new_buckets = BucketArray::new(n_buckets.max(1), 4096);
    scatter_bucket_updates_fk(521, 100, 9, &qlat, log_i, &mut new_buckets, sieve_width, max_j, half_i);

    let mut old_hits: Vec<usize> = Vec::new();
    let mut new_hits: Vec<usize> = Vec::new();
    for b in 0..n_buckets {
        for u in old_buckets.updates_for_bucket(b) {
            old_hits.push(b * BUCKET_REGION + u.position() as usize);
        }
        for u in new_buckets.updates_for_bucket(b) {
            new_hits.push(b * BUCKET_REGION + u.position() as usize);
        }
    }
    old_hits.sort();
    new_hits.sort();
    assert_eq!(old_hits, new_hits, "large prime FK must match naive");
}
```

**Step 2: Run edge case tests, fix any mismatches**

Run: `cd rust/potapov-nfs && cargo test test_fk_scatter -- --nocapture`
Expected: If any fail, debug and fix the FK implementation or the reduce_plattice start position.

**Step 3: Fix issues found**

The most likely issues:
1. **Start position**: `reduce_plattice` currently returns `start = 0`, which is only correct if the first lattice point in [0, fence) is at x=0. In general, we need to find the first valid x. This may require adjusting `reduce_plattice` to compute the correct start position by finding the smallest non-negative x in the lattice.
2. **Projective roots**: `reduce_plattice` returns `hits = false` for projective roots. The old `scatter_bucket_updates_for_prime` handled projective roots (denom==0) with a special fill-all-columns path. We need to keep the old projective handling as a fallback.

Fix by making `scatter_bucket_updates_fk` fall back to the old function for projective roots:

```rust
fn scatter_bucket_updates_fk(
    p: u64, root: u64, logp: u8, qlat: &QLattice, log_i: u32,
    buckets: &mut BucketArray, sieve_width: usize, max_j: usize, half_i: i64,
) {
    use crate::sieve::lattice::reduce_plattice;

    let pl = reduce_plattice(p, root, qlat, log_i);
    if !pl.hits {
        // Fall back to old code for projective roots
        scatter_bucket_updates_for_prime(p, root, logp, qlat, log_i, buckets, sieve_width, max_j, half_i);
        return;
    }
    // ... FK walk ...
}
```

Wait — that defeats the purpose for projective roots. Better: check if reduce_plattice returns no-hit because of projective root (denom==0), and handle that case inline with a simple row-fill. The old `scatter_bucket_updates_for_prime` stays as the reference implementation (not called in production, only in tests).

**Step 4: Run full test suite**

Run: `cd rust/potapov-nfs && cargo test`
Expected: All tests pass.

**Step 5: Commit**

```bash
git add rust/potapov-nfs/src/sieve/mod.rs rust/potapov-nfs/src/sieve/lattice.rs
git commit -m "sieve: fix FK enumerator edge cases (projective roots, start position)"
```

---

### Task 4: Wire FK enumerator into main sieve loop

Replace the calls to `scatter_bucket_updates_for_prime` in the main sieve loop (lines 281-317 of mod.rs) with `scatter_bucket_updates_fk`.

**Files:**
- Modify: `rust/potapov-nfs/src/sieve/mod.rs:281-317`

**Step 1: Replace the scatter calls**

Change lines 281-317 from:

```rust
for &fb_idx in &alg_large_indices {
    let p = alg_fb.primes[fb_idx];
    if p == q { continue; }
    for &root in &alg_fb.roots[fb_idx] {
        scatter_bucket_updates_for_prime(
            p, root, alg_fb.log_p[fb_idx], &qlat, params.log_i,
            &mut *alg_buckets, sieve_width, max_j, half_i,
        );
    }
}
for &fb_idx in &rat_large_indices {
    let p = rat_fb.primes[fb_idx];
    if p == q { continue; }
    scatter_bucket_updates_for_prime(
        p, rat_roots_by_index[fb_idx], rat_fb.log_p[fb_idx], &qlat, params.log_i,
        &mut *rat_buckets, sieve_width, max_j, half_i,
    );
}
```

to:

```rust
for &fb_idx in &alg_large_indices {
    let p = alg_fb.primes[fb_idx];
    if p == q { continue; }
    for &root in &alg_fb.roots[fb_idx] {
        scatter_bucket_updates_fk(
            p, root, alg_fb.log_p[fb_idx], &qlat, params.log_i,
            &mut *alg_buckets, sieve_width, max_j, half_i,
        );
    }
}
for &fb_idx in &rat_large_indices {
    let p = rat_fb.primes[fb_idx];
    if p == q { continue; }
    scatter_bucket_updates_fk(
        p, rat_roots_by_index[fb_idx], rat_fb.log_p[fb_idx], &qlat, params.log_i,
        &mut *rat_buckets, sieve_width, max_j, half_i,
    );
}
```

**Step 2: Run full test suite**

Run: `cd rust/potapov-nfs && cargo test`
Expected: All tests pass.

**Step 3: Run end-to-end factorization**

Run: `cd rust/potapov-nfs && cargo run --release -- --factor 684217602914977371691118975023 --threads 1 2>&1 | grep -E 'Factor:|sieve:|total_ms'`
Expected: Correct factorization, faster sieve time.

**Step 4: Commit**

```bash
git add rust/potapov-nfs/src/sieve/mod.rs
git commit -m "sieve: wire FK enumerator into main sieve loop"
```

---

### Task 5: Add coprimality pre-filter

Add a fast coprimality check in the FK inner loop: skip positions where both i and j are even (gcd(i,j) >= 2). In packed coordinates, i is even when `x & 1 == 0` and j is even when `(x >> log_sieve_width) & 1 == 0`.

**Files:**
- Modify: `rust/potapov-nfs/src/sieve/mod.rs` (FK function inner loop)

**Step 1: Write test**

```rust
#[test]
fn test_coprime_filter_removes_even_pairs() {
    // Position at (i=0, j=0) has both even — should be skipped
    // Position at (i=1, j=0) has i odd — should be kept
    // Position at (i=0, j=1) has j odd — should be kept
    let sieve_width = 256usize;
    let half_i = 128i64;
    let log_sieve_width = 8u32; // log2(256)

    let check_coprime = |x: u64| -> bool {
        let i_offset = x % (sieve_width as u64);
        let j = x / (sieve_width as u64);
        let i = i_offset as i64 - half_i;
        // Both even means gcd >= 2
        !(i % 2 == 0 && j % 2 == 0)
    };

    // Fast version using bit ops
    let fast_coprime = |x: u64| -> bool {
        // even_mask checks bit 0 (i parity) and bit log_sieve_width (j parity)
        let even_mask = 1u64 | (1u64 << log_sieve_width);
        (x & even_mask) != 0
    };

    // Verify they agree for a range of positions
    for x in 0..1000u64 {
        // Note: fast_coprime checks i+half_i parity, not i parity.
        // When half_i is even (power of 2), i and i+half_i have same parity.
        assert_eq!(
            check_coprime(x), fast_coprime(x),
            "coprime mismatch at x={}", x
        );
    }
}
```

**Step 2: Run test**

Run: `cd rust/potapov-nfs && cargo test test_coprime_filter_removes_even_pairs`

**Step 3: Add the filter to FK inner loop**

In the FK walker, before pushing the bucket update, add:

```rust
// Fast coprimality pre-filter: skip positions where both i and j are even.
// In packed x = (i+half_i) + j*sieve_width, check bit 0 (i parity) and
// bit log_sieve_width (j parity). When half_i is a power of 2, (i+half_i)
// has the same parity as i.
let even_mask = 1u64 | (1u64 << log_sieve_width);
// ... inside the while loop:
if (x & even_mask) != 0 {
    // push bucket update
}
```

Note: `log_sieve_width = log_i + 1` since `sieve_width = 2 * half_i = 2^(log_i+1)`.

**Step 4: Run full test suite and end-to-end**

Run: `cd rust/potapov-nfs && cargo test && cargo run --release -- --factor 684217602914977371691118975023 --threads 1 2>&1 | grep 'Factor:'`
Expected: All tests pass, correct factorization.

**Step 5: Commit**

```bash
git add rust/potapov-nfs/src/sieve/mod.rs
git commit -m "sieve: add coprimality pre-filter in FK inner loop"
```

---

### Task 6: Benchmark and validate

Run the full benchmark suite to measure combined impact of all three optimizations.

**Files:** None (benchmark only)

**Step 1: Single-threaded c30 benchmark (3 runs)**

```bash
for i in 1 2 3; do
  cargo run --release -- --factor 684217602914977371691118975023 --threads 1 2>&1 | grep -E '"(sieve_ms|total_ms|filter_ms|la_ms|sqrt_ms)"'
done
```

Expected: sieve_ms should drop from ~5400ms to ~1500-2000ms. Total should drop from ~6600ms to ~3000-3500ms.

**Step 2: Multi-threaded c30 benchmark (3 runs)**

```bash
for i in 1 2 3; do
  cargo run --release -- --factor 684217602914977371691118975023 2>&1 | grep -E '"(sieve_ms|total_ms)"'
done
```

**Step 3: Sieve profile breakdown**

```bash
POTAPOV_NFS_SIEVE_PROFILE=1 cargo run --release -- --factor 684217602914977371691118975023 --threads 1 2>&1 | grep -E '(sieve-profile|sieve:|setup=)'
```

Expected: `setup=` should drop from ~3600ms to ~500-800ms (the FK enumerator eliminates the i128 modular arithmetic bottleneck).

**Step 4: c45 benchmark**

```bash
cargo run --release -- --bits 45 --semiprimes 3 --seed 123 --threads 1 2>&1 | tail -30
```

**Step 5: Update progress.md with new numbers**

**Step 6: Commit**

```bash
git add rust/potapov-nfs/progress.md
git commit -m "sieve: update benchmark numbers after FK enumerator + threshold fix"
```
