# Multi-Level Bucket Lattice Sieve Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a bucket sieve with I=11 (16x larger sieve area) to reduce c45 SQ count from 3835 to ~850, targeting <5s sieve then <1s with tuning.

**Architecture:** Reuse the existing scatter sieve's FK walk + bucket infrastructure but with larger sieve area (I=11). The `bucket_sieve_specialq` function is a parameter-adjusted copy of `sieve_specialq` with c45-specific optimizations: higher bucket_thresh (matching larger I), tuned batch sizes, and sieve_width-aware norm initialization. Auto-selected for degree≥4 via pipeline toggle.

**Tech Stack:** Rust (existing crate), rayon for parallelism, existing NEON SIMD, existing BucketArray/BucketUpdate.

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `rust/potapov-nfs/src/params.rs` | Modify | Add `c45_bucket()` params with log_i=11 |
| `rust/potapov-nfs/src/sieve/mod.rs` | Modify | Add `bucket_sieve_specialq` function |
| `rust/potapov-nfs/src/pipeline.rs` | Modify | Auto-select bucket sieve for c45 |
| `rust/potapov-nfs/src/sieve/bucket.rs` | Modify | Scale BucketArray for larger n_buckets |

---

### Task 1: Add c45 Bucket Parameters

**Files:**
- Modify: `rust/potapov-nfs/src/params.rs`

- [ ] **Step 1: Add `c45_bucket()` parameter set**

Add a new parameter preset for bucket sieve c45 with log_i=11:

```rust
/// c45 parameters optimized for bucket sieve (larger sieve area).
pub fn c45_bucket() -> Self {
    Self {
        name: "c45_bucket",
        degree: 4,
        lim0: 40_000,
        lim1: 45_000,
        lpb0: 20,
        lpb1: 21,
        mfb0: 28,
        mfb1: 30,
        sieve_mfb0: 28,
        sieve_mfb1: 30,
        log_i: 11,              // I=11: sieve_width=4096, max_j=2048, area=8M
        qmin: 35_000,
        qrange: 3_000,
        rels_wanted: 45_000,
    }
}
```

- [ ] **Step 2: Run tests**

Run: `cargo test --release --lib -- params`
Expected: PASS (new function doesn't break existing)

- [ ] **Step 3: Commit**

```bash
git add rust/potapov-nfs/src/params.rs
git commit -m "params: add c45_bucket preset with log_i=11 for bucket sieve"
```

---

### Task 2: Implement `bucket_sieve_specialq`

**Files:**
- Modify: `rust/potapov-nfs/src/sieve/mod.rs`

This is the core task. The function is structurally identical to `sieve_specialq` but with:
1. Higher bucket_thresh (sieve_width instead of 256) — more primes use FK scatter
2. Larger sieve area (8M positions vs 512K)
3. Tuned batch precomputation for larger area

- [ ] **Step 1: Add function signature and setup**

Add `pub fn bucket_sieve_specialq(...)` with identical signature to `sieve_specialq`.
Copy the setup section (lines 64-228 of `sieve_specialq`) but with:
- `bucket_thresh = 256u64` (keep same — FK scatter handles primes >256 efficiently)
- Verify `n_buckets` scales correctly: for I=11, area=8M, n_buckets = 8M/65536 = 128

```rust
pub fn bucket_sieve_specialq(
    f_coeffs: &[i64],
    _m: u64,
    rat_fb: &FactorBase,
    alg_fb: &FactorBase,
    params: &NfsParams,
    q_start: u64,
    q_range: u64,
    max_relations: Option<usize>,
    sq_root_cache: Option<&HashMap<u64, Vec<u64>>>,
    g0_param: i64,
    g1_param: i64,
) -> SieveResult {
    // Identical to sieve_specialq setup
    // Key difference: params.log_i=11 → half_i=2048, sieve_width=4096, max_j=2048
    // n_buckets = 4096*2048 / 65536 = 128
    // ... (copy from sieve_specialq, the setup is parameter-driven)
}
```

- [ ] **Step 2: Copy the chunk processing loop**

Copy the entire chunk loop from `sieve_specialq` (lines 230-690). The batch_transform_roots, FK scatter, and region processing are all parameter-driven — they automatically adapt to larger I.

Key verification: with I=11 and 128 buckets, the BucketArray allocation is `128 * updates_per_bucket`. The `updates_per_bucket` estimate scales with area/n_fb_primes, so it should auto-adjust.

- [ ] **Step 3: Verify BucketArray handles 128 buckets**

Check `bucket.rs` — `BucketArray::new(n_buckets, updates_per_bucket)` allocates
`n_buckets * updates_per_bucket` total entries. With n_buckets=128 and updates_per_bucket=~4096,
total = 524K entries × 3 bytes = 1.5MB per side per thread. Acceptable.

- [ ] **Step 4: Run tests**

Run: `cargo test --release --lib`
Expected: PASS (new function not yet called from pipeline)

- [ ] **Step 5: Commit**

```bash
git add rust/potapov-nfs/src/sieve/mod.rs
git commit -m "sieve: add bucket_sieve_specialq for large-area c45 sieving"
```

---

### Task 3: Wire into Pipeline with Auto-Selection

**Files:**
- Modify: `rust/potapov-nfs/src/pipeline.rs`

- [ ] **Step 1: Add sieve mode selection logic**

Near the existing `use_line_sieve` toggle (line 874), add bucket sieve selection:

```rust
let sieve_mode = std::env::var("POTAPOV_NFS_SIEVE_MODE")
    .unwrap_or_else(|_| "auto".to_string());

let use_bucket_sieve = match sieve_mode.as_str() {
    "bucket" => true,
    "scatter" => false,
    "line" => false,  // handled by existing use_line_sieve
    _ => params.degree >= 4 && n_bits >= 130,  // auto: bucket for c45+
};

if use_bucket_sieve {
    // Use c45_bucket params (log_i=11) instead of default c45 params
    let bucket_params = NfsParams::c45_bucket();
    // Override relevant params while keeping polynomial-dependent values
    params.log_i = bucket_params.log_i;
    eprintln!("  sieve: using bucket sieve (log_i={})", params.log_i);
}
```

- [ ] **Step 2: Add bucket sieve call in adaptive loop**

Replace the sieve invocation (lines 1142-1155) with a 3-way dispatch:

```rust
let sieve_result = if use_bucket_sieve {
    crate::sieve::bucket_sieve_specialq(
        &f_coeffs_i64, m, &rat_fb, &alg_fb, &params,
        q_start, params.qrange,
        Some(remaining_to_target), Some(&sq_root_cache),
        g0_i64, g1_i64,
    )
} else if use_line_sieve {
    crate::sieve::line_sieve_specialq(...)
} else {
    crate::sieve::sieve_specialq(...)
};
```

- [ ] **Step 3: Run tests**

Run: `cargo test --release --lib`
Expected: PASS

- [ ] **Step 4: Benchmark c45 with bucket sieve**

```bash
POTAPOV_NFS_SIEVE_PROFILE=1 cargo run --release -- --bits 148 --semiprimes 1 --seed 42 --threads 1
```
Expected: Bucket sieve auto-selected, sieve time < 10s (initial, untuned)

- [ ] **Step 5: Benchmark c30 regression**

```bash
cargo run --release -- --bits 100 --semiprimes 3 --seed 42 --threads 1
```
Expected: 3/3 factored, mean < 1000ms (scatter sieve auto-selected)

- [ ] **Step 6: Commit**

```bash
git add rust/potapov-nfs/src/pipeline.rs
git commit -m "pipeline: auto-select bucket sieve for c45 (degree>=4, bits>=130)"
```

---

### Task 4: Tune Parameters for I=11

**Files:**
- Modify: `rust/potapov-nfs/src/params.rs`

- [ ] **Step 1: Sweep I=10 vs I=11 with bucket sieve**

```bash
POTAPOV_NFS_SIEVE_PROFILE=1 cargo run --release -- --bits 148 --semiprimes 1 --seed 42 --threads 1
```

Record: sieve time, rels/SQ, n_SQs, setup time, scan time.

- [ ] **Step 2: Adjust bucket_thresh and qrange based on results**

If I=11 produces significantly more rels/SQ, the qrange can be reduced (fewer SQs needed).
If region scan is too slow, increase bucket_thresh to route more primes to FK scatter.

- [ ] **Step 3: Try I=12 if I=11 is promising**

```bash
# Override log_i via env var if supported, or modify params temporarily
```

- [ ] **Step 4: Update c45_bucket parameters with optimal values**

- [ ] **Step 5: Run 3-semiprime reliability check**

```bash
cargo run --release -- --bits 148 --semiprimes 3 --seed 42 --threads 1
```
Expected: 3/3 factored, total < 5s

- [ ] **Step 6: Commit**

```bash
git add rust/potapov-nfs/src/params.rs
git commit -m "params: tune c45_bucket for optimal I and relation yield"
```

---

### Task 5: MT Optimization and Final Benchmark

**Files:**
- No new files (tuning only)

- [ ] **Step 1: MT benchmark**

```bash
POTAPOV_NFS_SIEVE_PROFILE=1 cargo run --release -- --bits 148 --semiprimes 3 --seed 42
```
Expected: Uses all available cores, total < 3s MT

- [ ] **Step 2: Profile MT scaling**

Check if rayon's par_iter distributes SQs evenly across cores.
If bottleneck is serial section (batch_transform_roots), investigate parallelizing it.

- [ ] **Step 3: 10-run reliability benchmark**

```bash
cargo run --release -- --bits 148 --semiprimes 10 --seed 42 --threads 1
```
Expected: ≥9/10 factored, mean sieve < 5s

- [ ] **Step 4: c30/c35/c40 regression**

```bash
cargo run --release -- --bits 100 --semiprimes 5 --seed 42 --threads 1
cargo run --release -- --bits 120 --semiprimes 3 --seed 42 --threads 1
cargo run --release -- --bits 135 --semiprimes 3 --seed 42 --threads 1
```
Expected: All factored, c30 mean < 1000ms

- [ ] **Step 5: Document results and commit**

```bash
git commit -m "bench: c45 bucket sieve benchmark results"
```

---

## Verification Summary

| Check | Command | Expected |
|-------|---------|----------|
| Unit tests | `cargo test --release --lib` | 173+ pass |
| c45 correctness | `--bits 148 --semiprimes 3 --seed 42 --threads 1` | 3/3 factored |
| c45 sieve speed | (from timing output) | < 5s sieve |
| c30 regression | `--bits 100 --semiprimes 3 --seed 42 --threads 1` | 3/3, mean < 1000ms |
| MT scaling | `--bits 148 --semiprimes 3 --seed 42` | < 3s total |
