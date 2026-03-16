# Multi-Level Bucket Lattice Sieve — Design Spec

## Context

c45 (148-bit semiprime) sieve takes 16.2s, dominated by FK scatter setup (10.8s, 63%).
The scatter sieve's O(n_fb x n_sq) scaling has a proven 9.3s floor.
Target: 1000ms ST / 500ms MT. c30 must not regress (currently 925ms ST, 3/3 factored).

c30 dominance was achieved through parameter tuning + tight Rust + aggressive parallelism.
Same playbook applies to c45, but requires architectural change to enable larger sieve areas.

## Problem

The scatter sieve processes each FB prime independently per SQ, writing directly to the
sieve array (random access pattern). With I=9 (area=512K), each SQ yields ~9 relations,
requiring 3835 SQs. Per-entry cost is 250ns (5x slower than CADO's 50ns).

Increasing I to 10+ would reduce SQ count but region scan scales linearly with area,
negating the benefit. The fix: cache-friendly bucket sieve that makes large areas practical.

## Design

### New Function: `bucket_sieve_specialq`

A new sieve entry point alongside existing `sieve_specialq` and `line_sieve_specialq`.
Auto-selected when `degree >= 4` or `bits >= 130`. Produces identical `SieveResult`.

### Architecture: 3-Phase Bucket Sieve

**Phase 1 — Bucket Fill-In** (per-SQ, parallelized across SQs)

For each special-q (q, r):
1. Reduce q-lattice: `reduce_qlattice(q, r, 1.0)` (existing)
2. Batch transform roots for all large FB primes (existing `batch_transform_roots`)
3. For each large prime entry (p, r', logp):
   - Partial-GCD reduction (existing, ~80ns)
   - FK walk: enumerate all hit positions in the sieve area
   - For each hit position x: `buckets[x >> LOG_BUCKET_REGION].push(x & MASK, logp)`
   - Sequential bucket writes (cache-friendly)

**Phase 2 — Region Processing** (per-SQ, sequential across regions)

For each bucket region (64KB, fits L1 cache):
1. Initialize norms (block-16 approximation, existing code)
2. Apply small sieve (primes < bucket_thresh, existing NEON for p<=7)
3. Apply bucket updates: iterate bucket, `sieve[offset] -= logp` (sequential reads)
4. Scan survivors: NEON 16-byte comparison (existing)
5. Coprimality check + cofactorization (existing)

**Phase 3 — Relation Collection** (identical to scatter sieve)

Build `Relation` structs from survivors, accumulate into `SieveResult`.

### Parameters for c45

| Parameter | Current (I=9) | Bucket (I=11) | Bucket (I=12) |
|-----------|---------------|---------------|---------------|
| sieve_width | 1024 | 4096 | 8192 |
| max_j | 512 | 2048 | 4096 |
| area | 512K | 8M | 32M |
| n_buckets | 8 | 128 | 512 |
| est_rels/SQ | 9 | ~40 | ~140 |
| SQs needed | 3835 | ~850 | ~240 |
| FK entries/SQ | 8400 | 8400 | 8400 |

### Data Structures

```rust
// Compact bucket update: 4 bytes (position + logp)
#[repr(C, packed)]
struct BucketHit {
    pos: u16,    // position within BUCKET_REGION (0..65535)
    logp: u8,    // log2(p) contribution
    _pad: u8,    // alignment padding (or slice index for resieving)
}

// Bucket array: pre-allocated per region
struct BucketSieveArray {
    data: Vec<BucketHit>,        // flat storage for all updates
    write_pos: Vec<usize>,       // per-bucket write cursor
    starts: Vec<usize>,          // per-bucket start offset
    n_buckets: usize,
}
```

Note: We already have `BucketArray` and `BucketUpdate` (3 bytes: pos:u16, logp:u8) in
`sieve/bucket.rs`. The existing infrastructure is reused directly.

### Sieve Area Scaling Strategy

Start with I=11 (conservative, 16x area increase), validate correctness and relation yield.
Then scale to I=12 if yield matches predictions.

The FK walk inner loop is unchanged from the scatter sieve — same partial-GCD, same walk
algorithm. The only difference is WHERE hits are written (bucket append vs direct sieve write).

### Integration

```rust
// In pipeline.rs, replace sieve call:
let sieve_result = if params.log_i >= 11 {
    crate::sieve::bucket_sieve_specialq(...)
} else {
    crate::sieve::sieve_specialq(...)  // scatter sieve for c30
};
```

### Auto-Selection

- c30 (degree 3, I=9): scatter sieve (proven, 925ms)
- c45 (degree 4, I=11+): bucket sieve (new)
- Controlled by `POTAPOV_NFS_SIEVE_MODE=bucket|scatter|auto` env var

### Critical Files

- `rust/potapov-nfs/src/sieve/mod.rs` — add `bucket_sieve_specialq` function
- `rust/potapov-nfs/src/sieve/bucket.rs` — reuse existing BucketArray
- `rust/potapov-nfs/src/params.rs` — update c45 params to I=11
- `rust/potapov-nfs/src/pipeline.rs` — auto-select sieve mode

### Verification

1. Unit tests: bucket sieve finds relations on small test cases
2. Correctness: identical relation format (a, b, factors, lp_keys)
3. c45 benchmark: 3 semiprimes seed 42, all must factor, sieve < 5s
4. c30 regression: 3 semiprimes seed 42, all must factor, mean < 1000ms
5. MT benchmark: c45 with all cores, target < 2s total

### Risk Mitigation

- Scatter sieve preserved unchanged for c30 (no regression possible)
- Bucket sieve opt-in via I threshold (only activates for I >= 11)
- Fallback: `POTAPOV_NFS_SIEVE_MODE=scatter` forces old path
