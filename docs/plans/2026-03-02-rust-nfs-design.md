# Rust-NFS: Production NFS Implementation to Beat CADO-NFS

**Date**: 2026-03-02
**Status**: Approved
**Target**: Factor 96-128 bit semiprimes faster than CADO-NFS

## Goal

Build a production-quality NFS relation sieve in Rust that matches CADO-NFS's
algorithmic sophistication while leveraging Rust's performance advantages.
The implementation covers the full NFS pipeline: polynomial selection, sieve,
cofactorization, filtering, linear algebra, and square root.

**Performance target**: Exceed CADO-NFS's ~50K rels/sec at 128-bit.

## Why Rust Can Beat CADO

1. **No Python/process overhead**: CADO coordinates phases via Python, spawning
   separate C processes. We run everything in-process with zero IPC cost.
2. **Zero-copy data flow**: Relations stay in memory from sieve through LA.
   CADO serializes to disk between every phase.
3. **Rayon work-stealing**: Trivial parallelism vs CADO's manual OpenMP with
   explicit thread pool management.
4. **Apple Silicon tuning**: 192KB L1, 12MB L2 cache hierarchy. CADO's
   LOG_BUCKET_REGION=16 (64KB) was tuned for x86 L1 (32-48KB). We can use
   larger bucket regions.
5. **No aliasing**: Rust's ownership model proves no aliasing, enabling more
   aggressive LLVM vectorization.
6. **PGO**: cargo-pgo for profile-guided optimization.
7. **Monomorphization**: Generic code compiles to specialized versions per type.

## Architecture

New crate `rust-nfs` under `rust/rust-nfs/`, depending on `gnfs` for LA/sqrt.

```
rust-nfs/
├── Cargo.toml
├── src/
│   ├── lib.rs           # Public API: factor(n) -> Vec<Integer>
│   ├── main.rs          # CLI: --factor N, --benchmark, --bits 96,112,128
│   ├── params.rs         # CADO-exact parameter tables
│   ├── factorbase.rs     # Factor base + polynomial root finding
│   ├── sieve/
│   │   ├── mod.rs        # Special-q loop orchestration
│   │   ├── bucket.rs     # Bucket array: scatter/gather
│   │   ├── norm.rs       # u8 log-norm initialization
│   │   ├── small.rs      # Small sieve (line sieve for p < I)
│   │   ├── lattice.rs    # Q-lattice + P-lattice reduction
│   │   └── region.rs     # Per-bucket-region processing
│   ├── cofactor/
│   │   ├── mod.rs        # Cofactorization pipeline
│   │   ├── trialdiv.rs   # Montgomery-form trial division
│   │   ├── pm1.rs        # Pollard P-1
│   │   ├── pp1.rs        # Williams P+1
│   │   └── ecm.rs        # ECM with u64 Montgomery arithmetic
│   ├── filter.rs         # Singleton + clique removal
│   ├── relation.rs       # Relation types, compatible with gnfs
│   └── pipeline.rs       # Full NFS orchestration
```

## Phase 1: Polynomial Selection (reuse gnfs)

For 96-128 bit, base-m polynomial selection is standard. Reuse
`gnfs::polyselect::select_base_m_variant()`.

- Degree 3 for <= 120 bits, degree 4 for 121-166 bits
- m = floor(N^(1/(d+1)))
- f(x) = c_d*x^d + ... + c_0 where N = f(m)

## Phase 2: Factor Base Construction

Generate primes up to lim0/lim1 via sieve of Eratosthenes.
For each prime p: find all roots r of f(x) ≡ 0 (mod p) via Tonelli-Shanks.
Precompute:
- `log_p: Vec<u8>` — quantized log2(p) * scale
- `roots: Vec<Vec<u64>>` — algebraic roots per prime
- `inv_p: Vec<u64>` — Montgomery inverse for trial division

## Phase 3: Sieve (the main event)

### Special-Q Loop

For each special-q prime q in [qmin, qmax], for each root r of f(x) mod q:

```
1. q_lattice = skew_gauss_reduce(q, r, skewness)
   → basis vectors (a0, b0), (a1, b1)

2. For each FB prime p with root R:
   p_lattice = franke_kleinjung_reduce(p, R, q_lattice, I)
   → walk pattern (inc_step, inc_warp, bounds)

3. Scatter: enumerate FK walk, push bucket updates
   For each lattice point (i, j) in sieve region:
     bucket_idx = j * (2*I) + i + I  →  bucket_region = pos >> LOG_BUCKET_REGION
     updates[bucket_region].push(BucketUpdate { offset: pos & MASK })

4. Per bucket region (L1-cache-resident):
   a. init_norms(S, region) — u8 log|F(i,j)| approximation
   b. apply_buckets(S, updates) — S[pos] -= log_p (saturated)
   c. small_sieve(S, small_primes) — line sieve p < I
   d. survivors = scan(S, threshold) — S[pos] <= bound
   e. cofactorize(survivors) → relations
```

### Bucket Sieve Data Structure

```rust
struct BucketUpdate {
    pos: u16,  // position within 64KB bucket region
}
// 2 bytes per update, maximally cache-friendly

struct BucketArray {
    data: Vec<u8>,           // contiguous allocation
    write_ptrs: Vec<usize>,  // per-bucket append pointer
    starts: Vec<usize>,      // per-bucket start
    n_buckets: usize,
}
```

For 96-128 bit with I=9: sieve area = 2^10 × 256 = 262K positions = 4 bucket
regions at LOG_BUCKET_REGION=16. For Apple Silicon with 192KB L1, we can use
LOG_BUCKET_REGION=17 (128KB) = 2 bucket regions.

### Franke-Kleinjung P-Lattice Walk

For each FB prime p with root R in the q-lattice:

```
Initial basis: (p, 0) and (R_transformed, 1)
Reduce via partial GCD until:
  -I < i0 <= 0 <= i1 < I
  i1 + |i0| >= I

Walk: starting from first lattice point,
  if pos % I < bound_step: pos += inc_step
  else: pos += inc_warp
```

This enumerates all hits of prime p in the sieve region in increasing
memory order — perfect for the bucket scatter.

### Q-Lattice Reduction (Skew Gaussian)

```
Input: q, r (f(r) ≡ 0 mod q), skewness S
Basis: v0 = (q, 0), v1 = (r, 1)
Quadratic form: Q(a,b) = a^2 + S^2 * b^2

Repeat Euclid-like steps until both vectors are short:
  if Q(v0) > Q(v1): swap
  v1 = v1 - round(dot(v0,v1)/Q(v0)) * v0

Output: a0, b0, a1, b1 (reduced basis vectors)
```

### Norm Initialization

For the rational side (degree 1): F_rat(i,j) = g1*i + g0*j in q-lattice coords.
This is linear, so we can compute the exact crossing points where the log value
changes and fill with memset between them (CADO's "smart" fill).

For the algebraic side (degree d): use piecewise-linear approximation of
log|F_alg(i,j)| with error bounded by 1 bit. Store as u8 scaled values.

```rust
fn init_norm_rat(sieve: &mut [u8], g0: f64, g1: f64, j: u32, scale: f64) {
    // For each j-line, compute breakpoints and memset between them
    let root = -g0 * j as f64 / g1;
    // Decreasing phase (before root): compute transition points
    // Increasing phase (after root): same in reverse
}
```

## Phase 4: Cofactorization

### Pipeline (matching CADO's facul_strategies)

```rust
pub fn cofactorize(cofactor: u64, lpb: u32, mfb: u32, lim: u64) -> CofactResult {
    let l = 1u64 << lpb;
    let b = lim;

    // Quick checks
    if cofactor == 1 { return FullSmooth; }
    if cofactor <= l { return OneLargePrime(cofactor); }
    if cofactor.bits() > mfb { return NotSmooth; }

    // Dead-zone check: if cofactor is prime and > l, reject
    if is_probable_prime(cofactor) { return NotSmooth; }

    // P-1 (B1=315, B2=2205)
    if let Some(f) = pm1(cofactor, 315, 2205) {
        return check_factors(f, cofactor/f, l);
    }
    // P+1 (B1=525, B2=3255)
    if let Some(f) = pp1(cofactor, 525, 3255) {
        return check_factors(f, cofactor/f, l);
    }
    // ECM chain (growing B1)
    for (b1, b2) in ecm_bounds(lpb) {
        if let Some(f) = ecm_one_curve(cofactor, b1, b2) {
            return check_factors(f, cofactor/f, l);
        }
    }
    NotSmooth
}
```

### Montgomery-Form Trial Division

Precompute for each prime p: `pinv = modular_inverse(p, 2^64)`, `plim = (2^64-1)/p`.
Test: `n * pinv <= plim` iff `p | n`. No actual division needed.

```rust
struct TrialDivisor {
    p: u64,
    pinv: u64,  // p^(-1) mod 2^64
    plim: u64,  // floor((2^64 - 1) / p)
}

fn divides(&self, n: u64) -> bool {
    n.wrapping_mul(self.pinv) <= self.plim
}
```

### P-1 Method

Stage 1: Compute a^M mod n where M = lcm(1..B1).
Stage 2: Check gcd(a^M - 1, n) for a factor.
Use u64 Montgomery multiplication for cofactors < 2^64.

### P+1 Method

Lucas sequence V_k(P, 1) mod n. Stage 1: k = lcm(1..B1).
Same u64 Montgomery arithmetic.

### ECM

For cofactors up to ~60 bits, u64 Montgomery arithmetic is sufficient.
Reuse our `ecm` crate's algorithm but with u64 arithmetic instead of BigUint.

B1 sequence: 105, 115, 126, 137, 149, 161, 174, ...
B2 = 50 * B1 (rounded to odd multiple of 105).
Number of curves: 1-2 for lpb=17-18.

## Phase 5: Filtering

For 96-128 bit, the matrix is small (< 50K relations). Filtering is lightweight:

### Singleton Removal

```rust
fn remove_singletons(relations: &mut Vec<Relation>, fb_size: usize) {
    loop {
        // Count column weights
        let mut weights = vec![0u32; fb_size + large_prime_count];
        for rel in relations.iter() {
            for &(col, _) in &rel.factors {
                weights[col] += 1;
            }
        }
        // Remove relations containing singleton columns
        let before = relations.len();
        relations.retain(|rel| {
            rel.factors.iter().all(|&(col, _)| weights[col] >= 2)
        });
        if relations.len() == before { break; }
    }
}
```

### Duplicate Removal

Hash-based: (a, b) pairs must be unique.

## Phase 6: Linear Algebra (reuse gnfs)

For 96-128 bit, the matrix is < 50K × 50K. Dense GF(2) Gaussian elimination
from `gnfs::linalg::find_dependencies()` works fine. Block Wiedemann would be
needed only for > 200-bit numbers.

Reuse: `gnfs::linalg::build_matrix()`, `find_dependencies()`,
`randomize_dependencies()`.

## Phase 7: Square Root (reuse gnfs)

Reuse `gnfs::sqrt::extract_factor_verbose()` and `extract_factor_diagnostic()`.
These compute rational and algebraic square roots from a dependency set.

## CADO-Matched Parameters

```rust
pub fn params_c30() -> NfsParams {
    NfsParams {
        degree: 3,
        lim0: 30_000, lim1: 30_000,
        lpb0: 17, lpb1: 17,
        mfb0: 18, mfb1: 18,
        log_i: 9,
        qmin: 50_000, qrange: 1_000,
        rels_wanted: 30_000,
    }
}

pub fn params_c35() -> NfsParams {
    NfsParams {
        degree: 3,
        lim0: 40_000, lim1: 40_000,
        lpb0: 18, lpb1: 18,
        mfb0: 20, mfb1: 20,
        log_i: 9,
        qmin: 55_000, qrange: 1_500,
        rels_wanted: 35_000,
    }
}

pub fn params_c40() -> NfsParams {
    NfsParams {
        degree: 4,
        lim0: 50_000, lim1: 55_000,
        lpb0: 18, lpb1: 18,
        mfb0: 22, mfb1: 22,
        log_i: 9,
        qmin: 55_000, qrange: 1_500,
        rels_wanted: 40_000,
    }
}
```

## Benchmark Plan

1. Generate 3 fixed semiprimes at each of 96, 112, 128 bits (seed=42).
2. Run CADO-NFS via `cado-evolve` wrapper on each. Record wall time, rels/sec.
3. Run rust-nfs on same semiprimes. Record wall time, rels/sec.
4. Both use 8 threads for fair comparison.
5. Compare: sieve time, total time, rels/sec, relations found.

## Implementation Order

1. Scaffold crate + params + types
2. Factor base construction
3. Bucket sieve data structure
4. Q-lattice reduction
5. P-lattice reduction (Franke-Kleinjung)
6. Norm initialization
7. Small sieve
8. Per-region processing (apply + threshold + survivors)
9. Special-q loop with rayon parallelism
10. Montgomery trial division
11. P-1 method
12. P+1 method
13. ECM (u64 Montgomery)
14. Cofactorization pipeline
15. Filtering (singleton + duplicate removal)
16. Pipeline integration (poly → sieve → filter → LA → sqrt)
17. CLI and benchmark harness
18. Head-to-head comparison with CADO-NFS

## Testing Strategy

- Unit tests for each module (lattice reduction, bucket ops, cofactorization)
- Integration test: factor known 64-bit semiprime end-to-end
- Validation: for each relation, verify a*g1 + b*g0 = product of rational factors
  and F(a,b) = product of algebraic factors (modulo large primes)
- Benchmark: compare rels/sec against CADO on identical semiprimes
