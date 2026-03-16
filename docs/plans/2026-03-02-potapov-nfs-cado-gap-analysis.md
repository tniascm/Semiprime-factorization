# Potapov-NFS vs CADO-NFS Gap Analysis (c30/c45)

Date: 2026-03-02

## 1) Fresh benchmark baseline (same random semiprimes)

Semiprimes used:
- c30: 684217602914977371691118975023, 291695886709214217732173542261, 457206828091130153032152360761
- c45: 327714360917956624476008484661697514621409657, 257924597305621056220013631148395117044787913, 960564208229425980896064150966530613470478361

CADO invocation (4 threads, local clients):
- `python3 cado-nfs.py --parameters parameters/factor/params.c30 -t 4 <N> server.whitelist=0.0.0.0/0 slaves.hostnames=localhost slaves.nrclients=2 server.address=localhost`
- same for `params.c45`

Raw benchmark artifacts:
- Before fixes: `rust/data/cado_rust_c30_c45_benchmark_2026-03-02.json`
- After fixes: `rust/data/cado_rust_c30_c45_benchmark_2026-03-02_after_fixes.json`

## 2) What was fixed in Potapov-NFS this pass

Files changed:
- `rust/potapov-nfs/src/sieve/mod.rs`
- `rust/potapov-nfs/src/sieve/small.rs`
- `rust/potapov-nfs/src/cofactor/mod.rs`
- `rust/potapov-nfs/src/cofactor/trialdiv.rs`
- `rust/potapov-nfs/src/params.rs`
- compatibility build fixes in `rust/potapov-nfs/src/relation.rs` and `rust/potapov-nfs/src/pipeline.rs`

Fixes applied:
1. Corrected i-coordinate offset in sieve hit placement (`i` is centered, row index is `i + I`).
2. Aligned c30/c45 params with CADO files (`qmin`, `qrange`, `rels_wanted`).
3. Added u128 algebraic trial-division/cofactor path (`cofactorize_u128`) to avoid dropping c45 norms on u64 overflow.

Validation:
- `cargo test --lib` passes (126 tests).

## 3) Measured impact (before -> after)

c30:
- Rust mean sieve time: 5.335s -> 1.838s (2.90x faster)
- Rust mean reported total: 5.335s -> 2.249s (2.37x faster)
- Rust mean wall script time: 26.693s -> 11.103s (2.40x faster)
- Rust mean relations found: 20 -> 349.3
- CADO mean solve time: 5.964s

c45:
- Rust mean sieve time: 18.270s -> 9.737s (1.88x faster)
- Rust mean reported total: 18.270s -> 12.516s (1.46x faster)
- Rust mean wall script time: 94.969s -> 61.113s (1.55x faster)
- Rust mean relations found: 0.33 -> 495
- CADO mean solve time: 8.867s

## 4) Why Rust is still much slower / not solving

### A. Hidden retry overhead dominates wall clock
`factor_nfs` runs up to 5 polynomial variants serially. `total_ms` only reflects one attempt, while real wall time includes all attempts.
- c45 after fixes: mean `total_ms` 12.5s vs mean wall 61.1s.

### B. Relation yield is still far below CADO quality per special-q
After fixes, yield improved massively, but still low against CADO relation economy.
- CADO c30 gets target relations in ~88 special-q.
- Rust c30 processes ~89 special-q but still only hundreds of raw relations and tiny usable full set.

### C. LA handoff currently drops almost everything
`remap_hybrid` skips most full relations (`remap_hybrid: skipped ...`) causing near-empty matrices.
This is the immediate reason factorization still fails even when sieve produces many candidates.

### D. No effective parallel sieving
Rust CLI sets rayon thread pool, but sieve hot loops are serial (no `par_iter` in potapov-nfs core path).
CADO is fully parallelized in LAS internals.

## 5) CADO behavior we must match (from source)

Key implementation points read from CADO sources:
- Special-q and region shaping: `sieve/las-choose-sieve-area.cpp`, `sieve/las-norms.cpp` (`sieve_info_adjust_IJ`, discard rules, rounded bucket-region alignment).
- Smart lognorm init (critical): `sieve/las-norms.cpp` (`lognorm_smart::fill_rat`, piecewise/transition-based fill, not per-cell pow/log).
- Bucket + small sieve + process region orchestration: `sieve/las-process-bucket-region.cpp`, `sieve/las-fill-in-buckets.cpp`, `sieve/las-smallsieve.cpp`.
- Cofactor strategy chain: `sieve/ecm/facul_strategies.cpp` default strategy (P-1 315/2205, P+1 525/3255, ECM chain with tuned B1/B2 progression).
- Filtering/merge pipeline: `filter/singleton_removal.c`, `filter/merge.c`.

## 6) Path to target

Step 1 (parity with CADO solve time, ignore warm-up):
1. Make wall metrics honest: report aggregate across all polynomial variants or disable retries in benchmark mode.
2. Fix LA remap path so full relations survive into matrix build.
3. Parallelize special-q processing / region loops.
4. Replace per-cell algebraic/rational norm init with CADO-like smart fill kernels.

Step 2 (+10% faster solve-time than CADO after warm-up):
1. Keep sieve/LA/sqrt in-process with zero serialization between stages.
2. Parallel special-q task queue + cache-local bucket-region workers.
3. Precompute and reuse transformed small-sieve structures per special-q root where possible.
4. Add benchmark mode with fixed poly variant count and deterministic special-q ranges for clean regression tracking.

Current status relative to target:
- c30: Rust sieve is now faster than CADO sieve wall (1.84s vs 0.43s still slower), but full solve is not achieved due LA/remap drop.
- c45: Rust still slower than CADO solve-time and not solving; biggest blockers are retry overhead + LA/remap loss + lack of parallelism.
