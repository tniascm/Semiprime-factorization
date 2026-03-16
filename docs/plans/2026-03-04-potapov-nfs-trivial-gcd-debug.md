# Potapov-NFS Trivial-GCD Debug Plan (Reproducible)

Date: 2026-03-04

## What changed

### Code updates (this pass)
- `rust/gnfs/src/linalg.rs`
  - Fixed randomized dependency generation bug for small nullspaces (`k >= n` previously collapsed randomization into repeated identical XORs).
  - Added dedup of randomized dependencies.
  - Randomized subset size in `[2, k_eff]` for better diversity.
- `rust/potapov-nfs/src/pipeline.rs`
  - Added env-controlled dependency knobs:
    - `POTAPOV_NFS_DEP_RANDOM_COUNT`
    - `POTAPOV_NFS_DEP_XOR_K`
    - `POTAPOV_NFS_MAX_DEPS_TRY`
    - `POTAPOV_NFS_TRIVIAL_BAIL`
    - `POTAPOV_NFS_SQRT_VERBOSE_DEPS`
    - `POTAPOV_NFS_QC_COUNT`
  - Added explicit sqrt candidate composition log (`GE` vs random counts).
  - Prioritizes randomized dependencies before basis dependencies.
  - Added all new knobs into structured `run_config.json`.
  - Added additional debug/tuning controls:
    - `POTAPOV_NFS_SKIP_SQRT`
    - `POTAPOV_NFS_MAX_DEP_LEN`
    - `POTAPOV_NFS_REQUIRE_COPRIME_AB`
    - `POTAPOV_NFS_FULL_ONLY`
    - `POTAPOV_NFS_IGNORE_SPECIAL_Q_COLUMN`
    - `POTAPOV_NFS_OVR_*` parameter overrides (`LIM/LPB/MFB/LOG_I/Q*`).
  - Added LA density diagnostics and explicit runtime parameter echo.
- `rust/potapov-nfs/src/sieve/mod.rs`
  - Parallelized bucket-region survivor discovery with rayon while preserving deterministic bucket order in output.
- `rust/gnfs/src/arith.rs`
  - Fixed quadratic-character selector edge case so `count=0` truly disables QC columns.
- `rust/gnfs/src/sqrt.rs`
  - Added verbose per-root GCD diagnostics (`gcd(x-y)`, `gcd(x+y)`) in `extract_factor_verbose` path.
  - Added optional debug check for evaluating algebraic roots at `-m` (`GNFS_TRY_NEG_M`).
- `rust/potapov-nfs/scripts/bench_cado_vs_potapov_nfs.py`
  - Added reproducibility controls and metadata for dependency knobs.
  - Added optional deterministic random semiprime generation:
    - `--random-semiprimes`
    - `--random-seed`
  - Added rust build step by default (`--skip-build-rust` to disable).
  - Added `rust_log_run_count` and serialized generated `cases` in output JSON.
- `rust/potapov-nfs/scripts/sweep_kernel_health.py`
  - New reproducible sweep harness to scan `(log_i, q_windows)` with `POTAPOV_NFS_SKIP_SQRT=1`.
  - Captures per-run rows/cols/deps, kernel margin, and raw logs.

## Repro benchmark artifacts (current)

1. Fixed-case reproducible benchmark:
- `rust/data/cado_rust_c30_c45_benchmark_2026-03-04_tuned1.json`
- `rust/data/cado_rust_c30_c45_benchmark_2026-03-04_tuned1/`
1b. Fixed-case benchmark after parallel bucket scan:
- `rust/data/cado_rust_c30_c45_benchmark_2026-03-04_tuned2_parallelbucket.json`
- `rust/data/cado_rust_c30_c45_benchmark_2026-03-04_tuned2_parallelbucket/`

2. Random-semiprime reproducible benchmark (seeded):
- `rust/data/cado_rust_c30_c45_benchmark_2026-03-04_randomseed123.json`
- `rust/data/cado_rust_c30_c45_benchmark_2026-03-04_randomseed123/`
3. c30 sqrt diagnostic sweep:
- `rust/data/potapov_nfs_sqrt_sweep_2026-03-04_c30.json`
- `rust/data/potapov_nfs_sqrt_sweep_2026-03-04_c30/raw/`
4. kernel-health sweep smoke artifact:
- `rust/data/potapov_nfs_kernel_sweep_2026-03-04_smoketest.json`
- `rust/data/potapov_nfs_kernel_sweep_2026-03-04_smoketest/`
5. strict kernel-health sweep:
- `rust/data/potapov_nfs_kernel_sweep_2026-03-04_strict_logi9_11_qw10.json`
- `rust/data/potapov_nfs_kernel_sweep_2026-03-04_strict_logi9_11_qw10/`
6. strict kernel-health point (real kernel):
- `rust/data/potapov_nfs_kernel_sweep_2026-03-04_strict_logi11_qw20.json`
- `rust/data/potapov_nfs_kernel_sweep_2026-03-04_strict_logi11_qw20/`

Both include structured rust logs (`rust_logs/*/run_config.json`).

## Key measured results

### Fixed-case run (`..._tuned1.json`, 1 case per tier)
- c30
  - CADO elapsed: `5.80251s`
  - potapov-nfs total: `16.33365s`
  - rust factor: `null`
- c45
  - CADO elapsed: `8.67259s`
  - potapov-nfs total: `84.21514s`
  - rust factor: `null`

Notable improvement in c45 sqrt stage vs previous:
- sqrt dropped from ~14.6s to ~1.16s due dependency randomization/`k` changes.
- Total still dominated by sieve time.

### Fixed-case run after parallel bucket scan (`..._tuned2_parallelbucket.json`, 1 case per tier)
- c30
  - CADO elapsed: `5.74171s`
  - potapov-nfs total: `11.87411s` (was `16.33365s`, 1.38x faster)
  - rust factor: `null`
- c45
  - CADO elapsed: `8.61720s`
  - potapov-nfs total: `56.63106s` (was `84.21514s`, 1.49x faster)
  - rust factor: `null`

### Follow-up experiment (rejected)
- `..._tuned3_parallelcofact.json` tried adding parallel cofactorization on top of tuned2.
- Result regressed slightly (about +0.4s c30, +1.2s c45), so that change was reverted.

### c30 sqrt sweep findings (`...potapov_nfs_sqrt_sweep_2026-03-04_c30.json`)
- Tested reproducibly across modes:
  - baseline (`a-b*alpha`)
  - `a+b*alpha`
  - `b*alpha-a`
  - `qc=0`
  - `try -m`
- All 5 modes still failed to extract factor with `trivial_gcd=100`.
- `a+b*alpha` changed the first-dependency signature from `gcd=N/1` to `gcd=1/1`,
  but did not produce nontrivial factors even with deeper dependency search.

### Seeded random-semiprime run (`..._randomseed123.json`)
- c30 case: `507472316304652726437478967651`
  - CADO elapsed: `5.75502s`
  - potapov-nfs total: `15.22318s`
  - factor: `null`
- c45 case: `164611785070490797662303504431770568238416303`
  - CADO elapsed: `8.64679s`
  - potapov-nfs total: `81.61717s`
  - factor: `null`

### Fresh baseline re-run after debug instrumentation (`..._tuned5_postdebugfinal.json`)
- Artifact:
  - `rust/data/cado_rust_c30_c45_benchmark_2026-03-04_tuned5_postdebugfinal.json`
- c30:
  - CADO elapsed: `5.91659s`
  - potapov-nfs total: `12.06856s`
  - rust factor: `null`
- c45:
  - CADO elapsed: `8.69962s`
  - potapov-nfs total: `59.36441s`
  - rust factor: `null`

### Standalone `gnfs` confirmation
- Running `rust/gnfs` directly on the c30 fixed case reproduces the same pattern:
  - many dependencies available
  - `rat_not_square=0`, `alg_not_square=0`
  - every tried dependency gives trivial gcd.
- Therefore this is a shared `gnfs` post-processing correctness issue, not specific to potapov-nfs remapping.

### Matrix health / dependency quality findings (new)
- Added relation and matrix diagnostics show the current LA input is dominated by sparse columns (`special_q` and LP columns), with far fewer rows than total columns in default runs.
- Enforcing mathematically stricter rows (`gcd(a,b)=1`, full-only rows) removes the previous spurious kernel in small runs:
  - with default `log_i=9`, no dependencies remain.
- With larger sieve area (`POTAPOV_NFS_OVR_LOG_I=11`, `POTAPOV_NFS_MAX_Q_WINDOWS=20`) and strict rows:
  - matrix reached `9387 x 8106` (rows > cols), yielding a real kernel (`3043` GE deps),
  - but dependencies are extremely long (all >1000 relations), making sqrt stage impractically expensive.
- Repro artifact for this point:
  - `logi11_qw20`: rows `9387`, cols `8106`, margin `+1281`, deps `3243`, wall `96.88s` (`skip_sqrt=1`).
- This points to a missing CADO-equivalent merge/replay stage: we currently expose sparse columns directly to LA instead of first collapsing them into compact relation-sets.
- Strict sweep (`log_i=9,10,11`, `q_windows=10`, `skip_sqrt=1`) confirms why default runs lack valid kernels:
  - `log_i=9`: `1420 x 8106` (margin `-6686`)
  - `log_i=10`: `2928 x 8106` (margin `-5178`)
  - `log_i=11`: `5559 x 8106` (margin `-2547`)

## Critical debug finding

Verbose sqrt diagnostics (c30) show:
- For each tried dependency, Newton produces two `y` candidates.
- One gives `gcd(x-y)=N` and the other gives `gcd(x+y)=N`.
- The opposite GCD is always `1`.
- Therefore `y \equiv \pm x (mod N)` for every tested dependency (systematic trivial-GCD pattern).

Additional diagnostic:
- With `GNFS_SQRT_RELAX_EXACT=1`, Couveignes path can emit many extra `y` candidates per dependency.
- Those additional roots mostly produce `gcd=1` on both `x-y` and `x+y`, and still never produce nontrivial factors on tested c30 dependencies.
- So the blocker is upstream dependency/relation-set semantics, not just Newton root branch selection.

Reproduction command:
```bash
cd rust/potapov-nfs
POTAPOV_NFS_MAX_VARIANTS=1 \
POTAPOV_NFS_MAX_Q_WINDOWS=10 \
POTAPOV_NFS_SQRT_VERBOSE_DEPS=3 \
POTAPOV_NFS_MAX_DEPS_TRY=3 \
POTAPOV_NFS_TRIVIAL_BAIL=3 \
./target/release/potapov-nfs --factor 684217602914977371691118975023
```

## Next steps (priority)

1. Square-root stage correctness against CADO
- Implement/port CADO-equivalent algebraic sqrt path (including unit handling / map constraints used in CADO postprocessing).
- Stop treating current exact-root recovery as production-correct until cross-validated on same relation sets.

2. Relation-level A/B verification harness
- For a fixed dependency, compare:
  - rust `x`, rust `y`, gcd outcomes
  - cado postprocess outputs for same dependency set (or equivalent exported relation matrix).
- Goal: identify first semantic divergence (relation normalization, signs, special-q handling, unit maps, or sqrt root selection).

3. Keep reproducibility strict while iterating
- Continue using:
  - fixed seed (`POTAPOV_NFS_DEP_SEED`)
  - fixed benchmark case lists (or `--random-semiprimes --random-seed`)
  - structured run logs (`POTAPOV_NFS_LOG_DIR`)
- Require artifact JSON + raw logs on every tuning pass.

4. Throughput optimization after correctness
- Once nontrivial factors are reproducibly extracted on c30/c45, optimize sieve throughput and add parallel special-q execution.
- Do not optimize further around current trivial-GCD behavior; it masks correctness risk.

## New debug knobs added (for reproducible sweeps)

- `GNFS_SQRT_RELAX_EXACT`
  - Couveignes debug mode: accept Hensel/CRT-consistent roots even if center-lifted coefficients do not satisfy exact `gamma^2 = product` in `Z[alpha]/(f)`.
- `POTAPOV_NFS_DEBUG_REL_STATS`
  - prints `gcd(a,b)>1` relation counts after filtering.
- `POTAPOV_NFS_REQUIRE_COPRIME_AB`
  - filters to relations with `gcd(a,b)=1`.
- `POTAPOV_NFS_FULL_ONLY`
  - keeps only full relations (drops 1LP partials before LA).
- `POTAPOV_NFS_IGNORE_SPECIAL_Q_COLUMN`
  - debug-only: drops special-q sparse columns from matrix rows.
- `POTAPOV_NFS_MAX_DEP_LEN`
  - skips dependencies above a configured relation-count cap to avoid runaway sqrt time.
- `POTAPOV_NFS_SKIP_SQRT`
  - skips Stage-5 entirely; used by kernel-health sweeps to profile sieve/filter/LA only.
- parameter override knobs in `factor_nfs_inner`:
  - `POTAPOV_NFS_OVR_LIM0`, `POTAPOV_NFS_OVR_LIM1`
  - `POTAPOV_NFS_OVR_LPB0`, `POTAPOV_NFS_OVR_LPB1`
  - `POTAPOV_NFS_OVR_MFB0`, `POTAPOV_NFS_OVR_MFB1`
  - `POTAPOV_NFS_OVR_LOG_I`, `POTAPOV_NFS_OVR_QMIN`, `POTAPOV_NFS_OVR_QRANGE`
  - `POTAPOV_NFS_OVR_RELS_WANTED`

## Update: 2026-03-04 (later pass)

### New code changes
- `rust/potapov-nfs/src/pipeline.rs`
  - Added adaptive dependency-length tiers in sqrt:
    - `POTAPOV_NFS_DEP_LEN_TIERS` (explicit tiers)
    - `POTAPOV_NFS_DEP_AUTO_RELAX` (default on; auto `cap,2x,4x,none`)
  - Added dependency-length percentile diagnostics:
    - min/p50/p90/p99/max expanded dependency lengths.
  - Added optional matrix column compaction:
    - `POTAPOV_NFS_COMPACT_ZERO_COLS` (default on).
    - Drops all-zero columns before LA (`ncols` reduction can be large).
  - Added optional dependency relation hygiene gate:
    - `POTAPOV_NFS_DEP_REQUIRE_COPRIME_REL`
    - Keeps non-coprime rows in LA but can reject dependency candidates containing them in sqrt.
  - Expanded structured run logging to include all new env knobs.
- `rust/potapov-nfs/scripts/sweep_factor_configs.py`
  - New reproducible sweep harness for full factoring configs (raw logs + parsed metrics).
  - Added `--dep-require-coprime-rel`.
- `rust/potapov-nfs/scripts/sweep_kernel_health.py`
  - Added `--sparse-premerge` toggle.

### New artifacts
- c30 factor sweeps:
  - `rust/data/potapov_nfs_factor_sweep_2026-03-04_c30_logi11_qw14_20.json`
  - `rust/data/potapov_nfs_factor_sweep_2026-03-04_c30_logi11_qw10_12_full1.json`
  - `rust/data/potapov_nfs_factor_sweep_2026-03-04_c30_logi11_qw13_full1.json`
  - `rust/data/potapov_nfs_factor_sweep_2026-03-04_c30_logi11_qw13_full0.json`
  - `rust/data/potapov_nfs_factor_sweep_2026-03-04_c30_logi11_qw12_full0.json`
  - `rust/data/potapov_nfs_factor_sweep_2026-03-04_c30_logi11_qw13_full0_compact.json`
- c45 factor/kernel sweeps:
  - `rust/data/potapov_nfs_factor_sweep_2026-03-04_c45_logi10_qw20_full0.json`
  - `rust/data/potapov_nfs_factor_sweep_2026-03-04_c45_logi10_qw20_full0_compact.json`
  - `rust/data/potapov_nfs_factor_sweep_2026-03-04_c45_logi10_qw20_full0_nopremerge.json`
  - `rust/data/potapov_nfs_kernel_sweep_2026-03-04_c45_logi10_qw20_40_premerge1.json`
  - `rust/data/potapov_nfs_kernel_sweep_2026-03-04_c45_logi10_qw20_nocoprime_premerge1.json`
  - `rust/data/potapov_nfs_factor_sweep_2026-03-04_c45_logi10_qw20_full0_nocoprime.json`
  - `rust/data/potapov_nfs_factor_sweep_2026-03-04_c45_logi10_qw20_full0_nocoprime_deepdeps.json`
  - `rust/data/potapov_nfs_factor_sweep_2026-03-04_c45_logi10_qw20_full0_nocoprime_dep_coprime_rel.json`

### Current measured state
- c30 (fixed case `684217602914977371691118975023`)
  - Best successful config so far:
    - `log_i=11`, `q_windows=13`, `full_only=0`, `require_coprime=1`, `sparse_premerge=1`, `max_variants=1`
    - total: `78.68s` (`78682ms`) after zero-column compaction.
  - Matrix in best run:
    - rows `6435`, cols `6259`, deps `3220`.
  - Notes:
    - `q_windows=12` fails (no deps), so `13` is current minimum successful window.
- c45 (fixed case `327714360917956624476008484661697514621409657`)
  - With strict coprime rows (`require_coprime=1`), still no kernel:
    - example: `427 x 3916`, deps `0` at `log_i=10,q_windows=20,sparse_premerge=1`.
  - With no-coprime rows (`require_coprime=0`), LA becomes near-square and yields many deps:
    - `3819 x 3917`, deps `4031`.
    - but sqrt gives only trivial gcd outcomes:
      - `30/30 trivial` (default try budget),
      - `300/300 trivial` (deep budget).
  - Enabling `POTAPOV_NFS_DEP_REQUIRE_COPRIME_REL=1` with no-coprime LA skips almost all deps:
    - `skipped_non_coprime=3997`, `tried=0`.
    - confirms dependency pool is dominated by non-coprime rows.

### c45 kernel-scaling insight (critical)
- Sweep with `require_coprime=1`, `sparse_premerge=1`, `log_i=10`:
  - `q_windows=20`: `427 x 3916` (margin `-3489`)
  - `q_windows=30`: `625 x 5025` (margin `-4400`)
  - `q_windows=40`: `789 x 5858` (margin `-5069`)
- Margin worsens as windows increase because active dense columns grow faster than usable premerged rows.
- This indicates current relation semantics/filtering are not producing c45-useful kernels under coprime constraints.

### Updated immediate priorities
1. c45 correctness-first path:
  - diagnose why non-coprime-heavy deps collapse to trivial gcd in sqrt.
  - either recover a mathematically valid way to include such rows, or generate enough coprime rows without exploding dense columns.
2. filtering/merge fidelity vs CADO:
  - reproduce CADO-style relation matching and sparse merge behavior more closely (especially around LP handling and row validity constraints).
3. keep every tuning pass reproducible:
  - continue with sweep scripts + JSON artifacts + raw logs + run_config env capture.

## Update: 2026-03-04 (singleton-prune + c45 structure checks)

### Additional code changes
- `rust/potapov-nfs/src/pipeline.rs`
  - Added optional iterative singleton-column row pruning before LA:
    - `POTAPOV_NFS_SINGLETON_PRUNE`
    - `POTAPOV_NFS_SINGLETON_PRUNE_MIN_WEIGHT` (default 2)
  - Preserved row-source mapping through LA so dependency replay to original relations remains correct after row pruning.
  - Added run-config logging for singleton-prune knobs.
- `rust/potapov-nfs/scripts/sweep_kernel_health.py`
  - Added:
    - `--singleton-prune`
    - `--singleton-min-weight`
- `rust/potapov-nfs/scripts/sweep_factor_configs.py`
  - Added:
    - `--singleton-prune`
    - `--singleton-min-weight`
    - `--nf-mode`
    - `--try-neg-m`

### New artifacts (this sub-pass)
- `rust/data/potapov_nfs_kernel_sweep_2026-03-04_c45_logi10_qw20_coprime_ignoreSQ_premerge1.json`
- `rust/data/potapov_nfs_kernel_sweep_2026-03-04_c45_logi10_qw20_coprime_premerge1_singleton2.json`
- `rust/data/potapov_nfs_kernel_sweep_2026-03-04_c45_logi10_qw20_nocoprime_premerge1_singleton2.json`
- `rust/data/potapov_nfs_kernel_sweep_2026-03-04_c45_logi10_qw20_nocoprime_nopremerge.json`
- `rust/data/potapov_nfs_factor_sweep_2026-03-04_c45_logi10_qw20_full0_nocoprime_singleton2.json`
- `rust/data/potapov_nfs_factor_sweep_2026-03-04_c45_logi10_qw20_full0_nocoprime_singleton2_nfplus.json`
- `rust/data/potapov_nfs_factor_sweep_2026-03-04_c45_logi10_qw20_full0_nocoprime_nopremerge.json`

### Key findings from this sub-pass
- c45, strict coprime path remains the bottleneck:
  - baseline strict/premerge: `427 x 3916`, deps `0`.
  - `ignore_special_q=1` improved rows but still underdetermined: `1384 x 5628`, deps `0`.
  - singleton-prune on strict path is over-aggressive currently:
    - all rows dropped (`427 -> 0`) at min-weight 2.
- c45, no-coprime paths now have very healthy dependency counts:
  - premerge + singleton-prune(2): `3562 x 2139`, deps `4029`, margin `+1423`.
  - no-premerge: `5578 x 8879`, deps `4372`.
- Despite those kernels, sqrt still fails identically:
  - all tried dependencies produce `trivial_gcd`.
  - reproduced across:
    - premerge on/off
    - singleton-prune on/off
    - NF mode change (`GNFS_NF_ELEMENT_MODE=a_plus_ba`)
  - failure signature example: `trivial_gcd=300`, `rat_not_square=0`, `alg_not_square=0`.

### Current interpretation
- For c45, this is no longer a “no-kernel” issue in no-coprime mode.
- The blocker is now square-root semantics/dependency validity:
  - generated dependencies are algebraically square but collapse to `y = ±x (mod N)`.
- Coprime-only relation sets still fail to provide enough viable rows under current relation/filter semantics.
