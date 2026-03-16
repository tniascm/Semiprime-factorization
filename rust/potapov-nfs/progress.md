# Rust-NFS vs CADO-NFS Benchmark (2026-03-07)

c30 semiprime: `684217602914977371691118975023` (100 bits)
10 runs each, Apple M-series (10 cores)

## Head-to-head summary (post-optimization)

| Metric | Rust-NFS | CADO-NFS | Ratio |
|--------|----------|----------|-------|
| 1-thread CPU | 5548 ms | 2149 ms | CADO **2.6x** faster |
| Multi-thread wall | 1923 ms | 7534 ms | Rust **3.9x** faster |

### Improvement vs pre-optimization baseline

| Metric | Before | After | Saved |
|--------|--------|-------|-------|
| 1-thread CPU | 6917 ms | 5548 ms | **19.8%** |
| Multi-thread wall | 2154 ms | 1923 ms | **10.7%** |
| Sieve (multi-thread) | 936 ms | 744 ms | **20.5%** |

## Single-threaded (10 runs, post-optimization)

| Run | Rust-NFS (ms) | CADO-NFS CPU (ms) |
|-----|---------------|-------------------|
| 1 | 5521 | 2140 |
| 2 | 5519 | 2140 |
| 3 | 5523 | 2180 |
| 4 | 5531 | 2160 |
| 5 | 5552 | 2150 |
| 6 | 5538 | 2130 |
| 7 | 5553 | 2150 |
| 8 | 5564 | 2140 |
| 9 | 5599 | 2180 |
| 10 | 5579 | 2120 |
| **Avg** | **5548** | **2149** |

## Multi-threaded (10 runs, post-optimization, default 10 threads)

| Run | Rust wall (ms) | Rust sieve (ms) |
|-----|----------------|-----------------|
| 1 | 1912 | 765 |
| 2 | 1889 | 736 |
| 3 | 1937 | 765 |
| 4 | 1913 | 743 |
| 5 | 1949 | 736 |
| 6 | 1914 | 734 |
| 7 | 1937 | 743 |
| 8 | 1910 | 736 |
| 9 | 1928 | 745 |
| 10 | 1936 | 740 |
| **Avg** | **1923** | **744** |

## Pre-optimization baseline (10 runs, for reference)

| Run | Rust 1-thread (ms) | Rust multi wall (ms) |
|-----|--------------------|--------------------|
| 1 | 6832 | 2150 |
| 2 | 6826 | 2127 |
| 3 | 6883 | 2172 |
| 4 | 6908 | 2129 |
| 5 | 6912 | 2181 |
| 6 | 6922 | 2123 |
| 7 | 6943 | 2177 |
| 8 | 6943 | 2156 |
| 9 | 6987 | 2166 |
| 10 | 7016 | 2158 |
| **Avg** | **6917** | **2154** |

## Optimizations applied (2026-03-07)

1. **i64 fast path for extended_gcd** — avoids i128 for p < 2^62 (commit af96e1f)
2. **CofactorConfig** — pre-compute prime lists for P-1/P+1/ECM once, not per-survivor (commit 91f0b12)
3. **FK lattice enumerator** — Franke-Kleinjung walk with partial-GCD reduction (commit fc91310)
4. **i64 root transform** — avoids i128 in root transformation when operands fit (commit 589c9a8)
5. **Scatter fast-path** — single compare per row when p > sieve_width (commit 589c9a8)
6. **Coprimality pre-filter** — skip both-even (i,j) positions in scatter (commit 0bf9675)

## Key remaining gaps

- **Sieve**: Rust ~4.4s vs CADO ~0.39s single-threaded (11x gap, still biggest target)
  - CADO uses: multi-level bucket tree, batch modular inversion, SIMD survivor scan
- **LA**: Rust dense GE vs CADO Block Wiedemann
- **Wall clock**: Rust wins 3.9x due to CADO Python/IPC overhead
