//! Special-q lattice sieve: the main sieve entry point.
//!
//! For each special-q prime q in [qmin, qmin+qrange):
//!   1. Reduce q-lattice
//!   2. For each FB prime: reduce p-lattice (bucket), or precompute small sieve
//!   3. For each bucket region: init norms, small sieve, apply bucket updates, scan
//!   4. Cofactorize survivors to produce relations

pub mod bucket;
pub mod lattice;
pub mod norm;
pub mod region;
pub mod small;

use crate::arith::sieve_primes;
use crate::cofactor::{self, CofactResult};
use crate::factorbase::{self, FactorBase};
use crate::lp_key::LpKey;
use crate::params::NfsParams;
use crate::relation::Relation;
use rayon::prelude::*;
use std::collections::HashMap;

use self::bucket::{BucketArray, BucketUpdate, BUCKET_REGION, LOG_BUCKET_REGION};
use self::lattice::{reduce_plattice, reduce_qlattice, QLattice};
// norm functions are used via direct inline computation in the sieve loop;
// the module is re-exported for external use.
use self::region::{apply_bucket_updates, pos_to_ij, scan_survivors};
use self::small::{precompute_small_sieve_alg, precompute_small_sieve_rat, small_sieve_region};

/// Result of sieving: relations + timing breakdown.
#[derive(Debug, Clone)]
pub struct SieveResult {
    pub relations: Vec<Relation>,
    pub special_qs_processed: usize,
    pub survivors_found: usize,
    pub total_ms: f64,
    pub root_enum_ms: f64,
    pub roots_from_fb: usize,
    pub roots_fallback: usize,
    pub bucket_setup_ms: f64,
    pub region_scan_ms: f64,
    pub sieve_ms: f64,
    pub cofactor_ms: f64,
}

/// Run the special-q lattice sieve.
///
/// For each special-q prime q in [qmin, qmin+qrange):
///   1. Reduce q-lattice
///   2. For each FB prime: reduce p-lattice, scatter bucket updates
///   3. For each bucket region: init norms, apply updates, scan survivors
///   4. Cofactorize survivors
pub fn sieve_specialq(
    f_coeffs: &[i64],
    m: u64,
    rat_fb: &FactorBase,
    alg_fb: &FactorBase,
    params: &NfsParams,
    q_start: u64,
    q_range: u64,
    max_relations: Option<usize>,
    sq_root_cache: Option<&HashMap<u64, Vec<u64>>>,
) -> SieveResult {
    let start_time = std::time::Instant::now();

    let half_i = params.sieve_half_width() as i64;
    let sieve_width = (2 * half_i) as usize;
    // Match GNFS/CADO-style region shape with J = I.
    let max_j = params.sieve_half_width().max(1) as usize;
    let total_sieve_area = sieve_width * max_j;
    let n_buckets = (total_sieve_area + BUCKET_REGION - 1) / BUCKET_REGION;

    // Bucket threshold: primes below this use small sieve, above use bucket sieve
    let bucket_thresh = (half_i as u64).max(64);

    let scale = rat_fb.scale;

    // Rational polynomial: g(x) = x - m, so g1 = 1, g0 = -m
    let g0 = -(m as f64);
    let g1 = 1.0f64;

    // Precompute FB partitions once; q-dependent exclusion is handled in-loop.
    let small_rat_primes_all: Vec<u64> = rat_fb
        .primes
        .iter()
        .copied()
        .filter(|&p| p < bucket_thresh)
        .collect();
    let small_rat_logp_all: Vec<u8> = rat_fb
        .primes
        .iter()
        .enumerate()
        .filter(|(_, &p)| p < bucket_thresh)
        .map(|(idx, _)| rat_fb.log_p[idx])
        .collect();
    let small_alg_primes_all: Vec<u64> = alg_fb
        .primes
        .iter()
        .copied()
        .filter(|&p| p < bucket_thresh)
        .collect();
    let small_alg_roots_all: Vec<Vec<u64>> = alg_fb
        .primes
        .iter()
        .enumerate()
        .filter(|(_, &p)| p < bucket_thresh)
        .map(|(idx, _)| alg_fb.roots[idx].clone())
        .collect();
    let small_alg_logp_all: Vec<u8> = alg_fb
        .primes
        .iter()
        .enumerate()
        .filter(|(_, &p)| p < bucket_thresh)
        .map(|(idx, _)| alg_fb.log_p[idx])
        .collect();
    let rat_large_indices: Vec<usize> = rat_fb
        .primes
        .iter()
        .enumerate()
        .filter_map(|(idx, &p)| if p >= bucket_thresh { Some(idx) } else { None })
        .collect();
    let alg_large_indices: Vec<usize> = alg_fb
        .primes
        .iter()
        .enumerate()
        .filter_map(|(idx, &p)| if p >= bucket_thresh { Some(idx) } else { None })
        .collect();
    let rat_roots_by_index: Vec<u64> = rat_fb.primes.iter().map(|&p| m % p).collect();

    let mut all_relations = Vec::new();
    let mut total_survivors = 0usize;
    let mut total_bucket_setup_ns = 0u64;
    let mut total_region_scan_ns = 0u64;
    let mut total_cofact_ns = 0u64;
    let mut total_norm_ns = 0u64;
    let mut total_small_sieve_ns = 0u64;
    let mut total_bucket_apply_ns = 0u64;
    let mut sq_count = 0usize;
    let norm_block = std::env::var("RUST_NFS_NORM_BLOCK")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(16usize);

    let q_end = q_start + q_range;
    let sq_primes: Vec<u64> = sieve_primes(q_end)
        .into_iter()
        .filter(|&p| p >= q_start)
        .collect();

    let root_enum_start = std::time::Instant::now();
    let mut roots_from_fb = 0usize;
    let mut roots_fallback = 0usize;
    let mut qr_pairs = Vec::new();
    for &q in &sq_primes {
        sq_count += 1;
        // Fast path: special-q roots are usually already available in the
        // algebraic factor base when q <= lim1. Avoid re-running exhaustive
        // root search per q in that common case.
        if let Ok(idx) = alg_fb.primes.binary_search(&q) {
            roots_from_fb += 1;
            for &r in &alg_fb.roots[idx] {
                qr_pairs.push((q, r));
            }
            continue;
        }

        // Second-tier: cached roots for special-q primes above lim1.
        if let Some(cache) = sq_root_cache {
            if let Some(cached_roots) = cache.get(&q) {
                roots_from_fb += 1;
                for &r in cached_roots {
                    qr_pairs.push((q, r));
                }
                continue;
            }
        }

        // Fallback for q outside both FB and cache.
        roots_fallback += 1;
        let q_roots = factorbase::find_roots_mod_p(f_coeffs, q);
        for r in q_roots {
            qr_pairs.push((q, r));
        }
    }
    let root_enum_ms = root_enum_start.elapsed().as_secs_f64() * 1000.0;

    let batch_size = std::env::var("RUST_NFS_SQ_BATCH_SIZE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(64usize);
    let est_updates = estimate_updates_per_bucket(
        &rat_fb.primes,
        &alg_fb.primes,
        &alg_fb.roots,
        bucket_thresh,
        0,
    );
    let updates_per_bucket = (est_updates * 2).max(1024);

    for chunk in qr_pairs.chunks(batch_size.max(1)) {
        let chunk_results: Vec<(Vec<Relation>, usize, u64, u64, u64, u64, u64, u64)> = chunk
            .par_iter()
            .copied()
            .map_init(
                || {
                    let rat_buckets = BucketArray::new(n_buckets.max(1), updates_per_bucket);
                    let alg_buckets = BucketArray::new(n_buckets.max(1), updates_per_bucket);
                    let rat_sieve = vec![0u8; BUCKET_REGION];
                    let alg_sieve = vec![0u8; BUCKET_REGION];
                    let survivors = Vec::with_capacity(1024);
                    (rat_buckets, alg_buckets, rat_sieve, alg_sieve, survivors)
                },
                |(rat_buckets, alg_buckets, rat_sieve, alg_sieve, survivors_this_sq), (q, r)| {
                    rat_buckets.clear();
                    alg_buckets.clear();
                    survivors_this_sq.clear();

                    let mut local_rels = Vec::new();
                    let mut local_survivors = 0usize;
                    let sieve_start = std::time::Instant::now();

                    // 1. Reduce q-lattice
                    let skewness = 1.0;
                    let qlat = reduce_qlattice(q, r, skewness);

                    // 2. Precompute small sieve entries
                    let small_rat = if q < bucket_thresh {
                        let mut small_rat_primes = Vec::with_capacity(small_rat_primes_all.len());
                        let mut small_rat_logp = Vec::with_capacity(small_rat_logp_all.len());
                        for (idx, &p) in small_rat_primes_all.iter().enumerate() {
                            if p != q {
                                small_rat_primes.push(p);
                                small_rat_logp.push(small_rat_logp_all[idx]);
                            }
                        }
                        precompute_small_sieve_rat(&small_rat_primes, &small_rat_logp, m, &qlat)
                    } else {
                        precompute_small_sieve_rat(
                            &small_rat_primes_all,
                            &small_rat_logp_all,
                            m,
                            &qlat,
                        )
                    };

                    let small_alg = if q < bucket_thresh {
                        let mut small_alg_primes = Vec::with_capacity(small_alg_primes_all.len());
                        let mut small_alg_roots = Vec::with_capacity(small_alg_roots_all.len());
                        let mut small_alg_logp = Vec::with_capacity(small_alg_logp_all.len());
                        for (idx, &p) in small_alg_primes_all.iter().enumerate() {
                            if p != q {
                                small_alg_primes.push(p);
                                small_alg_roots.push(small_alg_roots_all[idx].clone());
                                small_alg_logp.push(small_alg_logp_all[idx]);
                            }
                        }
                        precompute_small_sieve_alg(
                            &small_alg_primes,
                            &small_alg_roots,
                            &small_alg_logp,
                            &qlat,
                        )
                    } else {
                        precompute_small_sieve_alg(
                            &small_alg_primes_all,
                            &small_alg_roots_all,
                            &small_alg_logp_all,
                            &qlat,
                        )
                    };

                    // 3. Scatter bucket updates

                    for &fb_idx in &alg_large_indices {
                        let p = alg_fb.primes[fb_idx];
                        if p == q {
                            continue;
                        }
                        for &root in &alg_fb.roots[fb_idx] {
                            scatter_bucket_updates_for_prime(
                                p,
                                root,
                                alg_fb.log_p[fb_idx],
                                &qlat,
                                params.log_i,
                                &mut *alg_buckets,
                                sieve_width,
                                max_j,
                                half_i,
                            );
                        }
                    }

                    for &fb_idx in &rat_large_indices {
                        let p = rat_fb.primes[fb_idx];
                        if p == q {
                            continue;
                        }
                        scatter_bucket_updates_for_prime(
                            p,
                            rat_roots_by_index[fb_idx],
                            rat_fb.log_p[fb_idx],
                            &qlat,
                            params.log_i,
                            &mut *rat_buckets,
                            sieve_width,
                            max_j,
                            half_i,
                        );
                    }

                    let bucket_setup_elapsed = sieve_start.elapsed().as_nanos() as u64;

                    // 4. Process each bucket region (sequentially)
                    let region_start_time = std::time::Instant::now();
                    let mut norm_ns: u64 = 0;
                    let mut small_sieve_ns: u64 = 0;
                    let mut bucket_apply_ns: u64 = 0;
                    let rat_bound = ((params.mfb0 as f64) * scale).min(255.0) as u8;
                    let alg_bound = ((params.mfb1 as f64) * scale).min(255.0) as u8;
                    let d = f_coeffs.len().saturating_sub(1);

                    for bucket_idx in 0..n_buckets {
                        let region_start = bucket_idx * BUCKET_REGION;
                        let region_end = (region_start + BUCKET_REGION).min(total_sieve_area);
                        let region_len = region_end - region_start;
                        if region_len == 0 {
                            continue;
                        }

                        rat_sieve[..region_len].fill(0);
                        alg_sieve[..region_len].fill(0);

                        let first_j = region_start / sieve_width;
                        let last_j = (region_end.saturating_sub(1)) / sieve_width;

                        for j_row in first_j..=last_j {
                            let row_start_global = j_row * sieve_width;
                            let row_end_global = row_start_global + sieve_width;

                            let overlap_start = row_start_global.max(region_start);
                            let overlap_end = row_end_global.min(region_end);
                            if overlap_start >= overlap_end {
                                continue;
                            }

                            let local_start = overlap_start - region_start;
                            let local_end = overlap_end - region_start;
                            let overlap_len = local_end - local_start;
                            let i_offset_in_row = overlap_start - row_start_global;

                            let t_norm = std::time::Instant::now();
                            let slope_rat = g1 * (qlat.a0 as f64) + g0 * (qlat.b0 as f64);
                            let intercept_rat =
                                (g1 * (qlat.a1 as f64) + g0 * (qlat.b1 as f64)) * (j_row as f64);

                            if norm_block == 1 {
                                for k in 0..overlap_len {
                                    let i_in_row = (i_offset_in_row + k) as i32 - half_i as i32;
                                    let f_val = slope_rat * (i_in_row as f64) + intercept_rat;
                                    rat_sieve[local_start + k] = log_norm_to_u8(f_val.abs(), scale);
                                }
                            } else {
                                for block_start in (0..overlap_len).step_by(norm_block) {
                                    let block_len = (overlap_len - block_start).min(norm_block);
                                    let k_mid = block_start + (block_len / 2);
                                    let i_in_row = (i_offset_in_row + k_mid) as i32 - half_i as i32;
                                    let f_val = slope_rat * (i_in_row as f64) + intercept_rat;
                                    let v = log_norm_to_u8(f_val.abs(), scale);
                                    rat_sieve[local_start + block_start
                                        ..local_start + block_start + block_len]
                                        .fill(v);
                                }
                            }

                            let a0f = qlat.a0 as f64;
                            let b0f = qlat.b0 as f64;
                            let a1f = qlat.a1 as f64;
                            let b1f = qlat.b1 as f64;
                            let j_f = j_row as f64;

                            if norm_block == 1 {
                                for k in 0..overlap_len {
                                    let i_in_row = (i_offset_in_row + k) as i32 - half_i as i32;
                                    let i_f = i_in_row as f64;
                                    let a = a0f * i_f + a1f * j_f;
                                    let b = b0f * i_f + b1f * j_f;
                                    let abs_f = eval_homogeneous_norm_f64(f_coeffs, a, b, d);
                                    alg_sieve[local_start + k] = log_norm_to_u8(abs_f, scale);
                                }
                            } else {
                                for block_start in (0..overlap_len).step_by(norm_block) {
                                    let block_len = (overlap_len - block_start).min(norm_block);
                                    let k_mid = block_start + (block_len / 2);
                                    let i_in_row = (i_offset_in_row + k_mid) as i32 - half_i as i32;
                                    let i_f = i_in_row as f64;
                                    let a = a0f * i_f + a1f * j_f;
                                    let b = b0f * i_f + b1f * j_f;
                                    let abs_f = eval_homogeneous_norm_f64(f_coeffs, a, b, d);
                                    let v = log_norm_to_u8(abs_f, scale);
                                    alg_sieve[local_start + block_start
                                        ..local_start + block_start + block_len]
                                        .fill(v);
                                }
                            }

                            norm_ns += t_norm.elapsed().as_nanos() as u64;

                            let t_ss = std::time::Instant::now();
                            let region_offset = i_offset_in_row;
                            small_sieve_region(
                                &mut rat_sieve[local_start..local_end],
                                &small_rat,
                                j_row as i32,
                                region_offset,
                                overlap_len,
                                sieve_width,
                            );
                            small_sieve_region(
                                &mut alg_sieve[local_start..local_end],
                                &small_alg,
                                j_row as i32,
                                region_offset,
                                overlap_len,
                                sieve_width,
                            );
                            small_sieve_ns += t_ss.elapsed().as_nanos() as u64;
                        }

                        let t_ba = std::time::Instant::now();
                        let rat_updates = rat_buckets.updates_for_bucket(bucket_idx);
                        let alg_updates = alg_buckets.updates_for_bucket(bucket_idx);
                        apply_bucket_updates(&mut rat_sieve[..region_len], rat_updates);
                        apply_bucket_updates(&mut alg_sieve[..region_len], alg_updates);

                        let survivor_positions = scan_survivors(
                            &rat_sieve[..region_len],
                            &alg_sieve[..region_len],
                            rat_bound,
                            alg_bound,
                        );

                        for &pos in &survivor_positions {
                            let (i, j) = pos_to_ij(bucket_idx, pos, sieve_width, half_i);
                            if j == 0 {
                                continue;
                            }

                            let a = qlat.a0 as i128 * i as i128 + qlat.a1 as i128 * j as i128;
                            let b = qlat.b0 as i128 * i as i128 + qlat.b1 as i128 * j as i128;

                            if b <= 0 || a == 0 {
                                continue;
                            }
                            if a.unsigned_abs() > i64::MAX as u128 || b > u64::MAX as i128 {
                                continue;
                            }

                            let mut u = a.unsigned_abs() as u64;
                            let mut v = b as u64;
                            while v != 0 {
                                let t = v;
                                v = u % v;
                                u = t;
                            }
                            if u != 1 {
                                continue;
                            }

                            survivors_this_sq.push((a as i64, b as u64));
                        }
                        bucket_apply_ns += t_ba.elapsed().as_nanos() as u64;
                    }

                    let region_scan_elapsed = region_start_time.elapsed().as_nanos() as u64;
                    local_survivors += survivors_this_sq.len();

                    let cofact_start = std::time::Instant::now();
                    // drain to iterate by value while clearing out the vector for next iteration
                    for (a, b) in survivors_this_sq.drain(..) {
                        let rat_norm = compute_rat_norm(a, b, m);
                        let Some(alg_norm) = compute_alg_norm(a, b, f_coeffs) else {
                            continue;
                        };

                        if rat_norm == 0 || alg_norm == 0 {
                            continue;
                        }

                        let mut alg_norm_reduced = alg_norm;
                        let q128 = q as u128;
                        if alg_norm_reduced % q128 != 0 {
                            continue; // false survivor
                        }
                        alg_norm_reduced /= q128;

                        let rat_result = cofactor::cofactorize(
                            rat_norm,
                            &rat_fb.trial_divisors,
                            params.lpb0,
                            params.mfb0,
                            params.lim0,
                        );

                        let alg_norm_to_factor = if alg_norm_reduced == 0 {
                            1
                        } else {
                            alg_norm_reduced
                        };
                        let alg_result = cofactor::cofactorize_u128(
                            alg_norm_to_factor,
                            &alg_fb.trial_divisors,
                            params.lpb1,
                            params.mfb1,
                            params.lim1,
                        );

                        if let Some(rel) =
                            build_relation(a, b, Some((q, r)), rat_result, alg_result)
                        {
                            local_rels.push(rel);
                        }
                    }

                    let cofact_elapsed = cofact_start.elapsed().as_nanos() as u64;
                    (
                        local_rels,
                        local_survivors,
                        bucket_setup_elapsed,
                        region_scan_elapsed,
                        cofact_elapsed,
                        norm_ns,
                        small_sieve_ns,
                        bucket_apply_ns,
                    )
                },
            )
            .collect();

        for (rels, survivors, bucket_setup_ns, region_scan_ns, cofact_ns, n_ns, ss_ns, ba_ns) in chunk_results {
            all_relations.extend(rels);
            total_survivors += survivors;
            total_bucket_setup_ns += bucket_setup_ns;
            total_region_scan_ns += region_scan_ns;
            total_cofact_ns += cofact_ns;
            total_norm_ns += n_ns;
            total_small_sieve_ns += ss_ns;
            total_bucket_apply_ns += ba_ns;
        }

        if let Some(limit) = max_relations {
            if all_relations.len() >= limit {
                break;
            }
        }
    }

    if let Some(limit) = max_relations {
        if all_relations.len() > limit {
            all_relations.truncate(limit);
        }
    }

    let sieve_profile = std::env::var("RUST_NFS_SIEVE_PROFILE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if sieve_profile {
        eprintln!(
            "  sieve-profile: norm={:.0}ms small_sieve={:.0}ms bucket_apply+scan={:.0}ms (of region_scan={:.0}ms)",
            total_norm_ns as f64 / 1_000_000.0,
            total_small_sieve_ns as f64 / 1_000_000.0,
            total_bucket_apply_ns as f64 / 1_000_000.0,
            total_region_scan_ns as f64 / 1_000_000.0,
        );
    }

    SieveResult {
        relations: all_relations,
        special_qs_processed: sq_count,
        survivors_found: total_survivors,
        total_ms: start_time.elapsed().as_secs_f64() * 1000.0,
        root_enum_ms,
        roots_from_fb,
        roots_fallback,
        bucket_setup_ms: total_bucket_setup_ns as f64 / 1_000_000.0,
        region_scan_ms: total_region_scan_ns as f64 / 1_000_000.0,
        sieve_ms: (total_bucket_setup_ns + total_region_scan_ns) as f64 / 1_000_000.0,
        cofactor_ms: total_cofact_ns as f64 / 1_000_000.0,
    }
}

/// Estimate the expected number of bucket updates per bucket region.
///
/// For each large FB prime p (above bucket_thresh), each root contributes
/// approximately BUCKET_REGION / p updates per bucket. Summing over all
/// such primes gives the expected load.
fn estimate_updates_per_bucket(
    rat_primes: &[u64],
    alg_primes: &[u64],
    alg_roots: &[Vec<u64>],
    bucket_thresh: u64,
    q: u64,
) -> usize {
    let mut sum = 0.0f64;

    // Rational side: one root per prime
    for &p in rat_primes {
        if p >= bucket_thresh && p != q {
            sum += (BUCKET_REGION as f64) / (p as f64);
        }
    }

    // Algebraic side: one entry per (prime, root) pair
    for (idx, &p) in alg_primes.iter().enumerate() {
        if p >= bucket_thresh && p != q {
            let n_roots = alg_roots[idx].len();
            sum += (BUCKET_REGION as f64) * (n_roots as f64) / (p as f64);
        }
    }

    (sum.ceil() as usize).max(256)
}

/// Scatter bucket updates for a single factor base prime using the reduced p-lattice.
///
/// For each row j in 0..max_j, we compute the transformed hit positions and scatter
/// them into the bucket array. The transformed root in (i,j)-space is precomputed
/// via `reduce_plattice`; for a prime p with transformed root R', hits in row j
/// occur at i positions: start = (R' * j) mod p, then stride by p.
fn scatter_bucket_updates_for_prime(
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
    let p_i128 = p as i128;

    // Recompute the transformed root R' = (root*b1 - a1) * (a0 - root*b0)^{-1} mod p
    let denom = ((qlat.a0 as i128 - (root as i128) * (qlat.b0 as i128)) % p_i128 + p_i128) % p_i128;
    let numer = (((root as i128) * (qlat.b1 as i128) - qlat.a1 as i128) % p_i128 + p_i128) % p_i128;
    if denom == 0 {
        // Projective in q-lattice coordinates: the prime hits every cell in
        // each p-th row (or every row in the degenerate numer==0 case).
        let row_period = if numer == 0 { 1usize } else { p as usize };
        for j in (0..max_j).step_by(row_period) {
            let row_base = j * sieve_width;
            for i_pos in 0..sieve_width {
                let global_pos = row_base + i_pos;
                buckets.push(
                    global_pos >> LOG_BUCKET_REGION,
                    BucketUpdate {
                        pos: (global_pos & (BUCKET_REGION - 1)) as u16,
                        logp,
                    },
                );
            }
        }
        return;
    }

    // Compute transformed root R' = numer * denom^{-1} mod p
    let inv = match crate::arith::mod_inverse(denom as u64, p) {
        Some(v) => v,
        None => return,
    };
    let r_prime = ((numer as u128 * inv as u128) % p as u128) as u64;

    // Quick hit check: if r_prime == 0 and p >= half_width, no hits in sieve area.
    if r_prime == 0 && p >= half_i as u64 {
        return;
    }

    let p_usize = p as usize;
    let mut start_mod_p = (half_i as u64) % p;
    let step = r_prime % p;

    for j in 0..max_j {
        let row_base = j * sieve_width;
        let mut i_pos = start_mod_p as usize;

        while i_pos < sieve_width {
            let global_pos = row_base + i_pos;
            buckets.push(
                global_pos >> LOG_BUCKET_REGION,
                BucketUpdate {
                    pos: (global_pos & (BUCKET_REGION - 1)) as u16,
                    logp,
                },
            );
            i_pos += p_usize;
        }

        start_mod_p += step;
        if start_mod_p >= p { start_mod_p -= p; }
    }
}

/// Evaluate |F(a, b)| for the homogeneous algebraic polynomial using f64.
///
/// F(a, b) = c_0 * b^d + c_1 * a * b^{d-1} + ... + c_d * a^d
fn eval_homogeneous_norm_f64(f_coeffs: &[i64], a: f64, b: f64, d: usize) -> f64 {
    if f_coeffs.is_empty() {
        return 0.0;
    }
    if b.abs() > a.abs() * 1e-15 {
        let t = a / b;
        let mut poly = f_coeffs[d] as f64;
        for idx in (0..d).rev() {
            poly = poly * t + f_coeffs[idx] as f64;
        }
        let abs_b = b.abs();
        let abs_bd = if d == 4 {
            let b2 = abs_b * abs_b;
            b2 * b2
        } else if d == 5 {
            let b2 = abs_b * abs_b;
            b2 * b2 * abs_b
        } else {
            abs_b.powi(d as i32)
        };
        (poly * abs_bd).abs()
    } else {
        let c_d = f_coeffs[d] as f64;
        let abs_a = a.abs();
        let a_d = if d == 4 {
            let a2 = abs_a * abs_a;
            a2 * a2
        } else if d == 5 {
            let a2 = abs_a * abs_a;
            a2 * a2 * abs_a
        } else {
            abs_a.powi(d as i32)
        };
        (c_d * a_d).abs()
    }
}

#[inline]
fn log_norm_to_u8(abs_f: f64, scale: f64) -> u8 {
    if abs_f < 1.0 {
        0
    } else {
        // fast log2 approximation using exponent of IEEE 754 f64
        let bits = abs_f.to_bits();
        let exponent = ((bits >> 52) & 0x7ff) as i64 - 1023;
        let mantissa = (bits & 0xfffffffffffff) as f64 * (1.0 / 4503599627370496.0); // 1.0 / 2^52
        let fast_log2 = exponent as f64 + mantissa;
        let v = fast_log2 * scale;
        if v >= 255.0 {
            255
        } else {
            v as u8
        }
    }
}

/// Compute |a - b*m| (rational norm).
fn compute_rat_norm(a: i64, b: u64, m: u64) -> u64 {
    let val = a as i128 - (b as i128) * (m as i128);
    val.unsigned_abs() as u64
}

/// Compute |F(a, b)| (algebraic norm) using the homogeneous polynomial.
///
/// F(a, b) = c_0 * b^d + c_1 * a * b^{d-1} + ... + c_d * a^d
///
/// Uses i128 arithmetic for exact computation.
fn compute_alg_norm(a: i64, b: u64, f_coeffs: &[i64]) -> Option<u128> {
    if f_coeffs.is_empty() {
        return Some(0);
    }
    let d = f_coeffs.len() - 1;

    if b == 0 {
        // F(a, 0) = c_d * a^d
        let a128 = a as i128;
        let mut a_pow: i128 = 1;
        for _ in 0..d {
            a_pow = a_pow.checked_mul(a128)?;
        }
        let v = (f_coeffs[d] as i128).checked_mul(a_pow)?;
        return Some(v.unsigned_abs());
    }

    // General case: use Horner evaluation
    // F(a, b) = b^d * f(a/b) but we compute with integers:
    // F(a, b) = c_0 * b^d + c_1 * a * b^{d-1} + ... + c_d * a^d
    let a128 = a as i128;
    let b128 = b as i128;

    let mut result: i128 = 0;
    let mut a_pow: i128 = 1;
    let mut b_pow: i128 = {
        let mut bp: i128 = 1;
        for _ in 0..d {
            bp = bp.checked_mul(b128)?;
        }
        bp
    };

    for k in 0..=d {
        let term = (f_coeffs[k] as i128)
            .checked_mul(a_pow)?
            .checked_mul(b_pow)?;
        result = result.checked_add(term)?;
        a_pow = a_pow.checked_mul(a128)?;
        if k < d {
            // b_pow goes from b^d down to b^0
            b_pow /= b128;
        }
    }

    Some(result.unsigned_abs())
}

/// Build a Relation from cofactorization results, or None if not smooth enough.
fn build_relation(
    a: i64,
    b: u64,
    special_q: Option<(u64, u64)>,
    rat_result: CofactResult,
    alg_result: CofactResult,
) -> Option<Relation> {
    let max_lp_keys = std::env::var("RUST_NFS_MAX_LP_KEYS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(2usize);

    let (rat_factors, rat_lps): (Vec<(u32, u8)>, Vec<u64>) = match rat_result {
        CofactResult::Smooth(f) => (f, Vec::new()),
        CofactResult::OneLargePrime(f, lp) => (f, vec![lp]),
        CofactResult::TwoLargePrimes(f, lp1, lp2) => (f, vec![lp1, lp2]),
        CofactResult::NotSmooth => return None,
    };
    let (alg_factors, alg_lps): (Vec<(u32, u8)>, Vec<u64>) = match alg_result {
        CofactResult::Smooth(f) => (f, Vec::new()),
        CofactResult::OneLargePrime(f, lp) => (f, vec![lp]),
        CofactResult::TwoLargePrimes(f, lp1, lp2) => (f, vec![lp1, lp2]),
        CofactResult::NotSmooth => return None,
    };

    // Keep LP keys with odd multiplicity only (GF(2) parity).
    let mut lp_set = std::collections::HashSet::<LpKey>::new();
    for p in rat_lps {
        let key = LpKey::Rational(p);
        if !lp_set.remove(&key) {
            lp_set.insert(key);
        }
    }
    for p in alg_lps {
        let Some(key) = compute_alg_lp_key(a, b, p) else {
            return None;
        };
        if !lp_set.remove(&key) {
            lp_set.insert(key);
        }
    }
    let mut lp_keys: Vec<LpKey> = lp_set.into_iter().collect();
    lp_keys.sort_unstable();
    if lp_keys.len() > max_lp_keys {
        return None;
    }

    // Legacy compatibility fields: only set when there is exactly one key per side.
    let rat_only: Vec<u64> = lp_keys
        .iter()
        .filter_map(|k| match k {
            LpKey::Rational(p) => Some(*p),
            _ => None,
        })
        .collect();
    let alg_only: Vec<u64> = lp_keys
        .iter()
        .filter_map(|k| match k {
            LpKey::Algebraic(p, _) => Some(*p),
            _ => None,
        })
        .collect();
    let rat_cofactor = if rat_only.len() == 1 { rat_only[0] } else { 0 };
    let alg_cofactor = if alg_only.len() == 1 { alg_only[0] } else { 0 };

    Some(Relation {
        a,
        b,
        rational_factors: rat_factors,
        algebraic_factors: alg_factors,
        rational_sign_negative: a < 0,
        algebraic_sign_negative: false,
        special_q,
        rat_cofactor,
        alg_cofactor,
        lp_keys,
    })
}

fn compute_alg_lp_key(a: i64, b: u64, p: u64) -> Option<LpKey> {
    if p < 2 {
        return None;
    }
    let b_mod_p = b % p;
    if b_mod_p == 0 {
        return Some(LpKey::Algebraic(p, p));
    }
    let b_inv = match gnfs::arith::mod_inverse_u64(b_mod_p, p) {
        Some(v) => v,
        None => return Some(LpKey::Algebraic(p, p)),
    };
    let a_mod_p = (a as i128).rem_euclid(p as i128) as u64;
    let r = ((a_mod_p as u128 * b_inv as u128) % p as u128) as u64;
    Some(LpKey::Algebraic(p, r))
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_rat_norm() {
        assert_eq!(compute_rat_norm(100, 1, 50), 50); // |100 - 1*50|
        assert_eq!(compute_rat_norm(-10, 2, 5), 20); // |-10 - 2*5| = 20
        assert_eq!(compute_rat_norm(0, 1, 5), 5); // |0 - 5| = 5
        assert_eq!(compute_rat_norm(5, 1, 5), 0); // |5 - 5| = 0
    }

    #[test]
    fn test_compute_alg_norm() {
        // f(x) = x^2 + 1 -> coeffs = [1, 0, 1]
        // F(a, b) = c_0 * b^2 + c_1 * a * b + c_2 * a^2 = b^2 + a^2
        // F(3, 1) = 1 + 9 = 10
        assert_eq!(compute_alg_norm(3, 1, &[1, 0, 1]), Some(10));
        // F(2, 3) = 9 + 0 + 4 = 13
        assert_eq!(compute_alg_norm(2, 3, &[1, 0, 1]), Some(13));
    }

    #[test]
    fn test_compute_alg_norm_b_zero() {
        // f(x) = x^2 + 1, coeffs = [1, 0, 1]
        // F(5, 0) = c_2 * 5^2 = 25
        assert_eq!(compute_alg_norm(5, 0, &[1, 0, 1]), Some(25));
    }

    #[test]
    fn test_compute_alg_norm_degree3() {
        // f(x) = x^3 + 2x + 1, coeffs = [1, 2, 0, 1]
        // F(a, b) = b^3 + 2*a*b^2 + 0*a^2*b + a^3
        // F(1, 1) = 1 + 2 + 0 + 1 = 4
        assert_eq!(compute_alg_norm(1, 1, &[1, 2, 0, 1]), Some(4));
        // F(2, 1) = 1 + 4 + 0 + 8 = 13
        assert_eq!(compute_alg_norm(2, 1, &[1, 2, 0, 1]), Some(13));
    }

    #[test]
    fn test_eval_homogeneous_f64() {
        // f = [1.0, 0.0, 1.0] -> F(a,b) = b^2 + a^2
        let v = eval_homogeneous_norm_f64(&[1, 0, 1], 3.0, 4.0, 2);
        assert!((v - 25.0).abs() < 0.01); // 9 + 16 = 25
    }

    #[test]
    fn test_sieve_finds_relations() {
        // Small test: verify the function runs without panicking
        let f_coeffs = vec![1i64, 2, 0, 1]; // x^3 + 2x + 1
        let m = 100u64;
        let params = NfsParams::c30();
        let rat_fb = FactorBase::new(&[-(m as i64), 1], params.lim0, 1.442);
        let alg_fb = FactorBase::new(&f_coeffs, params.lim1, 1.442);

        let result = sieve_specialq(
            &f_coeffs,
            m,
            &rat_fb,
            &alg_fb,
            &params,
            params.qmin,
            params.qrange,
            None,
            None,
        );
        // Just check it runs and produces some output
        assert!(result.special_qs_processed > 0);
    }

    #[test]
    fn test_sieve_result_timing() {
        let f_coeffs = vec![1i64, 0, 1]; // x^2 + 1
        let m = 50u64;
        let mut params = NfsParams::c30();
        // Use tiny range to keep test fast
        params.qmin = 100;
        params.qrange = 50;
        params.rels_wanted = 0; // don't wait for relations

        let rat_fb = FactorBase::new(&[-(m as i64), 1], params.lim0, 1.442);
        let alg_fb = FactorBase::new(&f_coeffs, params.lim1, 1.442);

        let result = sieve_specialq(
            &f_coeffs,
            m,
            &rat_fb,
            &alg_fb,
            &params,
            params.qmin,
            params.qrange,
            None,
            None,
        );
        assert!(result.total_ms >= 0.0);
        assert!(result.sieve_ms >= 0.0);
        assert!(result.cofactor_ms >= 0.0);
    }

    #[test]
    fn test_scatter_bucket_updates_projective_rows_reach_bucket_path() {
        let qlat = QLattice {
            a0: 5,
            b0: 1,
            a1: 1,
            b1: 0,
        };
        let mut buckets = BucketArray::new(1, 64);
        scatter_bucket_updates_for_prime(5, 0, 7, &qlat, 4, &mut buckets, 8, 10, 4);

        let updates = buckets.updates_for_bucket(0);
        // denom == 0 and numer != 0 => every p-th row is hit in full.
        assert_eq!(updates.len(), 16);
        assert!(updates.iter().all(|u| u.log_prime() == 7));
    }

    #[test]
    fn test_build_relation_smooth() {
        let rat = CofactResult::Smooth(vec![(0, 1), (1, 2)]);
        let alg = CofactResult::Smooth(vec![(2, 3)]);
        let rel = build_relation(5, 3, None, rat, alg).unwrap();
        assert_eq!(rel.a, 5);
        assert_eq!(rel.b, 3);
        assert_eq!(rel.rat_cofactor, 0);
        assert_eq!(rel.alg_cofactor, 0);
        assert!(rel.lp_keys.is_empty());
        assert!(!rel.rational_sign_negative);
    }

    #[test]
    fn test_build_relation_not_smooth() {
        let rat = CofactResult::NotSmooth;
        let alg = CofactResult::Smooth(vec![]);
        assert!(build_relation(1, 1, None, rat, alg).is_none());
    }

    #[test]
    fn test_build_relation_negative_a() {
        let rat = CofactResult::Smooth(vec![]);
        let alg = CofactResult::Smooth(vec![]);
        let rel = build_relation(-7, 2, None, rat, alg).unwrap();
        assert!(rel.rational_sign_negative);
    }

    #[test]
    fn test_build_relation_one_large_prime() {
        let rat = CofactResult::OneLargePrime(vec![(0, 1)], 12347);
        let alg = CofactResult::Smooth(vec![(1, 1)]);
        let rel = build_relation(10, 3, None, rat, alg).unwrap();
        assert_eq!(rel.rat_cofactor, 12347);
        assert_eq!(rel.alg_cofactor, 0);
        assert_eq!(rel.lp_keys, vec![LpKey::Rational(12347)]);
    }

    #[test]
    fn test_build_relation_two_large_primes_kept() {
        let rat = CofactResult::TwoLargePrimes(vec![(0, 1)], 1009, 1013);
        let alg = CofactResult::Smooth(vec![(1, 1)]);
        let rel = build_relation(10, 3, None, rat, alg).unwrap();
        assert_eq!(
            rel.lp_keys,
            vec![LpKey::Rational(1009), LpKey::Rational(1013)]
        );
    }
}
