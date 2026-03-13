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
use crate::cofactor::{self, CofactResult, CofactorConfig};
use crate::factorbase::{self, FactorBase};
use crate::lp_key::LpKey;
use crate::params::NfsParams;
use crate::relation::Relation;
use rayon::prelude::*;
use std::collections::HashMap;

use self::bucket::{BucketArray, BucketUpdate, BUCKET_REGION, LOG_BUCKET_REGION};
use self::lattice::{reduce_qlattice, QLattice};
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
    /// Sub-step: time spent in small sieve precomputation (ms).
    pub small_precomp_ms: f64,
    /// Sub-step: time spent in FK scatter for algebraic side (ms).
    pub fk_scatter_alg_ms: f64,
    /// Sub-step: time spent in FK scatter for rational side (ms).
    pub fk_scatter_rat_ms: f64,
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
    let mut total_small_precomp_ns = 0u64;
    let mut total_fk_scatter_alg_ns = 0u64;
    let mut total_fk_scatter_rat_ns = 0u64;
    let mut sq_count = 0usize;
    let norm_block = std::env::var("RUST_NFS_NORM_BLOCK")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(16usize);
    let verbose_sq = std::env::var("RUST_NFS_VERBOSE_SQ")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

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
    // Pre-compute cofactoring prime lists once (avoids thousands of sieve_primes per-survivor).
    let cofact_config_rat = CofactorConfig::new(params.lpb0);
    let cofact_config_alg = CofactorConfig::new(params.lpb1);

    let est_updates = estimate_updates_per_bucket(
        &rat_fb.primes,
        &alg_fb.primes,
        &alg_fb.roots,
        bucket_thresh,
        0,
    );
    let updates_per_bucket = (est_updates * 2).max(1024);

    for chunk in qr_pairs.chunks(batch_size.max(1)) {
        let chunk_results: Vec<(Vec<Relation>, usize, u64, u64, u64, u64, u64, u64, u64, u64, u64)> = chunk
            .par_iter()
            .copied()
            .map_init(
                || {
                    let rat_buckets = BucketArray::new(n_buckets.max(1), updates_per_bucket);
                    let alg_buckets = BucketArray::new(n_buckets.max(1), updates_per_bucket);
                    let rat_sieve = vec![0u8; BUCKET_REGION];
                    let alg_sieve = vec![0u8; BUCKET_REGION];
                    let survivors = Vec::with_capacity(1024);
                    let fk_entries: Vec<(u64, u64, u8)> = Vec::with_capacity(alg_large_indices.len() * 2 + rat_large_indices.len());
                    let walk_buf: Vec<FkWalkParams> = Vec::with_capacity(alg_large_indices.len() * 2 + rat_large_indices.len());
                    (rat_buckets, alg_buckets, rat_sieve, alg_sieve, survivors, fk_entries, walk_buf)
                },
                |(rat_buckets, alg_buckets, rat_sieve, alg_sieve, survivors_this_sq, fk_entries, walk_buf), (q, r)| {
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
                    let t_small_precomp = std::time::Instant::now();
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

                    let small_precomp_ns = t_small_precomp.elapsed().as_nanos() as u64;

                    // 3. Scatter bucket updates (batch FK)

                    let t_fk_alg = std::time::Instant::now();
                    fk_entries.clear();
                    for &fb_idx in &alg_large_indices {
                        let p = alg_fb.primes[fb_idx];
                        if p == q {
                            continue;
                        }
                        let logp = alg_fb.log_p[fb_idx];
                        for &root in &alg_fb.roots[fb_idx] {
                            fk_entries.push((p, root, logp));
                        }
                    }
                    scatter_bucket_updates_fk_batch(
                        &fk_entries,
                        &qlat,
                        params.log_i,
                        &mut *alg_buckets,
                        sieve_width,
                        max_j,
                        half_i,
                        &mut *walk_buf,
                    );

                    let fk_scatter_alg_ns = t_fk_alg.elapsed().as_nanos() as u64;

                    let t_fk_rat = std::time::Instant::now();
                    fk_entries.clear();
                    for &fb_idx in &rat_large_indices {
                        let p = rat_fb.primes[fb_idx];
                        if p == q {
                            continue;
                        }
                        fk_entries.push((p, rat_roots_by_index[fb_idx], rat_fb.log_p[fb_idx]));
                    }
                    scatter_bucket_updates_fk_batch(
                        &fk_entries,
                        &qlat,
                        params.log_i,
                        &mut *rat_buckets,
                        sieve_width,
                        max_j,
                        half_i,
                        &mut *walk_buf,
                    );
                    let fk_scatter_rat_ns = t_fk_rat.elapsed().as_nanos() as u64;

                    let bucket_setup_elapsed = sieve_start.elapsed().as_nanos() as u64;

                    // 4. Process each bucket region (sequentially)
                    let region_start_time = std::time::Instant::now();
                    let mut norm_ns: u64 = 0;
                    let mut small_sieve_ns: u64 = 0;
                    let mut bucket_apply_ns: u64 = 0;
                    let rat_bound = ((params.sieve_mfb0 as f64) * scale).min(255.0) as u8;
                    let alg_bound = ((params.sieve_mfb1 as f64) * scale).min(255.0) as u8;
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

                        let rat_result = cofactor::cofactorize_with_config(
                            rat_norm,
                            &rat_fb.trial_divisors,
                            params.lpb0,
                            params.mfb0,
                            params.lim0,
                            &cofact_config_rat,
                        );

                        // Early exit: skip expensive u128 algebraic cofactoring
                        // when rational side already failed (~91% of survivors).
                        if matches!(rat_result, cofactor::CofactResult::NotSmooth) {
                            continue;
                        }

                        let alg_norm_to_factor = if alg_norm_reduced == 0 {
                            1
                        } else {
                            alg_norm_reduced
                        };
                        // Fast path: dispatch to u64 cofactoring when the
                        // reduced algebraic norm fits, avoiding u128 trial
                        // division overhead.
                        let alg_result = if alg_norm_to_factor <= u64::MAX as u128 {
                            cofactor::cofactorize_with_config(
                                alg_norm_to_factor as u64,
                                &alg_fb.trial_divisors,
                                params.lpb1,
                                params.mfb1,
                                params.lim1,
                                &cofact_config_alg,
                            )
                        } else {
                            cofactor::cofactorize_u128_with_config(
                                alg_norm_to_factor,
                                &alg_fb.trial_divisors,
                                params.lpb1,
                                params.mfb1,
                                params.lim1,
                                &cofact_config_alg,
                            )
                        };

                        if let Some(rel) =
                            build_relation(a, b, Some((q, r)), rat_result, alg_result)
                        {
                            local_rels.push(rel);
                        }
                    }

                    let cofact_elapsed = cofact_start.elapsed().as_nanos() as u64;

                    if verbose_sq {
                        let sq_ms = sieve_start.elapsed().as_secs_f64() * 1000.0;
                        let n_rels = local_rels.len();
                        let rels_per_sec = if sq_ms > 0.0 {
                            n_rels as f64 / (sq_ms / 1000.0)
                        } else {
                            0.0
                        };
                        eprintln!(
                            "  sq={}: {} rels in {:.1}ms ({:.1} rels/s)",
                            q, n_rels, sq_ms, rels_per_sec
                        );
                    }

                    (
                        local_rels,
                        local_survivors,
                        bucket_setup_elapsed,
                        region_scan_elapsed,
                        cofact_elapsed,
                        norm_ns,
                        small_sieve_ns,
                        bucket_apply_ns,
                        small_precomp_ns,
                        fk_scatter_alg_ns,
                        fk_scatter_rat_ns,
                    )
                },
            )
            .collect();

        for (rels, survivors, bucket_setup_ns, region_scan_ns, cofact_ns, n_ns, ss_ns, ba_ns, sp_ns, fk_alg_ns, fk_rat_ns) in chunk_results {
            all_relations.extend(rels);
            total_survivors += survivors;
            total_bucket_setup_ns += bucket_setup_ns;
            total_region_scan_ns += region_scan_ns;
            total_cofact_ns += cofact_ns;
            total_norm_ns += n_ns;
            total_small_sieve_ns += ss_ns;
            total_bucket_apply_ns += ba_ns;
            total_small_precomp_ns += sp_ns;
            total_fk_scatter_alg_ns += fk_alg_ns;
            total_fk_scatter_rat_ns += fk_rat_ns;
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
        eprintln!(
            "  sieve-profile: small_precomp={:.0}ms fk_scatter_alg={:.0}ms fk_scatter_rat={:.0}ms (of setup={:.0}ms)",
            total_small_precomp_ns as f64 / 1_000_000.0,
            total_fk_scatter_alg_ns as f64 / 1_000_000.0,
            total_fk_scatter_rat_ns as f64 / 1_000_000.0,
            total_bucket_setup_ns as f64 / 1_000_000.0,
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
        small_precomp_ms: total_small_precomp_ns as f64 / 1_000_000.0,
        fk_scatter_alg_ms: total_fk_scatter_alg_ns as f64 / 1_000_000.0,
        fk_scatter_rat_ms: total_fk_scatter_rat_ns as f64 / 1_000_000.0,
    }
}

// ===========================================================================
// Line Sieve: direct per-line sieving (no bucket scatter overhead)
// ===========================================================================

/// Precomputed per-prime sieve info for the line sieve.
///
/// Stores the transformed root in q-lattice (i,j) coordinates so that
/// for each line j the starting offset can be computed as
/// `(half_i + root_i * j) mod p`.
struct LineSieveEntry {
    p: u64,
    logp: u8,
    /// Transformed root in (i,j)-space.  For row j, the first hit in the
    /// sieve row is at position `(half_i + root_i * j_unsigned) mod p`.
    root_i: u64,
    /// Non-zero for projective roots: every `projective_row_period`-th row
    /// has all positions hit.
    projective_row_period: u64,
}

/// Running offset for a single prime across sieve lines.
///
/// Instead of computing `(half_i + root_i * j) mod p` from scratch each
/// line, we maintain a running offset that advances by `step = root_i`
/// when j increases by 1.  This converts a modular multiply per (prime,
/// line) pair into a single addition + comparison.
struct LineSieveRunning {
    p: u32,
    logp: u8,
    /// Current starting position in the sieve row (0 <= offset < p).
    offset: u32,
    /// Amount to advance when moving to the next line: root_i mod p.
    step: u32,
}

/// Run the line sieve for a batch of special-q primes.
///
/// The line sieve processes one sieve row at a time.  For each row j in
/// `[0, J)`:
///
///  1. Initialise the rational and algebraic sieve arrays (length 2I) with
///     approximate log-norms.
///  2. For **every** FB prime, stride through the row at step p subtracting
///     log(p) at each hit.
///  3. Scan for survivors (cells where both sides are below threshold).
///  4. Cofactorize survivors to produce relations.
///
/// Because the inner loop touches each sieve cell in L1-friendly order and
/// avoids the per-SQ FK setup phase, the line sieve trades some per-prime
/// overhead (computing start offsets per line) for far better cache locality
/// and no O(FB) scatter setup.
pub fn line_sieve_specialq(
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
    let max_j = params.sieve_half_width().max(1) as usize;
    let scale = rat_fb.scale;

    // Rational polynomial: g(x) = x - m
    let g0 = -(m as f64);
    let g1 = 1.0f64;

    let d = f_coeffs.len().saturating_sub(1);

    // Pre-compute cofactoring config once.
    let cofact_config_rat = CofactorConfig::new(params.lpb0);
    let cofact_config_alg = CofactorConfig::new(params.lpb1);

    let rat_bound = ((params.sieve_mfb0 as f64) * scale).min(255.0) as u8;
    let alg_bound = ((params.sieve_mfb1 as f64) * scale).min(255.0) as u8;

    let verbose_sq = std::env::var("RUST_NFS_VERBOSE_SQ")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    let q_end = q_start + q_range;
    let sq_primes: Vec<u64> = sieve_primes(q_end)
        .into_iter()
        .filter(|&p| p >= q_start)
        .collect();

    // --- Root enumeration (same as scatter sieve) ---
    let root_enum_start = std::time::Instant::now();
    let mut roots_from_fb = 0usize;
    let mut roots_fallback = 0usize;
    let mut qr_pairs = Vec::new();
    let mut sq_count = 0usize;
    for &q in &sq_primes {
        sq_count += 1;
        if let Ok(idx) = alg_fb.primes.binary_search(&q) {
            roots_from_fb += 1;
            for &r in &alg_fb.roots[idx] {
                qr_pairs.push((q, r));
            }
            continue;
        }
        if let Some(cache) = sq_root_cache {
            if let Some(cached_roots) = cache.get(&q) {
                roots_from_fb += 1;
                for &r in cached_roots {
                    qr_pairs.push((q, r));
                }
                continue;
            }
        }
        roots_fallback += 1;
        let q_roots = factorbase::find_roots_mod_p(f_coeffs, q);
        for r in q_roots {
            qr_pairs.push((q, r));
        }
    }
    let root_enum_ms = root_enum_start.elapsed().as_secs_f64() * 1000.0;

    // --- Build FB entry lists (all primes, not partitioned by size) ---
    // Rational FB entries: one root per prime (root = m mod p).
    let rat_entries_all: Vec<LineSieveEntry> = rat_fb
        .primes
        .iter()
        .enumerate()
        .map(|(idx, &p)| LineSieveEntry {
            p,
            logp: rat_fb.log_p[idx],
            root_i: 0,              // placeholder; computed per-q
            projective_row_period: 0,
        })
        .collect();

    // Algebraic FB entries: multiple roots per prime.
    let alg_entries_all: Vec<(usize, usize)> = alg_fb
        .primes
        .iter()
        .enumerate()
        .flat_map(|(idx, _)| alg_fb.roots[idx].iter().enumerate().map(move |(ri, _)| (idx, ri)))
        .collect();

    let batch_size = std::env::var("RUST_NFS_SQ_BATCH_SIZE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(64usize);

    let mut all_relations = Vec::new();
    let mut total_survivors = 0usize;
    let mut total_setup_ns = 0u64;
    let mut total_sieve_ns = 0u64;
    let mut total_cofact_ns = 0u64;

    for chunk in qr_pairs.chunks(batch_size.max(1)) {
        let chunk_results: Vec<(Vec<Relation>, usize, u64, u64, u64)> = chunk
            .par_iter()
            .copied()
            .map_init(
                || {
                    let rat_sieve = vec![0u8; sieve_width];
                    let alg_sieve = vec![0u8; sieve_width];
                    let survivors: Vec<(i64, u64)> = Vec::with_capacity(1024);
                    let rat_xformed: Vec<LineSieveEntry> = Vec::with_capacity(rat_entries_all.len());
                    let alg_xformed: Vec<LineSieveEntry> = Vec::with_capacity(alg_entries_all.len());
                    let rat_running: Vec<LineSieveRunning> = Vec::new();
                    let alg_running: Vec<LineSieveRunning> = Vec::new();
                    (rat_sieve, alg_sieve, survivors, rat_xformed, alg_xformed,
                     rat_running, alg_running)
                },
                |(rat_sieve, alg_sieve, survivors_buf, rat_xformed, alg_xformed,
                  rat_running, alg_running), (q, r)| {
                    survivors_buf.clear();
                    let sq_start = std::time::Instant::now();

                    // 1. Reduce q-lattice
                    let skewness = 1.0;
                    let qlat = reduce_qlattice(q, r, skewness);

                    // 2. Precompute transformed roots and running offsets for all FB primes.
                    //    We initialise each running entry with offset = half_i mod p
                    //    (the starting position for j=0).  As we advance through lines
                    //    the offset is updated in-place by adding `step` (no multiply).
                    rat_xformed.clear();
                    for (idx, &p) in rat_fb.primes.iter().enumerate() {
                        if p == q { continue; }
                        let entry = transform_root_for_line_sieve(
                            p, m % p, rat_fb.log_p[idx], &qlat,
                        );
                        if entry.projective_row_period != 0 {
                            // Keep projective entries in the static list
                            rat_xformed.push(entry);
                        } else {
                            rat_xformed.push(entry);
                        }
                    }

                    alg_xformed.clear();
                    for &(fb_idx, root_idx) in &alg_entries_all {
                        let p = alg_fb.primes[fb_idx];
                        if p == q { continue; }
                        let root = alg_fb.roots[fb_idx][root_idx];
                        let entry = transform_root_for_line_sieve(
                            p, root, alg_fb.log_p[fb_idx], &qlat,
                        );
                        alg_xformed.push(entry);
                    }

                    // Build running offset arrays (reuse pre-allocated buffers).
                    let half_i_u64 = half_i as u64;

                    // All FB primes go into a single running-offset array.
                    // Small primes (p < sieve_width) stride multiple times per
                    // row; large primes (p >= sieve_width) hit 0-1 times via
                    // the same while loop.  Keeping everything in one vector
                    // avoids the overhead of iterating two separate arrays per
                    // row and lets the CPU prefetcher stay on one allocation.
                    rat_running.clear();
                    for e in rat_xformed.iter() {
                        if e.projective_row_period != 0 || e.p <= 1 { continue; }
                        rat_running.push(LineSieveRunning {
                            p: e.p as u32,
                            logp: e.logp,
                            offset: (half_i_u64 % e.p) as u32,
                            step: e.root_i as u32,
                        });
                    }

                    alg_running.clear();
                    for e in alg_xformed.iter() {
                        if e.projective_row_period != 0 || e.p <= 1 { continue; }
                        alg_running.push(LineSieveRunning {
                            p: e.p as u32,
                            logp: e.logp,
                            offset: (half_i_u64 % e.p) as u32,
                            step: e.root_i as u32,
                        });
                    }

                    if verbose_sq && q == sq_primes[0] {
                        eprintln!("  line_sieve FB: rat={} alg={} sieve_width={}",
                            rat_running.len(), alg_running.len(), sieve_width);
                    }

                    // Collect projective entries for separate handling
                    let rat_projective: Vec<&LineSieveEntry> = rat_xformed.iter()
                        .filter(|e| e.projective_row_period != 0)
                        .collect();
                    let alg_projective: Vec<&LineSieveEntry> = alg_xformed.iter()
                        .filter(|e| e.projective_row_period != 0)
                        .collect();

                    let setup_ns = sq_start.elapsed().as_nanos() as u64;

                    // 3. Line sieve: for each row j
                    let sieve_start = std::time::Instant::now();
                    let mut local_rels = Vec::new();

                    let a0f = qlat.a0 as f64;
                    let b0f = qlat.b0 as f64;
                    let a1f = qlat.a1 as f64;
                    let b1f = qlat.b1 as f64;

                    // Rational side: slope and intercept
                    let slope_rat = g1 * a0f + g0 * b0f;
                    let intercept_rat_per_j = g1 * a1f + g0 * b1f;

                    // Block-norm approximation: evaluate at block midpoint, fill block.
                    // Same approach as scatter sieve's norm_block=16.
                    let norm_block: usize = 16;

                    for j in 0..max_j {
                        // --- Init norms ---
                        let j_f = j as f64;
                        let intercept_rat = intercept_rat_per_j * j_f;

                        // Rational norms (linear: fast, block-approximated)
                        for block_start in (0..sieve_width).step_by(norm_block) {
                            let block_len = (sieve_width - block_start).min(norm_block);
                            let k_mid = block_start + (block_len / 2);
                            let i_val = (k_mid as i32) - (half_i as i32);
                            let f_val = slope_rat * (i_val as f64) + intercept_rat;
                            let v = log_norm_to_u8(f_val.abs(), scale);
                            rat_sieve[block_start..block_start + block_len].fill(v);
                        }

                        // Algebraic norms (degree d, block-approximated)
                        for block_start in (0..sieve_width).step_by(norm_block) {
                            let block_len = (sieve_width - block_start).min(norm_block);
                            let k_mid = block_start + (block_len / 2);
                            let i_val = (k_mid as i32) - (half_i as i32);
                            let i_f = i_val as f64;
                            let a = a0f * i_f + a1f * j_f;
                            let b = b0f * i_f + b1f * j_f;
                            let abs_f = eval_homogeneous_norm_f64(f_coeffs, a, b, d);
                            let v = log_norm_to_u8(abs_f, scale);
                            alg_sieve[block_start..block_start + block_len].fill(v);
                        }

                        // --- Sieve: subtract log(p) at hit positions using running offsets ---

                        // Small primes (p < sieve_width): stride through line
                        for entry in rat_running.iter_mut() {
                            let p = entry.p as usize;
                            let logp = entry.logp;
                            let mut pos = entry.offset as usize;

                            while pos < sieve_width {
                                unsafe {
                                    let cell = rat_sieve.get_unchecked_mut(pos);
                                    *cell = cell.saturating_sub(logp);
                                }
                                pos += p;
                            }

                            entry.offset += entry.step;
                            if entry.offset >= entry.p {
                                entry.offset -= entry.p;
                            }
                        }

                        for entry in alg_running.iter_mut() {
                            let p = entry.p as usize;
                            let logp = entry.logp;
                            let mut pos = entry.offset as usize;

                            while pos < sieve_width {
                                unsafe {
                                    let cell = alg_sieve.get_unchecked_mut(pos);
                                    *cell = cell.saturating_sub(logp);
                                }
                                pos += p;
                            }

                            entry.offset += entry.step;
                            if entry.offset >= entry.p {
                                entry.offset -= entry.p;
                            }
                        }

                        // Handle projective entries
                        for entry in &rat_projective {
                            if (j as u64) % entry.projective_row_period == 0 {
                                let logp = entry.logp;
                                for cell in rat_sieve[..sieve_width].iter_mut() {
                                    *cell = cell.saturating_sub(logp);
                                }
                            }
                        }
                        for entry in &alg_projective {
                            if (j as u64) % entry.projective_row_period == 0 {
                                let logp = entry.logp;
                                for cell in alg_sieve[..sieve_width].iter_mut() {
                                    *cell = cell.saturating_sub(logp);
                                }
                            }
                        }

                        // --- Scan for survivors ---
                        for k in 0..sieve_width {
                            if rat_sieve[k] <= rat_bound && alg_sieve[k] <= alg_bound {
                                let i = (k as i64) - half_i;
                                if j == 0 { continue; }

                                let a = qlat.a0 as i128 * i as i128 + qlat.a1 as i128 * j as i128;
                                let b = qlat.b0 as i128 * i as i128 + qlat.b1 as i128 * j as i128;

                                if b <= 0 || a == 0 { continue; }
                                if a.unsigned_abs() > i64::MAX as u128 || b > u64::MAX as i128 {
                                    continue;
                                }

                                // gcd(|a|, b) == 1 check
                                let mut u = a.unsigned_abs() as u64;
                                let mut v = b as u64;
                                while v != 0 {
                                    let t = v;
                                    v = u % v;
                                    u = t;
                                }
                                if u != 1 { continue; }

                                survivors_buf.push((a as i64, b as u64));
                            }
                        }
                    }

                    let local_survivors = survivors_buf.len();
                    let sieve_ns = sieve_start.elapsed().as_nanos() as u64;

                    // 4. Cofactorize survivors
                    let cofact_start = std::time::Instant::now();
                    for &(a, b) in survivors_buf.iter() {
                        let rat_norm = compute_rat_norm(a, b, m);
                        let Some(alg_norm) = compute_alg_norm(a, b, f_coeffs) else {
                            continue;
                        };

                        if rat_norm == 0 || alg_norm == 0 { continue; }

                        let mut alg_norm_reduced = alg_norm;
                        let q128 = q as u128;
                        if alg_norm_reduced % q128 != 0 { continue; }
                        alg_norm_reduced /= q128;

                        let rat_result = cofactor::cofactorize_with_config(
                            rat_norm,
                            &rat_fb.trial_divisors,
                            params.lpb0,
                            params.mfb0,
                            params.lim0,
                            &cofact_config_rat,
                        );

                        if matches!(rat_result, cofactor::CofactResult::NotSmooth) {
                            continue;
                        }

                        let alg_norm_to_factor = if alg_norm_reduced == 0 { 1 } else { alg_norm_reduced };
                        let alg_result = if alg_norm_to_factor <= u64::MAX as u128 {
                            cofactor::cofactorize_with_config(
                                alg_norm_to_factor as u64,
                                &alg_fb.trial_divisors,
                                params.lpb1,
                                params.mfb1,
                                params.lim1,
                                &cofact_config_alg,
                            )
                        } else {
                            cofactor::cofactorize_u128_with_config(
                                alg_norm_to_factor,
                                &alg_fb.trial_divisors,
                                params.lpb1,
                                params.mfb1,
                                params.lim1,
                                &cofact_config_alg,
                            )
                        };

                        if let Some(rel) = build_relation(a, b, Some((q, r)), rat_result, alg_result) {
                            local_rels.push(rel);
                        }
                    }
                    let cofact_ns = cofact_start.elapsed().as_nanos() as u64;

                    if verbose_sq {
                        let sq_ms = sq_start.elapsed().as_secs_f64() * 1000.0;
                        let n_rels = local_rels.len();
                        let rels_per_sec = if sq_ms > 0.0 {
                            n_rels as f64 / (sq_ms / 1000.0)
                        } else {
                            0.0
                        };
                        eprintln!(
                            "  line_sq={}: {} rels, {} survivors in {:.1}ms ({:.1} rels/s)",
                            q, n_rels, local_survivors, sq_ms, rels_per_sec
                        );
                    }

                    (local_rels, local_survivors, setup_ns, sieve_ns, cofact_ns)
                },
            )
            .collect();

        for (rels, survivors, setup_ns, sieve_ns, cofact_ns) in chunk_results {
            all_relations.extend(rels);
            total_survivors += survivors;
            total_setup_ns += setup_ns;
            total_sieve_ns += sieve_ns;
            total_cofact_ns += cofact_ns;
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

    let total_ms = start_time.elapsed().as_secs_f64() * 1000.0;
    let setup_ms = total_setup_ns as f64 / 1_000_000.0;
    let sieve_ms = total_sieve_ns as f64 / 1_000_000.0;
    let cofact_ms = total_cofact_ns as f64 / 1_000_000.0;

    SieveResult {
        relations: all_relations,
        special_qs_processed: sq_count,
        survivors_found: total_survivors,
        total_ms,
        root_enum_ms,
        roots_from_fb,
        roots_fallback,
        bucket_setup_ms: setup_ms,
        region_scan_ms: sieve_ms,
        sieve_ms: setup_ms + sieve_ms,
        cofactor_ms: cofact_ms,
        small_precomp_ms: 0.0,
        fk_scatter_alg_ms: 0.0,
        fk_scatter_rat_ms: 0.0,
    }
}

/// Transform an FB prime root through the q-lattice for the line sieve.
///
/// Returns a `LineSieveEntry` with the transformed root `root_i` in
/// (i,j) coordinates.  For row j the first hit is at position
/// `(half_i + root_i * j) mod p`, then every p-th position after that.
///
/// The transformation is:
///   root_i = -(a1 - R*b1) * (a0 - R*b0)^{-1}  mod p
///
/// where R is the polynomial root mod p.
fn transform_root_for_line_sieve(
    p: u64,
    root: u64,
    logp: u8,
    qlat: &QLattice,
) -> LineSieveEntry {
    if p <= 1 {
        return LineSieveEntry { p, logp, root_i: 0, projective_row_period: 0 };
    }

    let p_i128 = p as i128;

    // denom = (a0 - R*b0) mod p
    let denom = ((qlat.a0 as i128 - (root as i128) * (qlat.b0 as i128)) % p_i128 + p_i128) % p_i128;
    // numer = -(a1 - R*b1) mod p  =  (R*b1 - a1) mod p
    let numer = (((root as i128) * (qlat.b1 as i128) - qlat.a1 as i128) % p_i128 + p_i128) % p_i128;

    if denom == 0 {
        // Projective root
        let period = if numer == 0 { 1 } else { p };
        return LineSieveEntry {
            p,
            logp,
            root_i: 0,
            projective_row_period: period,
        };
    }

    let inv = match crate::arith::mod_inverse(denom as u64, p) {
        Some(v) => v,
        None => {
            return LineSieveEntry { p, logp, root_i: 0, projective_row_period: 0 };
        }
    };

    let root_i = ((numer as u128 * inv as u128) % p as u128) as u64;

    LineSieveEntry {
        p,
        logp,
        root_i,
        projective_row_period: 0,
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
/// Transform root R from (a,b)-space to (i,j)-space through the q-lattice.
///
/// Returns `Ok(r_prime)` for affine roots, `Err(is_projective)` for projective
/// roots (denom == 0) where `is_projective` indicates whether numer is also 0.
///
/// Uses i64 arithmetic (fast) when products fit, i128 as fallback.
#[inline(always)]
fn transform_root(p: u64, root: u64, qlat: &QLattice) -> Result<u64, bool> {
    // R' = (root*b1 - a1) * (a0 - root*b0)^{-1} mod p
    //
    // For typical NFS parameters (p < 100k, |qlat coeffs| < 1000),
    // root * b0 and root * b1 fit in i64 (< 10^8). Use i64 when safe.
    let r = root as i64;
    let (denom_raw, numer_raw) = if root < (1u64 << 31)
        && qlat.b0.unsigned_abs() < (1u64 << 31)
        && qlat.b1.unsigned_abs() < (1u64 << 31)
    {
        // Fast path: all products fit in i64 (no overflow for 31+31 < 62 bits)
        let d = qlat.a0 - r * qlat.b0;
        let n = r * qlat.b1 - qlat.a1;
        (d, n)
    } else {
        // Fallback: use i128 for large values
        let p_i128 = p as i128;
        let d = ((qlat.a0 as i128 - (root as i128) * (qlat.b0 as i128)) % p_i128 + p_i128)
            % p_i128;
        let n = (((root as i128) * (qlat.b1 as i128) - qlat.a1 as i128) % p_i128 + p_i128)
            % p_i128;
        (d as i64, n as i64)
    };

    let p_i64 = p as i64;
    let denom = ((denom_raw % p_i64) + p_i64) % p_i64;
    let numer = ((numer_raw % p_i64) + p_i64) % p_i64;

    if denom == 0 {
        return Err(numer == 0);
    }

    let inv = match crate::arith::mod_inverse(denom as u64, p) {
        Some(v) => v,
        None => return Ok(0), // treat as no-hit
    };
    // For typical NFS primes (p < 2^32), numer * inv < p^2 < 2^64,
    // so u64 multiplication suffices. Use u128 only for large p.
    let r_prime = if p < (1u64 << 32) {
        ((numer as u64).wrapping_mul(inv)) % p
    } else {
        ((numer as u64 as u128 * inv as u128) % p as u128) as u64
    };
    Ok(r_prime)
}

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
    // Coprimality pre-filter mask (same as FK function).
    let even_mask = 1usize | (1usize << (log_i + 1));

    let r_prime = match transform_root(p, root, qlat) {
        Err(both_zero) => {
            // Projective root: hits every cell in each p-th row
            let row_period = if both_zero { 1usize } else { p as usize };
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
        Ok(rp) => rp,
    };

    // Quick hit check: if r_prime == 0 and p >= half_width, no hits in sieve area.
    if r_prime == 0 && p >= half_i as u64 {
        return;
    }

    let p_usize = p as usize;
    let mut start_mod_p = (half_i as u64) % p;
    let step = r_prime % p;

    if p_usize > sieve_width {
        // Fast path: at most one hit per row (since stride p > row width).
        for j in 0..max_j {
            if (start_mod_p as usize) < sieve_width {
                let global_pos = j * sieve_width + start_mod_p as usize;
                if (global_pos & even_mask) != 0 {
                    buckets.push(
                        global_pos >> LOG_BUCKET_REGION,
                        BucketUpdate {
                            pos: (global_pos & (BUCKET_REGION - 1)) as u16,
                            logp,
                        },
                    );
                }
            }
            start_mod_p += step;
            if start_mod_p >= p { start_mod_p -= p; }
        }
    } else {
        for j in 0..max_j {
            let row_base = j * sieve_width;
            let mut i_pos = start_mod_p as usize;

            while i_pos < sieve_width {
                let global_pos = row_base + i_pos;
                if (global_pos & even_mask) != 0 {
                    buckets.push(
                        global_pos >> LOG_BUCKET_REGION,
                        BucketUpdate {
                            pos: (global_pos & (BUCKET_REGION - 1)) as u16,
                            logp,
                        },
                    );
                }
                i_pos += p_usize;
            }

            start_mod_p += step;
            if start_mod_p >= p { start_mod_p -= p; }
        }
    }
}

/// Scatter bucket updates using the Franke-Kleinjung lattice walk.
///
/// This is functionally equivalent to `scatter_bucket_updates_for_prime` but uses
/// a reduced p-lattice walk with two short basis vectors instead of row-by-row
/// modular arithmetic. The FK walk replaces per-row i128 modular arithmetic with
/// integer additions and comparisons per step.
///
/// # Algorithm
///
/// 1. Transform the root R through the q-lattice to get R' in (i,j)-space.
/// 2. Perform partial-GCD reduction on (p, 0) and (R', 1) to get two short
///    basis vectors with opposite i-signs.
/// 3. Encode the two vectors as 1D increments `inc = v * sieve_width + u` and
///    a third "sum" increment for dead-zone handling.
/// 4. Walk through the sieve region in increasing x order, emitting bucket
///    updates at each lattice point.
///
/// When both basis vectors keep the i-coordinate in bounds, the walk emits both
/// targets (smaller increment first, then the difference to the larger). When
/// neither individual vector works, the sum vector (step + warp) is used.
///
/// Falls back to `scatter_bucket_updates_for_prime` for projective roots,
/// degenerate partial-GCD outputs, and cases where the reduced vectors cannot
/// cover the full i-range (dead-zone condition).
#[allow(dead_code)]
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
    // Use shared root transform (i64 fast path when possible).
    let r_prime = match transform_root(p, root, qlat) {
        Err(_) => {
            // Projective root: delegate to old function.
            scatter_bucket_updates_for_prime(
                p, root, logp, qlat, log_i, buckets, sieve_width, max_j, half_i,
            );
            return;
        }
        Ok(rp) => rp,
    };

    // Quick hit check.
    if r_prime == 0 && p >= half_i as u64 {
        return;
    }

    let half_width = half_i;
    let p_i64 = p as i64;

    // --- Partial-GCD reduction ---
    //
    // Basis: (u0, v0) = (p, 0), (u1, v1) = (r', 1)
    // Reduce until |u1| < half_width (= I).
    let mut u0 = p_i64;
    let mut v0: i64 = 0;
    let mut u1 = r_prime as i64;
    let mut v1: i64 = 1;

    // Center r' into [-p/2, p/2).
    if u1 > p_i64 / 2 {
        u1 -= p_i64;
    }

    // Run the partial-GCD (truncated extended Euclidean).
    while u1 != 0 && u1.unsigned_abs() >= half_width as u64 {
        let q_div = u0 / u1;
        let new_u = u0 - q_div * u1;
        let new_v = v0 - q_div * v1;
        u0 = u1;
        v0 = v1;
        u1 = new_u;
        v1 = new_v;
    }

    // The FK walk requires both vectors to have |v| >= 1 so that each step
    // advances in j and x increases monotonically. This holds when the
    // partial-GCD performs at least one iteration (centered |r'| >= I).
    // Fall back for degenerate cases.
    if u1 == 0 || v0 == 0 || v1 == 0 {
        scatter_bucket_updates_for_prime(
            p, root, logp, qlat, log_i, buckets, sieve_width, max_j, half_i,
        );
        return;
    }

    // Normalize: ensure v > 0 for both vectors (flip sign if needed).
    if v1 < 0 {
        u1 = -u1;
        v1 = -v1;
    }
    if v0 < 0 {
        u0 = -u0;
        v0 = -v0;
    }

    // The partial-GCD with centered remainders naturally produces opposite
    // u-signs. Verify this and fall back if violated.
    if (u0 > 0) == (u1 > 0) || u0 == 0 || u1 == 0 {
        scatter_bucket_updates_for_prime(
            p, root, logp, qlat, log_i, buckets, sieve_width, max_j, half_i,
        );
        return;
    }

    // Identify step (short |u|, < I) and warp (long |u|) vectors.
    let (u_step, v_step, u_warp, v_warp) = if u1.unsigned_abs() < u0.unsigned_abs() {
        (u1, v1, u0, v0)
    } else {
        (u0, v0, u1, v1)
    };

    let sieve_w = sieve_width as i64;

    // Encode as 1D increments: inc = v * sieve_width + u.
    // Both must be positive (v * sieve_width >> |u|).
    let inc_step = v_step * sieve_w + u_step;
    let inc_warp = v_warp * sieve_w + u_warp;

    if inc_step <= 0 || inc_warp <= 0 {
        scatter_bucket_updates_for_prime(
            p, root, logp, qlat, log_i, buckets, sieve_width, max_j, half_i,
        );
        return;
    }

    // --- Dead-zone check (O(1)) ---
    //
    // The walk uses three vectors: step, warp, and their sum (step + warp).
    // At each position, the walk picks whichever keeps i in [-I, I).
    // A "dead zone" exists if some i-coordinate has no valid vector.
    //
    // step covers i in [s_lo, I) where s_lo = max(-I, -I - u_step)
    // warp covers i in [-I, w_hi) where w_hi = min(I, I - u_warp)
    // (these ranges represent values where the respective vector keeps i in bounds)
    //
    // The gap between step and warp (if any) is [w_hi, s_lo).
    // The sum vector fills this gap if its range covers [w_hi, s_lo).
    let u_sum = u_step + u_warp;

    let s_lo = (-half_width - u_step).max(-half_width);
    let w_hi = (half_width - u_warp).min(half_width);

    if w_hi < s_lo {
        // There is a gap [w_hi, s_lo). Check if the sum vector fills it.
        let m_lo = (-half_width - u_sum).max(-half_width);
        let m_hi = (half_width - u_sum).min(half_width);

        if m_lo > w_hi || m_hi < s_lo {
            // Sum does not fully cover the gap. Fall back.
            scatter_bucket_updates_for_prime(
                p, root, logp, qlat, log_i, buckets, sieve_width, max_j, half_i,
            );
            return;
        }
    }

    // --- FK walk ---
    //
    // Sort increments: inc_a < inc_b so that when both vectors are valid,
    // inc_a's target comes first in x-order.
    let (inc_a, u_a, inc_b, u_b) = if inc_step < inc_warp {
        (inc_step, u_step, inc_warp, u_warp)
    } else {
        (inc_warp, u_warp, inc_step, u_step)
    };

    let inc_diff = inc_b - inc_a;
    let u_diff = u_b - u_a;
    let inc_sum = inc_a + inc_b;

    let total_area = (sieve_width * max_j) as i64;
    let start_i_pos = (half_i as u64 % p) as i64;
    let start_ic = start_i_pos - half_width;

    // Coprimality pre-filter: skip positions where both i and j are even
    // (gcd(i,j) >= 2 means gcd(a,b) >= 2, useless for NFS).
    // Bit 0 of x = parity of i (since half_i is even for log_i >= 1).
    // Bit (log_i+1) of x = parity of j (since sieve_width = 2^(log_i+1)).
    let even_mask = 1i64 | (1i64 << (log_i + 1));

    let mut x = start_i_pos;
    let mut ic = start_ic;

    // Precompute single-comparison thresholds for a_ok/b_ok.
    // The partial-GCD guarantees u_a and u_b have opposite signs.
    // When u < 0: only the lower bound can be violated (ic + u >= -half_width).
    // When u > 0: only the upper bound can be violated (ic + u < half_width).
    // The dead-zone check + a_ok/b_ok selection guarantees ic stays in
    // [-half_width, half_width) throughout the walk, so emit bounds checks
    // are redundant.
    let a_thresh = if u_a < 0 { -half_width - u_a } else { half_width - u_a };
    let b_thresh = if u_b < 0 { -half_width - u_b } else { half_width - u_b };
    let u_a_neg = u_a < 0;

    while x < total_area {
        // Emit bucket update if position is coprime.
        if ic >= -half_width && ic < half_width && (x & even_mask) != 0 {
            let gpos = x as usize;
            buckets.push(
                gpos >> LOG_BUCKET_REGION,
                BucketUpdate {
                    pos: (gpos & (BUCKET_REGION - 1)) as u16,
                    logp,
                },
            );
        }

        // Single-comparison a_ok/b_ok: opposite-sign u vectors mean only
        // one side of the bounds can be violated per vector.
        let (a_ok, b_ok) = if u_a_neg {
            (ic >= a_thresh, ic < b_thresh)
        } else {
            (ic < a_thresh, ic >= b_thresh)
        };

        if a_ok && b_ok {
            x += inc_a;
            ic += u_a;
            // ic guaranteed in bounds by a_ok; only check x bound and coprimality.
            if x < total_area && (x & even_mask) != 0 {
                let gpos = x as usize;
                buckets.push(
                    gpos >> LOG_BUCKET_REGION,
                    BucketUpdate {
                        pos: (gpos & (BUCKET_REGION - 1)) as u16,
                        logp,
                    },
                );
            }
            x += inc_diff;
            ic += u_diff;
        } else if a_ok {
            x += inc_a;
            ic += u_a;
        } else if b_ok {
            x += inc_b;
            ic += u_b;
        } else {
            x += inc_sum;
            ic += u_sum;
        }
    }
}

/// Compact FK walk parameters for batch processing.
///
/// Pre-computed from root transform + partial-GCD; consumed by the FK walk loop.
struct FkWalkParams {
    /// Smaller 1D increment.
    inc_a: i64,
    /// Larger 1D increment.
    inc_b: i64,
    /// Difference increment (inc_b - inc_a).
    inc_diff: i64,
    /// Sum increment (inc_a + inc_b).
    inc_sum: i64,
    /// i-component of vector a.
    u_a: i64,
    /// i-component of vector b.
    u_b: i64,
    /// i-component of difference (u_b - u_a).
    u_diff: i64,
    /// i-component of sum (u_a + u_b).
    u_sum: i64,
    /// Threshold for a_ok check.
    a_thresh: i64,
    /// Threshold for b_ok check.
    b_thresh: i64,
    /// True if u_a < 0.
    u_a_neg: bool,
    /// Starting 1D position.
    start_x: i64,
    /// Starting i-coordinate (centered).
    start_ic: i64,
    /// Quantized log2(p).
    logp: u8,
}

/// Batch scatter bucket updates for multiple (prime, root, logp) triples.
///
/// Phase 1: Pre-compute all FK walk parameters (root transform + partial-GCD).
/// Phase 2: Execute all FK walks, emitting bucket updates.
///
/// This separation improves instruction cache utilization and branch prediction
/// compared to calling scatter_bucket_updates_fk per prime.
fn scatter_bucket_updates_fk_batch(
    entries: &[(u64, u64, u8)], // (prime, root, logp) triples
    qlat: &QLattice,
    log_i: u32,
    buckets: &mut BucketArray,
    sieve_width: usize,
    max_j: usize,
    half_i: i64,
    walk_buf: &mut Vec<FkWalkParams>,
) {
    let half_width = half_i;
    let sieve_w = sieve_width as i64;
    let total_area = (sieve_width * max_j) as i64;
    let even_mask = 1i64 | (1i64 << (log_i + 1));

    walk_buf.clear();

    // --- Phase 1: compute FK walk parameters for all primes ---
    for &(p, root, logp) in entries {
        // Pre-filter: primes below sieve_width will have |centered r'| < half_width,
        // so partial-GCD won't iterate and FK walk falls back anyway.
        // Skip the FK setup overhead and go directly to row-by-row scatter.
        if p < sieve_width as u64 {
            scatter_bucket_updates_for_prime(
                p, root, logp, qlat, log_i, buckets, sieve_width, max_j, half_i,
            );
            continue;
        }

        // Root transform
        let r_prime = match transform_root(p, root, qlat) {
            Err(_) => {
                // Projective root: delegate to fallback.
                scatter_bucket_updates_for_prime(
                    p, root, logp, qlat, log_i, buckets, sieve_width, max_j, half_i,
                );
                continue;
            }
            Ok(rp) => rp,
        };

        // Quick hit check.
        if r_prime == 0 && p >= half_i as u64 {
            continue;
        }

        let p_i64 = p as i64;

        // Partial-GCD reduction
        let mut u0 = p_i64;
        let mut v0: i64 = 0;
        let mut u1 = r_prime as i64;
        let mut v1: i64 = 1;

        if u1 > p_i64 / 2 {
            u1 -= p_i64;
        }

        while u1 != 0 && u1.unsigned_abs() >= half_width as u64 {
            let q_div = u0 / u1;
            let new_u = u0 - q_div * u1;
            let new_v = v0 - q_div * v1;
            u0 = u1;
            v0 = v1;
            u1 = new_u;
            v1 = new_v;
        }

        // Degenerate cases: fall back
        if u1 == 0 || v0 == 0 || v1 == 0 {
            scatter_bucket_updates_for_prime(
                p, root, logp, qlat, log_i, buckets, sieve_width, max_j, half_i,
            );
            continue;
        }

        // Normalize v > 0
        if v1 < 0 { u1 = -u1; v1 = -v1; }
        if v0 < 0 { u0 = -u0; v0 = -v0; }

        // Check opposite u-signs
        if (u0 > 0) == (u1 > 0) || u0 == 0 || u1 == 0 {
            scatter_bucket_updates_for_prime(
                p, root, logp, qlat, log_i, buckets, sieve_width, max_j, half_i,
            );
            continue;
        }

        // Identify step/warp
        let (u_step, v_step, u_warp, v_warp) = if u1.unsigned_abs() < u0.unsigned_abs() {
            (u1, v1, u0, v0)
        } else {
            (u0, v0, u1, v1)
        };

        let inc_step = v_step * sieve_w + u_step;
        let inc_warp = v_warp * sieve_w + u_warp;

        if inc_step <= 0 || inc_warp <= 0 {
            scatter_bucket_updates_for_prime(
                p, root, logp, qlat, log_i, buckets, sieve_width, max_j, half_i,
            );
            continue;
        }

        // Dead-zone check
        let u_sum = u_step + u_warp;
        let s_lo = (-half_width - u_step).max(-half_width);
        let w_hi = (half_width - u_warp).min(half_width);

        if w_hi < s_lo {
            let m_lo = (-half_width - u_sum).max(-half_width);
            let m_hi = (half_width - u_sum).min(half_width);
            if m_lo > w_hi || m_hi < s_lo {
                scatter_bucket_updates_for_prime(
                    p, root, logp, qlat, log_i, buckets, sieve_width, max_j, half_i,
                );
                continue;
            }
        }

        // Compute walk parameters
        let (inc_a, u_a, inc_b, u_b) = if inc_step < inc_warp {
            (inc_step, u_step, inc_warp, u_warp)
        } else {
            (inc_warp, u_warp, inc_step, u_step)
        };

        let start_i_pos = (half_i as u64 % p) as i64;

        walk_buf.push(FkWalkParams {
            inc_a,
            inc_b,
            inc_diff: inc_b - inc_a,
            inc_sum: inc_a + inc_b,
            u_a,
            u_b,
            u_diff: u_b - u_a,
            u_sum,
            a_thresh: if u_a < 0 { -half_width - u_a } else { half_width - u_a },
            b_thresh: if u_b < 0 { -half_width - u_b } else { half_width - u_b },
            u_a_neg: u_a < 0,
            start_x: start_i_pos,
            start_ic: start_i_pos - half_width,
            logp,
        });
    }

    // --- Phase 2: execute all FK walks ---
    for params in walk_buf.iter() {
        let logp = params.logp;
        let inc_a = params.inc_a;
        let inc_b = params.inc_b;
        let inc_diff = params.inc_diff;
        let inc_sum = params.inc_sum;
        let u_a = params.u_a;
        let u_b = params.u_b;
        let u_diff = params.u_diff;
        let u_sum = params.u_sum;
        let a_thresh = params.a_thresh;
        let b_thresh = params.b_thresh;
        let u_a_neg = params.u_a_neg;

        let mut x = params.start_x;
        let mut ic = params.start_ic;

        while x < total_area {
            if ic >= -half_width && ic < half_width && (x & even_mask) != 0 {
                let gpos = x as usize;
                buckets.push(
                    gpos >> LOG_BUCKET_REGION,
                    BucketUpdate {
                        pos: (gpos & (BUCKET_REGION - 1)) as u16,
                        logp,
                    },
                );
            }

            let (a_ok, b_ok) = if u_a_neg {
                (ic >= a_thresh, ic < b_thresh)
            } else {
                (ic < a_thresh, ic >= b_thresh)
            };

            if a_ok && b_ok {
                x += inc_a;
                ic += u_a;
                // ic is guaranteed in bounds here (a_ok verified this).
                // Only check x < total_area and coprimality.
                if x < total_area && (x & even_mask) != 0 {
                    let gpos = x as usize;
                    buckets.push(
                        gpos >> LOG_BUCKET_REGION,
                        BucketUpdate {
                            pos: (gpos & (BUCKET_REGION - 1)) as u16,
                            logp,
                        },
                    );
                }
                x += inc_diff;
                ic += u_diff;
            } else if a_ok {
                x += inc_a;
                ic += u_a;
            } else if b_ok {
                x += inc_b;
                ic += u_b;
            } else {
                x += inc_sum;
                ic += u_sum;
            }
        }
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

    // -----------------------------------------------------------------------
    // Coprimality pre-filter tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_coprime_filter_removes_even_pairs() {
        // Verify that even_mask correctly identifies positions where both i and j are even.
        let log_i = 8u32; // half_i = 256, sieve_width = 512
        let sieve_width = 1usize << (log_i + 1);
        let half_i = 1i64 << log_i;
        let even_mask = 1u64 | (1u64 << (log_i + 1));

        for x in 0..2048u64 {
            let i_offset = (x as usize) % sieve_width;
            let j = (x as usize) / sieve_width;
            let i = i_offset as i64 - half_i;

            let both_even = i % 2 == 0 && j as i64 % 2 == 0;
            let mask_says_coprime = (x & even_mask) != 0;

            // When both i and j are even, the mask should say non-coprime (filter it out).
            assert_eq!(
                !both_even, mask_says_coprime,
                "coprime mismatch at x={} (i={}, j={})", x, i, j
            );
        }
    }

    #[test]
    fn test_coprime_filter_various_log_i() {
        // Test the mask for several log_i values used in practice.
        for log_i in [4u32, 7, 8, 9, 10] {
            let sieve_width = 1usize << (log_i + 1);
            let half_i = 1i64 << log_i;
            let even_mask = 1u64 | (1u64 << (log_i + 1));

            for x in 0..(sieve_width as u64 * 4).min(4096) {
                let i_offset = (x as usize) % sieve_width;
                let j = (x as usize) / sieve_width;
                let i = i_offset as i64 - half_i;

                let both_even = i % 2 == 0 && j as i64 % 2 == 0;
                let mask_says_coprime = (x & even_mask) != 0;

                assert_eq!(
                    !both_even, mask_says_coprime,
                    "coprime mismatch at x={} (i={}, j={}, log_i={})", x, i, j, log_i
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // FK scatter comparison tests
    // -----------------------------------------------------------------------

    /// Extract the set of global positions from a BucketArray (across all buckets).
    fn extract_positions(buckets: &BucketArray) -> Vec<usize> {
        let mut positions = Vec::new();
        for b in 0..buckets.n_buckets() {
            let updates = buckets.updates_for_bucket(b);
            for u in updates {
                let global = (b << LOG_BUCKET_REGION) | (u.position() as usize);
                positions.push(global);
            }
        }
        positions.sort();
        positions
    }

    /// Run both scatter functions on the same inputs and compare output positions.
    fn compare_scatter(
        p: u64,
        root: u64,
        logp: u8,
        qlat: &QLattice,
        log_i: u32,
        sieve_width: usize,
        max_j: usize,
        half_i: i64,
    ) -> (Vec<usize>, Vec<usize>) {
        let total_area = sieve_width * max_j;
        let n_buckets = (total_area + BUCKET_REGION - 1) / BUCKET_REGION;
        let n_buckets = n_buckets.max(1);
        // Generous capacity
        let cap = (total_area / (p as usize).max(1) + 1) * 2 + 256;

        let mut buckets_old = BucketArray::new(n_buckets, cap.max(16));
        scatter_bucket_updates_for_prime(
            p, root, logp, qlat, log_i, &mut buckets_old, sieve_width, max_j, half_i,
        );

        let mut buckets_fk = BucketArray::new(n_buckets, cap.max(16));
        scatter_bucket_updates_fk(
            p, root, logp, qlat, log_i, &mut buckets_fk, sieve_width, max_j, half_i,
        );

        // Also test batch version
        let mut buckets_batch = BucketArray::new(n_buckets, cap.max(16));
        let entries = vec![(p, root, logp)];
        let mut walk_buf = Vec::new();
        scatter_bucket_updates_fk_batch(
            &entries, qlat, log_i, &mut buckets_batch, sieve_width, max_j, half_i, &mut walk_buf,
        );
        let batch_pos = extract_positions(&buckets_batch);

        let old_pos = extract_positions(&buckets_old);
        let fk_pos = extract_positions(&buckets_fk);

        assert_eq!(
            fk_pos, batch_pos,
            "FK batch mismatch for p={}, root={}", p, root
        );

        (old_pos, fk_pos)
    }

    #[test]
    fn test_fk_scatter_matches_old_small_prime_fallback() {
        // Small primes (p < sieve_width) should fall back to old function
        // and produce identical results.
        let qlat = reduce_qlattice(97, 30, 1.0);
        let log_i = 4; // half_i = 16, sieve_width = 32
        let half_i = 1i64 << log_i;
        let sieve_width = (2 * half_i) as usize;
        let max_j = half_i as usize;

        for root in [0u64, 1, 2, 3, 4] {
            let (old, fk) = compare_scatter(5, root, 7, &qlat, log_i, sieve_width, max_j, half_i);
            assert_eq!(
                old, fk,
                "FK mismatch for p=5, root={}, qlat={:?}",
                root, qlat
            );
        }
    }

    #[test]
    fn test_fk_scatter_matches_old_projective_root() {
        // Projective root case (denom == 0): FK should delegate to old function.
        let qlat = QLattice {
            a0: 5,
            b0: 1,
            a1: 1,
            b1: 0,
        };
        let log_i = 4;
        let half_i = 1i64 << log_i;
        let sieve_width = (2 * half_i) as usize;
        let max_j = 10;

        // p=5, root=0: denom = (5 - 0*1) mod 5 = 0
        let (old, fk) = compare_scatter(5, 0, 7, &qlat, log_i, sieve_width, max_j, half_i);
        assert_eq!(old, fk, "FK mismatch for projective root case");
    }

    #[test]
    fn test_fk_scatter_matches_old_prime_above_sieve_width() {
        // Primes above sieve_width: this is where FK walk should activate.
        // log_i = 4 => half_i = 16, sieve_width = 32
        // Primes >= 32 trigger FK walk (if partial-GCD produces valid vectors).
        let qlat = reduce_qlattice(97, 30, 1.0);
        let log_i = 4;
        let half_i = 1i64 << log_i;
        let sieve_width = (2 * half_i) as usize;
        let max_j = half_i as usize;

        // Test primes above sieve_width = 32
        let test_cases: Vec<(u64, Vec<u64>)> = vec![
            (37, vec![0, 1, 10, 20, 36]),
            (41, vec![0, 1, 15, 30, 40]),
            (43, vec![0, 1, 20, 42]),
            (53, vec![0, 1, 25, 52]),
            (61, vec![0, 1, 30, 50, 60]),
            (67, vec![0, 1, 33, 66]),
            (71, vec![0, 1, 35, 70]),
            (79, vec![0, 1, 40, 78]),
            (83, vec![0, 1, 41, 82]),
            (89, vec![0, 1, 44, 88]),
        ];

        for (p, roots) in &test_cases {
            for &root in roots {
                let (old, fk) =
                    compare_scatter(*p, root, 7, &qlat, log_i, sieve_width, max_j, half_i);
                assert_eq!(
                    old, fk,
                    "FK mismatch for p={}, root={}, qlat={:?}",
                    p, root, qlat
                );
            }
        }
    }

    #[test]
    fn test_fk_scatter_matches_old_larger_sieve_large_primes() {
        // Larger sieve with primes above sieve_width
        // log_i = 8 => half_i = 256, sieve_width = 512
        let qlat = reduce_qlattice(65537, 12345, 1.0);
        let log_i = 8;
        let half_i = 1i64 << log_i;
        let sieve_width = (2 * half_i) as usize;
        let max_j = half_i as usize;

        // Primes above sieve_width = 512
        let test_cases: Vec<(u64, Vec<u64>)> = vec![
            (521, vec![0, 1, 100, 300, 520]),
            (541, vec![0, 1, 200, 540]),
            (547, vec![0, 1, 273, 546]),
            (557, vec![0, 1, 278, 556]),
            (563, vec![0, 1, 281, 562]),
            (569, vec![0, 1, 284, 568]),
            (1009, vec![0, 1, 500, 1008]),
            (1021, vec![0, 1, 510, 1020]),
            (2003, vec![0, 1, 1001, 2002]),
            (4099, vec![0, 1, 2049, 4098]),
        ];

        for (p, roots) in &test_cases {
            for &root in roots {
                let (old, fk) =
                    compare_scatter(*p, root, 7, &qlat, log_i, sieve_width, max_j, half_i);
                assert_eq!(
                    old, fk,
                    "FK mismatch for p={}, root={} with log_i={}",
                    p, root, log_i
                );
            }
        }
    }

    #[test]
    fn test_fk_scatter_matches_old_various_qlattices() {
        // Test with various q-lattice configurations, primes above sieve_width
        let test_qlats = vec![
            reduce_qlattice(97, 30, 1.0),
            reduce_qlattice(101, 42, 1.0),
            reduce_qlattice(65537, 12345, 1.0),
            reduce_qlattice(1009, 500, 2.0),
        ];

        // log_i = 4 => sieve_width = 32
        let log_i = 4;
        let half_i = 1i64 << log_i;
        let sieve_width = (2 * half_i) as usize;
        let max_j = half_i as usize;

        let primes_and_roots: Vec<(u64, u64)> = vec![
            (37, 10),
            (41, 20),
            (43, 15),
            (53, 25),
            (61, 30),
            (67, 33),
            (71, 35),
            (79, 40),
            (83, 41),
            (89, 44),
        ];

        for qlat in &test_qlats {
            for &(p, root) in &primes_and_roots {
                let (old, fk) =
                    compare_scatter(p, root, 7, qlat, log_i, sieve_width, max_j, half_i);
                assert_eq!(
                    old, fk,
                    "FK mismatch for p={}, root={}, qlat={:?}",
                    p, root, qlat
                );
            }
        }
    }

    #[test]
    fn test_fk_scatter_matches_old_medium_primes_fallback() {
        // Medium primes (between half_i and sieve_width) should fall back
        // correctly since the partial-GCD won't iterate.
        let qlat = reduce_qlattice(97, 30, 1.0);
        let log_i = 5; // half_i = 32, sieve_width = 64
        let half_i = 1i64 << log_i;
        let sieve_width = (2 * half_i) as usize;
        let max_j = half_i as usize;

        // Primes in [32, 64) range
        for root in [0u64, 1, 10, 15, 20, 30, 36] {
            let (old, fk) = compare_scatter(37, root, 7, &qlat, log_i, sieve_width, max_j, half_i);
            assert_eq!(
                old, fk,
                "FK mismatch for p=37, root={}, qlat={:?}",
                root, qlat
            );
        }
    }

    #[test]
    fn test_fk_scatter_matches_old_p_greater_than_total_area() {
        // Prime larger than total sieve area (at most 1 hit total)
        let qlat = reduce_qlattice(97, 30, 1.0);
        let log_i = 4; // half_i = 16, sieve_width = 32, max_j = 16, total = 512
        let half_i = 1i64 << log_i;
        let sieve_width = (2 * half_i) as usize;
        let max_j = half_i as usize;

        for root in [0u64, 1, 100, 500, 1000] {
            let (old, fk) =
                compare_scatter(1009, root, 7, &qlat, log_i, sieve_width, max_j, half_i);
            assert_eq!(
                old, fk,
                "FK mismatch for p=1009 (> total area), root={}",
                root
            );
        }
    }

    #[test]
    fn test_fk_scatter_exhaustive_small_sieve() {
        // Exhaustive test: try ALL valid (p, root) combinations for a small sieve
        // log_i = 3 => half_i = 8, sieve_width = 16, primes above 16
        let qlat = reduce_qlattice(97, 30, 1.0);
        let log_i = 3;
        let half_i = 1i64 << log_i;
        let sieve_width = (2 * half_i) as usize;
        let max_j = half_i as usize;

        // All primes from 17 to 100
        let primes: Vec<u64> = vec![
            17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
        ];

        for &p in &primes {
            for root in 0..p {
                let (old, fk) =
                    compare_scatter(p, root, 7, &qlat, log_i, sieve_width, max_j, half_i);
                assert_eq!(
                    old, fk,
                    "FK mismatch for p={}, root={} (exhaustive, log_i=3)",
                    p, root
                );
            }
        }
    }
}
