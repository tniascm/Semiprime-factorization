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
use crate::params::NfsParams;
use crate::relation::Relation;

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
) -> SieveResult {
    let start_time = std::time::Instant::now();

    let half_i = params.sieve_half_width() as i64;
    let sieve_width = (2 * half_i) as usize;
    let max_j = (params.sieve_half_width() / 2).max(1) as usize;
    let total_sieve_area = sieve_width * max_j;
    let n_buckets = (total_sieve_area + BUCKET_REGION - 1) / BUCKET_REGION;

    // Bucket threshold: primes below this use small sieve, above use bucket sieve
    let bucket_thresh = (half_i as u64).max(64);

    // Generate special-q primes
    let sq_primes: Vec<u64> = sieve_primes(params.qmin + params.qrange)
        .into_iter()
        .filter(|&p| p >= params.qmin)
        .collect();

    let scale = rat_fb.scale;

    // Rational polynomial: g(x) = x - m, so g1 = 1, g0 = -m
    let g0 = -(m as f64);
    let g1 = 1.0f64;

    let mut all_relations = Vec::new();
    let mut total_survivors = 0usize;
    let mut total_sieve_ns = 0u64;
    let mut total_cofact_ns = 0u64;
    let mut sq_count = 0usize;

    for &q in &sq_primes {
        // Find roots of f(x) mod q
        let q_roots = factorbase::find_roots_mod_p(f_coeffs, q);
        if q_roots.is_empty() {
            sq_count += 1;
            continue;
        }

        for &r in &q_roots {
            let sieve_start = std::time::Instant::now();

            // 1. Reduce q-lattice
            let skewness = 1.0;
            let qlat = reduce_qlattice(q, r, skewness);

            // 2. Precompute small sieve entries (primes below bucket_thresh)
            let small_rat_primes: Vec<u64> = rat_fb
                .primes
                .iter()
                .copied()
                .filter(|&p| p < bucket_thresh && p != q)
                .collect();
            let small_rat_logp: Vec<u8> = rat_fb
                .primes
                .iter()
                .enumerate()
                .filter(|(_, &p)| p < bucket_thresh && p != q)
                .map(|(idx, _)| rat_fb.log_p[idx])
                .collect();
            let small_rat =
                precompute_small_sieve_rat(&small_rat_primes, &small_rat_logp, m, &qlat);

            let small_alg_primes: Vec<u64> = alg_fb
                .primes
                .iter()
                .copied()
                .filter(|&p| p < bucket_thresh && p != q)
                .collect();
            let small_alg_roots: Vec<Vec<u64>> = alg_fb
                .primes
                .iter()
                .enumerate()
                .filter(|(_, &p)| p < bucket_thresh && p != q)
                .map(|(idx, _)| alg_fb.roots[idx].clone())
                .collect();
            let small_alg_logp: Vec<u8> = alg_fb
                .primes
                .iter()
                .enumerate()
                .filter(|(_, &p)| p < bucket_thresh && p != q)
                .map(|(idx, _)| alg_fb.log_p[idx])
                .collect();
            let small_alg = precompute_small_sieve_alg(
                &small_alg_primes,
                &small_alg_roots,
                &small_alg_logp,
                &qlat,
            );

            // 3. Scatter bucket updates for large primes
            // Estimate updates per bucket: BUCKET_REGION * sum(1/p) for large primes.
            // Mertens' theorem gives sum(1/p, p<=x) ~ ln(ln(x)), so for the range
            // [bucket_thresh, lim] the sum is modest. Use a generous upper bound.
            let est_updates = estimate_updates_per_bucket(
                &rat_fb.primes,
                &alg_fb.primes,
                &alg_fb.roots,
                bucket_thresh,
                q,
            );
            let updates_per_bucket = (est_updates * 2).max(1024); // 2x safety margin
            let mut rat_buckets = BucketArray::new(n_buckets.max(1), updates_per_bucket);
            let mut alg_buckets = BucketArray::new(n_buckets.max(1), updates_per_bucket);

            // Algebraic large FB primes
            for (fb_idx, &p) in alg_fb.primes.iter().enumerate() {
                if p < bucket_thresh || p == q {
                    continue;
                }
                for &root in &alg_fb.roots[fb_idx] {
                    scatter_bucket_updates_for_prime(
                        p,
                        root,
                        alg_fb.log_p[fb_idx],
                        &qlat,
                        params.log_i,
                        &mut alg_buckets,
                        sieve_width,
                        max_j,
                        half_i,
                    );
                }
            }

            // Rational large FB primes (rational root is m mod p)
            for (fb_idx, &p) in rat_fb.primes.iter().enumerate() {
                if p < bucket_thresh || p == q {
                    continue;
                }
                let rat_root = m % p;
                scatter_bucket_updates_for_prime(
                    p,
                    rat_root,
                    rat_fb.log_p[fb_idx],
                    &qlat,
                    params.log_i,
                    &mut rat_buckets,
                    sieve_width,
                    max_j,
                    half_i,
                );
            }

            let sieve_elapsed = sieve_start.elapsed().as_nanos() as u64;
            total_sieve_ns += sieve_elapsed;

            // 4. Process each bucket region
            let cofact_start = std::time::Instant::now();
            let mut survivors_this_sq: Vec<(i64, u64)> = Vec::new();

            for bucket_idx in 0..n_buckets {
                let region_start = bucket_idx * BUCKET_REGION;
                let region_end = (region_start + BUCKET_REGION).min(total_sieve_area);
                let region_len = region_end - region_start;
                if region_len == 0 {
                    continue;
                }

                // Init norm arrays row by row within this bucket region
                let mut rat_sieve = vec![0u8; region_len];
                let mut alg_sieve = vec![0u8; region_len];

                // Determine the range of j-rows this bucket region covers
                let first_j = region_start / sieve_width;
                let last_j = (region_end.saturating_sub(1)) / sieve_width;

                for j_row in first_j..=last_j {
                    let row_start_global = j_row * sieve_width;
                    let row_end_global = row_start_global + sieve_width;

                    // Overlap of this row with this bucket region
                    let overlap_start = row_start_global.max(region_start);
                    let overlap_end = row_end_global.min(region_end);
                    if overlap_start >= overlap_end {
                        continue;
                    }

                    let local_start = overlap_start - region_start;
                    let local_end = overlap_end - region_start;
                    let overlap_len = local_end - local_start;

                    // The i-offset within the row where this sub-region starts
                    let i_offset_in_row = overlap_start - row_start_global;

                    // Init rational norms for this row-slice
                    // init_norm_rat expects a slice covering [k_start..k_end] where i = k - half_i
                    // We need to init just the sub-range [i_offset_in_row..i_offset_in_row+overlap_len]
                    // Since init_norm_rat writes the entire slice treating index k as i = k - half_i,
                    // we need a temporary full-row buffer and copy the relevant part, OR we can
                    // compute directly.

                    // Direct computation for rational norms
                    let slope_rat = g1 * (qlat.a0 as f64) + g0 * (qlat.b0 as f64);
                    let intercept_rat =
                        (g1 * (qlat.a1 as f64) + g0 * (qlat.b1 as f64)) * (j_row as f64);

                    for k in 0..overlap_len {
                        let i_in_row = (i_offset_in_row + k) as i32 - half_i as i32;
                        let f_val = slope_rat * (i_in_row as f64) + intercept_rat;
                        let abs_f = f_val.abs();
                        rat_sieve[local_start + k] = if abs_f >= 1.0 {
                            (abs_f.log2() * scale).min(255.0).max(0.0) as u8
                        } else {
                            0
                        };
                    }

                    // Direct computation for algebraic norms
                    let a0f = qlat.a0 as f64;
                    let b0f = qlat.b0 as f64;
                    let a1f = qlat.a1 as f64;
                    let b1f = qlat.b1 as f64;
                    let j_f = j_row as f64;
                    let d = f_coeffs.len().saturating_sub(1);

                    for k in 0..overlap_len {
                        let i_in_row = (i_offset_in_row + k) as i32 - half_i as i32;
                        let i_f = i_in_row as f64;
                        let a = a0f * i_f + a1f * j_f;
                        let b = b0f * i_f + b1f * j_f;

                        let abs_f = eval_homogeneous_norm_f64(f_coeffs, a, b, d);
                        alg_sieve[local_start + k] = if abs_f >= 1.0 {
                            (abs_f.log2() * scale).min(255.0).max(0.0) as u8
                        } else {
                            0
                        };
                    }

                    // Apply small sieve for this row-slice
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
                }

                // Apply bucket updates from large primes
                let rat_updates = rat_buckets.updates_for_bucket(bucket_idx);
                let alg_updates = alg_buckets.updates_for_bucket(bucket_idx);
                apply_bucket_updates(&mut rat_sieve, rat_updates);
                apply_bucket_updates(&mut alg_sieve, alg_updates);

                // Scan for survivors
                let rat_bound = ((params.mfb0 as f64) * scale).min(255.0) as u8;
                let alg_bound = ((params.mfb1 as f64) * scale).min(255.0) as u8;

                let survivor_positions =
                    scan_survivors(&rat_sieve, &alg_sieve, rat_bound, alg_bound);

                for &pos in &survivor_positions {
                    let (i, j) = pos_to_ij(bucket_idx, pos, sieve_width, half_i);
                    if j == 0 {
                        continue;
                    }

                    // Convert (i,j) back to (a,b) via q-lattice
                    let a = qlat.a0 as i128 * i as i128 + qlat.a1 as i128 * j as i128;
                    let b = qlat.b0 as i128 * i as i128 + qlat.b1 as i128 * j as i128;

                    if b <= 0 {
                        continue;
                    }
                    if a == 0 {
                        continue;
                    }

                    if a.unsigned_abs() > i64::MAX as u128 || b > u64::MAX as i128 {
                        continue;
                    }

                    survivors_this_sq.push((a as i64, b as u64));
                }
            }

            total_survivors += survivors_this_sq.len();

            // 5. Cofactorize survivors
            for (a, b) in survivors_this_sq {
                let rat_norm = compute_rat_norm(a, b, m);
                let alg_norm = compute_alg_norm(a, b, f_coeffs);

                if rat_norm == 0 || alg_norm == 0 {
                    continue;
                }

                let rat_result = cofactor::cofactorize(
                    rat_norm,
                    &rat_fb.trial_divisors,
                    params.lpb0,
                    params.mfb0,
                    params.lim0,
                );
                let alg_result = cofactor::cofactorize(
                    alg_norm,
                    &alg_fb.trial_divisors,
                    params.lpb1,
                    params.mfb1,
                    params.lim1,
                );

                if let Some(rel) = build_relation(a, b, rat_result, alg_result) {
                    all_relations.push(rel);
                }
            }

            total_cofact_ns += cofact_start.elapsed().as_nanos() as u64;
        }

        sq_count += 1;

        // Check if we have enough relations
        if all_relations.len() as u64 >= params.rels_wanted {
            break;
        }
    }

    let total_elapsed = start_time.elapsed().as_secs_f64() * 1000.0;

    SieveResult {
        relations: all_relations,
        special_qs_processed: sq_count,
        survivors_found: total_survivors,
        total_ms: total_elapsed,
        sieve_ms: total_sieve_ns as f64 / 1_000_000.0,
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
    _half_i: i64,
) {
    let pl = reduce_plattice(p, root, qlat, log_i);
    if !pl.hits {
        return;
    }

    // Use the FK walk parameters for 1D traversal through the sieve area.
    // The PLattice provides inc_step (primary stride in 1D sieve coords).
    // For correctness, we use a simpler per-row approach that directly computes
    // hit positions, which is equivalent in work.

    // To compute per-row hits, we need the transformed root R' in (i,j) space.
    // The PLattice reduction already computed this implicitly. We can recover it
    // by using the original root transformation.

    let p_i128 = p as i128;

    // Recompute the transformed root R' = (root*b1 - a1) * (a0 - root*b0)^{-1} mod p
    let denom =
        ((qlat.a0 as i128 - (root as i128) * (qlat.b0 as i128)) % p_i128 + p_i128) % p_i128;
    if denom == 0 {
        return;
    }
    let inv = match crate::arith::mod_inverse(denom as u64, p) {
        Some(v) => v,
        None => return,
    };
    let numer = (((root as i128) * (qlat.b1 as i128) - qlat.a1 as i128) % p_i128 + p_i128)
        % p_i128;
    let r_prime = ((numer as u128 * inv as u128) % p as u128) as u64;

    let p_usize = p as usize;

    for j in 0..max_j {
        // First hit in row j: i_pos = (r_prime * j) mod p
        let j_mod_p = (j as u64) % p;
        let start_in_row = ((r_prime as u128 * j_mod_p as u128) % p as u128) as usize;

        // Walk through row at stride p, computing global 1D positions
        let row_base = j * sieve_width;
        let mut i_pos = start_in_row;

        while i_pos < sieve_width {
            let global_pos = row_base + i_pos;
            let bucket_idx = global_pos >> LOG_BUCKET_REGION;
            let bucket_pos = (global_pos & (BUCKET_REGION - 1)) as u16;

            if bucket_idx < buckets.n_buckets() {
                buckets.push(bucket_idx, BucketUpdate { pos: bucket_pos, logp });
            }

            i_pos += p_usize;
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

    if b.abs() > a.abs().max(1.0) * 1e-15 {
        // Horner in t = a/b: poly(t) = c_d * t^d + c_{d-1} * t^{d-1} + ... + c_0
        // F(a,b) = b^d * poly(a/b)
        let t = a / b;
        let mut poly = f_coeffs[d] as f64;
        for idx in (0..d).rev() {
            poly = poly * t + f_coeffs[idx] as f64;
        }
        let abs_bd = b.abs().powi(d as i32);
        (poly * abs_bd).abs()
    } else if a.abs() > 1e-30 {
        // b is essentially zero: F(a, 0) = c_d * a^d
        let c_d = f_coeffs[d] as f64;
        (c_d * a.powi(d as i32)).abs()
    } else {
        0.0
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
fn compute_alg_norm(a: i64, b: u64, f_coeffs: &[i64]) -> u64 {
    if f_coeffs.is_empty() {
        return 0;
    }
    let d = f_coeffs.len() - 1;

    if b == 0 {
        // F(a, 0) = c_d * a^d
        let a128 = a as i128;
        let mut a_pow: i128 = 1;
        for _ in 0..d {
            a_pow *= a128;
        }
        return ((f_coeffs[d] as i128) * a_pow).unsigned_abs() as u64;
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
            bp *= b128;
        }
        bp
    };

    for k in 0..=d {
        result += (f_coeffs[k] as i128) * a_pow * b_pow;
        a_pow *= a128;
        if k < d {
            // b_pow goes from b^d down to b^0
            b_pow /= b128;
        }
    }

    result.unsigned_abs() as u64
}

/// Build a Relation from cofactorization results, or None if not smooth enough.
fn build_relation(
    a: i64,
    b: u64,
    rat_result: CofactResult,
    alg_result: CofactResult,
) -> Option<Relation> {
    let (rat_factors, rat_cofactor) = match rat_result {
        CofactResult::Smooth(f) => (f, 0),
        CofactResult::OneLargePrime(f, lp) => (f, lp),
        CofactResult::TwoLargePrimes(f, lp1, _lp2) => (f, lp1),
        CofactResult::NotSmooth => return None,
    };
    let (alg_factors, alg_cofactor) = match alg_result {
        CofactResult::Smooth(f) => (f, 0),
        CofactResult::OneLargePrime(f, lp) => (f, lp),
        CofactResult::TwoLargePrimes(f, lp1, _lp2) => (f, lp1),
        CofactResult::NotSmooth => return None,
    };

    Some(Relation {
        a,
        b,
        rational_factors: rat_factors,
        algebraic_factors: alg_factors,
        rational_sign_negative: a < 0,
        algebraic_sign_negative: false,
        rat_cofactor,
        alg_cofactor,
    })
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
        assert_eq!(compute_alg_norm(3, 1, &[1, 0, 1]), 10);
        // F(2, 3) = 9 + 0 + 4 = 13
        assert_eq!(compute_alg_norm(2, 3, &[1, 0, 1]), 13);
    }

    #[test]
    fn test_compute_alg_norm_b_zero() {
        // f(x) = x^2 + 1, coeffs = [1, 0, 1]
        // F(5, 0) = c_2 * 5^2 = 25
        assert_eq!(compute_alg_norm(5, 0, &[1, 0, 1]), 25);
    }

    #[test]
    fn test_compute_alg_norm_degree3() {
        // f(x) = x^3 + 2x + 1, coeffs = [1, 2, 0, 1]
        // F(a, b) = b^3 + 2*a*b^2 + 0*a^2*b + a^3
        // F(1, 1) = 1 + 2 + 0 + 1 = 4
        assert_eq!(compute_alg_norm(1, 1, &[1, 2, 0, 1]), 4);
        // F(2, 1) = 1 + 4 + 0 + 8 = 13
        assert_eq!(compute_alg_norm(2, 1, &[1, 2, 0, 1]), 13);
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

        let result = sieve_specialq(&f_coeffs, m, &rat_fb, &alg_fb, &params);
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

        let result = sieve_specialq(&f_coeffs, m, &rat_fb, &alg_fb, &params);
        assert!(result.total_ms >= 0.0);
        assert!(result.sieve_ms >= 0.0);
        assert!(result.cofactor_ms >= 0.0);
    }

    #[test]
    fn test_build_relation_smooth() {
        let rat = CofactResult::Smooth(vec![(0, 1), (1, 2)]);
        let alg = CofactResult::Smooth(vec![(2, 3)]);
        let rel = build_relation(5, 3, rat, alg).unwrap();
        assert_eq!(rel.a, 5);
        assert_eq!(rel.b, 3);
        assert_eq!(rel.rat_cofactor, 0);
        assert_eq!(rel.alg_cofactor, 0);
        assert!(!rel.rational_sign_negative);
    }

    #[test]
    fn test_build_relation_not_smooth() {
        let rat = CofactResult::NotSmooth;
        let alg = CofactResult::Smooth(vec![]);
        assert!(build_relation(1, 1, rat, alg).is_none());
    }

    #[test]
    fn test_build_relation_negative_a() {
        let rat = CofactResult::Smooth(vec![]);
        let alg = CofactResult::Smooth(vec![]);
        let rel = build_relation(-7, 2, rat, alg).unwrap();
        assert!(rel.rational_sign_negative);
    }

    #[test]
    fn test_build_relation_one_large_prime() {
        let rat = CofactResult::OneLargePrime(vec![(0, 1)], 12347);
        let alg = CofactResult::Smooth(vec![(1, 1)]);
        let rel = build_relation(10, 3, rat, alg).unwrap();
        assert_eq!(rel.rat_cofactor, 12347);
        assert_eq!(rel.alg_cofactor, 0);
    }
}
