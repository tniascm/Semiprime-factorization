//! Production sieve pipeline for NFS relation collection.
//!
//! Architecture:
//!   Phase A: Fast polynomial root computation (pre-reduced u64 arithmetic)
//!   Phase B: Line sieve per b-row (rayon-parallel, f32 scores)
//!   Phase C: Per-cell threshold scan with approximate norm estimates
//!   Phase D: Cofactorization with large prime support (rational-first)

use std::time::Instant;

use classical_nfs::polynomial::NfsPolynomial;
use classical_nfs::sieve::{eval_homogeneous_abs, sieve_primes};
use num_bigint::BigUint;
use num_traits::ToPrimitive;
use rayon::prelude::*;

use crate::smoothness::{gcd, trial_divide};

/// Configuration for the sieve pipeline.
#[derive(Debug, Clone)]
pub struct SieveMcmcConfig {
    pub sieve_area: i64,
    pub max_b: i64,
    pub fb_bound: u64,
    pub lpb: u32,
    pub seed: u64,
}

impl SieveMcmcConfig {
    pub fn for_bits(bits: u32) -> Self {
        let (sieve_area, max_b, fb_bound, lpb) = match bits {
            0..=40 => (1i64 << 12, 1i64 << 8, 1u64 << 8, 14u32),
            41..=50 => (1 << 14, 1 << 9, 1 << 10, 16),
            51..=60 => (1 << 15, 1 << 10, 1 << 11, 17),
            61..=70 => (1 << 16, 1 << 11, 1 << 12, 18),
            71..=80 => (1 << 17, 1 << 12, 1 << 13, 18),
            81..=96 => (1 << 18, 1 << 12, 1 << 14, 18),
            97..=112 => (1 << 19, 1 << 13, 1 << 15, 19),
            113..=128 => (1 << 20, 1 << 13, 1 << 16, 20),
            129..=160 => (1 << 20, 1 << 13, 1 << 17, 22),
            _ => (1 << 21, 1 << 14, 1 << 18, 24),
        };
        Self {
            sieve_area,
            max_b,
            fb_bound,
            lpb,
            seed: 42,
        }
    }
}

/// A verified NFS relation.
#[derive(Debug, Clone)]
pub struct Relation {
    pub a: i64,
    pub b: i64,
    pub rat_factors: Vec<(u64, u32)>,
    pub alg_factors: Vec<(u64, u32)>,
    pub rat_cofactor: u64,
    pub alg_cofactor: u64,
}

impl Relation {
    pub fn is_full(&self) -> bool {
        self.rat_cofactor == 1 && self.alg_cofactor == 1
    }
    pub fn is_partial(&self) -> bool {
        !self.is_full()
    }
}

/// Timing breakdown for performance analysis.
#[derive(Debug, Clone, Default)]
pub struct PipelineTimings {
    pub setup_ms: f64,
    pub sieve_ms: f64,
    pub mcmc_ms: f64,
    pub cofactor_ms: f64,
    pub total_ms: f64,
    pub rows_processed: usize,
    pub candidates_found: usize,
    pub relations_found: usize,
    pub full_relations: usize,
    pub partial_relations: usize,
}

/// Algebraic roots for a single prime, computed with pre-reduced u64 arithmetic.
struct FastRoot {
    prime: u64,
    log_p: f32,
    roots: Vec<u64>,
}

/// Algebraic roots for a single prime, computed with pre-reduced u64 arithmetic.
pub(crate) struct FastRootPub {
    pub prime: u64,
    pub log_p: f32,
    pub roots: Vec<u64>,
}

/// Compute polynomial roots mod each prime using pre-reduced u64 coefficients.
///
/// This replaces the catastrophically slow `compute_polynomial_roots` from classical-nfs
/// which creates BigUint::from(p) per evaluation. Here we pre-reduce all coefficients
/// to u64 ONCE per prime, then use pure u128 Horner evaluation in the inner loop.
pub(crate) fn compute_roots_fast_pub(coeffs: &[BigUint], primes: &[u64]) -> Vec<FastRootPub> {
    primes
        .iter()
        .filter(|&&p| p > 1)
        .map(|&p| {
            let p_big = BigUint::from(p);
            let reduced: Vec<u64> = coeffs
                .iter()
                .map(|c| (c % &p_big).to_u64().unwrap_or(0))
                .collect();
            let d = reduced.len() - 1;
            let log_p = (p as f32).log2();
            let p128 = p as u128;
            let mut roots = Vec::new();
            for x in 0..p {
                let x128 = x as u128;
                let mut val = reduced[d] as u128;
                for i in (0..d).rev() {
                    val = (val * x128 + reduced[i] as u128) % p128;
                }
                if val == 0 {
                    roots.push(x);
                }
            }
            FastRootPub {
                prime: p,
                log_p,
                roots,
            }
        })
        .collect()
}

/// Compute polynomial roots mod each prime using pre-reduced u64 coefficients (private version).
fn compute_roots_fast(coeffs: &[BigUint], primes: &[u64]) -> Vec<FastRoot> {
    primes
        .iter()
        .filter(|&&p| p > 1)
        .map(|&p| {
            let p_big = BigUint::from(p);
            let reduced: Vec<u64> = coeffs
                .iter()
                .map(|c| (c % &p_big).to_u64().unwrap_or(0))
                .collect();

            let d = reduced.len() - 1;
            let log_p = (p as f32).log2();
            let p128 = p as u128;

            let mut roots = Vec::new();
            for x in 0..p {
                let x128 = x as u128;
                let mut val = reduced[d] as u128;
                for i in (0..d).rev() {
                    val = (val * x128 + reduced[i] as u128) % p128;
                }
                if val == 0 {
                    roots.push(x);
                }
            }

            FastRoot {
                prime: p,
                log_p,
                roots,
            }
        })
        .collect()
}

/// Fast log2 approximation using IEEE 754 exponent extraction.
/// Accuracy: ±0.09 (sufficient for sieve threshold comparisons).
/// ~3 cycles vs ~30 for f64::log2().
#[inline(always)]
pub(crate) fn fast_log2(x: f64) -> f32 {
    if x <= 0.0 {
        return -1000.0;
    }
    // IEEE 754 double: sign(1) | exponent(11) | mantissa(52)
    // For x = 2^e * (1 + f), log2(x) = e + log2(1+f) ≈ e + f
    // Treating the bits as an integer and scaling gives a good linear approx.
    let bits = x.to_bits();
    let biased = (bits >> 52) & 0x7ff;
    let mantissa_frac = (bits & 0x000f_ffff_ffff_ffff) as f32 / (1u64 << 52) as f32;
    (biased as f32 - 1023.0) + mantissa_frac
}

/// Trial divide a u128 value by primes in the factor base.
pub(crate) fn trial_divide_u128(mut val: u128, primes: &[u64]) -> (Vec<u32>, u128) {
    let mut exponents = vec![0u32; primes.len()];
    for (i, &p) in primes.iter().enumerate() {
        if p == 0 {
            continue;
        }
        let p128 = p as u128;
        while val % p128 == 0 {
            val /= p128;
            exponents[i] += 1;
        }
        if val == 1 {
            break;
        }
    }
    (exponents, val)
}

/// Precomputed data shared across all b-rows.
struct SieveData {
    primes: Vec<u64>,
    log_primes: Vec<f32>,
    alg_roots: Vec<FastRoot>,
    m: u64,
    m_f64: f64,
    width: usize,
    sieve_area: i64,
    lpb_bound: u64,
    lpb_f32: f32,
    /// Polynomial coefficients in f64 for fast approximate norm computation.
    coeffs_f64: Vec<f64>,
    degree: usize,
}

struct RowResult {
    relations: Vec<Relation>,
    candidates_found: usize,
    sieve_ms: f64,
    scan_ms: f64,
    cofactor_ms: f64,
}

/// Run the full sieve pipeline. Returns (relations, timings).
pub fn collect_relations(
    poly: &NfsPolynomial,
    m: u64,
    config: &SieveMcmcConfig,
) -> (Vec<Relation>, PipelineTimings) {
    let total_start = Instant::now();
    let setup_start = Instant::now();

    let primes = sieve_primes(config.fb_bound);
    let alg_roots = compute_roots_fast(&poly.coefficients, &primes);
    let log_primes: Vec<f32> = primes.iter().map(|&p| (p as f32).log2()).collect();

    let width = (2 * config.sieve_area + 1) as usize;
    let lpb_bound = 1u64 << config.lpb;

    let coeffs_f64: Vec<f64> = poly
        .coefficients
        .iter()
        .map(|c| c.to_f64().unwrap_or(0.0))
        .collect();

    let sieve_data = SieveData {
        primes,
        log_primes,
        alg_roots,
        m,
        m_f64: m as f64,
        width,
        sieve_area: config.sieve_area,
        lpb_bound,
        lpb_f32: config.lpb as f32,
        coeffs_f64,
        degree: poly.degree,
    };

    let setup_ms = setup_start.elapsed().as_secs_f64() * 1000.0;

    // Process b-rows in parallel with rayon
    let sieve_start = Instant::now();

    let row_results: Vec<RowResult> = (1..=config.max_b)
        .into_par_iter()
        .map(|b| process_row(b, &sieve_data, poly))
        .collect();

    let _wall_ms = sieve_start.elapsed().as_secs_f64() * 1000.0;

    // Aggregate results
    let mut all_relations = Vec::new();
    let mut total_candidates = 0usize;
    let mut sieve_ms = 0.0f64;
    let mut scan_ms = 0.0f64;
    let mut cofactor_ms = 0.0f64;

    for rr in &row_results {
        all_relations.extend(rr.relations.iter().cloned());
        total_candidates += rr.candidates_found;
        sieve_ms += rr.sieve_ms;
        scan_ms += rr.scan_ms;
        cofactor_ms += rr.cofactor_ms;
    }

    // Divide by thread count for wall-clock estimate
    let thread_count = rayon::current_num_threads() as f64;
    sieve_ms /= thread_count;
    scan_ms /= thread_count;
    cofactor_ms /= thread_count;

    let full_count = all_relations.iter().filter(|r| r.is_full()).count();
    let partial_count = all_relations.iter().filter(|r| r.is_partial()).count();

    let timings = PipelineTimings {
        setup_ms,
        sieve_ms,
        mcmc_ms: scan_ms,
        cofactor_ms,
        total_ms: total_start.elapsed().as_secs_f64() * 1000.0,
        rows_processed: config.max_b as usize,
        candidates_found: total_candidates,
        relations_found: all_relations.len(),
        full_relations: full_count,
        partial_relations: partial_count,
    };

    (all_relations, timings)
}

fn process_row(b: i64, data: &SieveData, poly: &NfsPolynomial) -> RowResult {
    let width = data.width;
    let b_u64 = b as u64;
    let a_min = -data.sieve_area;

    // === Phase A: Sieve this row ===
    let sieve_start = Instant::now();
    let mut rat_scores = vec![0.0f32; width];
    let mut alg_scores = vec![0.0f32; width];

    // Rational side: for each prime p, a ≡ -b*m (mod p), stride by p
    for (i, &p) in data.primes.iter().enumerate() {
        if p < 2 {
            continue;
        }
        let bm_mod_p = ((b_u64 as u128 * data.m as u128) % p as u128) as u64;
        let start_a_mod_p = if bm_mod_p == 0 { 0 } else { p - bm_mod_p };

        let a_min_mod_p = ((a_min % p as i64) + p as i64) as u64 % p;
        let offset = if start_a_mod_p >= a_min_mod_p {
            start_a_mod_p - a_min_mod_p
        } else {
            p - a_min_mod_p + start_a_mod_p
        };

        let log_p = data.log_primes[i];
        let mut idx = offset as usize;
        while idx < width {
            rat_scores[idx] += log_p;
            idx += p as usize;
        }
    }

    // Algebraic side: for each root r of f mod p, a ≡ -r*b (mod p), stride by p
    for fr in &data.alg_roots {
        let p = fr.prime;
        for &r in &fr.roots {
            let rb_mod_p = ((r as u128 * b_u64 as u128) % p as u128) as u64;
            let start_a_mod_p = if rb_mod_p == 0 { 0 } else { p - rb_mod_p };

            let a_min_mod_p = ((a_min % p as i64) + p as i64) as u64 % p;
            let offset = if start_a_mod_p >= a_min_mod_p {
                start_a_mod_p - a_min_mod_p
            } else {
                p - a_min_mod_p + start_a_mod_p
            };

            let mut idx = offset as usize;
            while idx < width {
                alg_scores[idx] += fr.log_p;
                idx += p as usize;
            }
        }
    }

    let sieve_ms = sieve_start.elapsed().as_secs_f64() * 1000.0;

    // === Phase B: Per-cell threshold scan ===
    // For each cell, compute approximate norms and use per-cell thresholds.
    // This fixes the fatal bug where a global max-norm threshold rejects
    // cells near polynomial roots (which have small norms and are most likely smooth).
    let scan_start = Instant::now();
    let mut candidates: Vec<(i64, i64)> = Vec::new();

    let b_f64 = b as f64;
    let d = data.degree;
    // Pre-compute b^d for homogeneous polynomial evaluation
    let b_pow_d = b_f64.powi(d as i32);
    let lpb_f32 = data.lpb_f32;

    // Minimum sieve score to even consider a cell (cheap pre-filter).
    const MIN_SCORE: f32 = 4.0;

    for i in 0..width {
        // Cheap pre-filter: skip cells with negligible sieve scores
        if rat_scores[i] < MIN_SCORE || alg_scores[i] < MIN_SCORE {
            continue;
        }

        let a = i as i64 + a_min;
        if a == 0 {
            continue;
        }

        // Per-cell rational threshold: check |a - b*m| <= 2^(sieve_score + lpb)
        // Using fast IEEE 754 log2 approximation (exponent extraction, ~3 cycles vs ~30)
        let rat_norm_f64 = (a as f64 - b_f64 * data.m_f64).abs();
        let rat_log = fast_log2(rat_norm_f64);
        let rat_thresh = (rat_log - lpb_f32).max(0.0);
        if rat_scores[i] < rat_thresh {
            continue;
        }

        // Per-cell algebraic norm estimate via Horner's method:
        // F(a,b) = b^d * f(a/b) where f(x) = sum c_k * x^k
        let x = a as f64 / b_f64;
        let mut val = data.coeffs_f64[d];
        for k in (0..d).rev() {
            val = val * x + data.coeffs_f64[k];
        }
        let alg_norm_f64 = val.abs() * b_pow_d;
        let alg_log = fast_log2(alg_norm_f64);
        let alg_thresh = (alg_log - lpb_f32).max(0.0);
        if alg_scores[i] < alg_thresh {
            continue;
        }

        // GCD check: coprimality required for valid (a,b) pair
        if gcd(a.unsigned_abs(), b_u64) == 1 {
            candidates.push((a, b));
        }
    }

    let scan_ms = scan_start.elapsed().as_secs_f64() * 1000.0;

    // === Phase C: Cofactorization (rational-first to skip expensive BigUint) ===
    let cofactor_start = Instant::now();
    let mut relations = Vec::new();

    for &(a, b_val) in &candidates {
        if let Some(rel) = try_cofactorize(a, b_val, data, poly) {
            relations.push(rel);
        }
    }

    let cofactor_ms = cofactor_start.elapsed().as_secs_f64() * 1000.0;

    RowResult {
        relations,
        candidates_found: candidates.len(),
        sieve_ms,
        scan_ms,
        cofactor_ms,
    }
}

/// Compute actual norms, trial divide, check large prime bounds.
/// Checks rational side FIRST to avoid expensive BigUint algebraic norm
/// computation for candidates that fail the rational check (~95% of candidates).
fn try_cofactorize(
    a: i64,
    b: i64,
    data: &SieveData,
    poly: &NfsPolynomial,
) -> Option<Relation> {
    // Rational norm: |a - b*m| (cheap, fits in u128)
    let rat_norm_i128 = a as i128 - (b as i128) * (data.m as i128);
    let rat_norm = rat_norm_i128.unsigned_abs();
    if rat_norm == 0 {
        return None;
    }

    // Trial divide rational side FIRST
    let (rat_exps, rat_cofactor) = if rat_norm <= u64::MAX as u128 {
        let (e, c) = trial_divide(rat_norm as u64, &data.primes);
        (e, c as u128)
    } else {
        trial_divide_u128(rat_norm, &data.primes)
    };

    // Quick reject: if rational cofactor exceeds large prime bound, skip
    if rat_cofactor > 1 && rat_cofactor > data.lpb_bound as u128 {
        return None;
    }

    // Algebraic norm: |F(a,b)| — expensive BigUint, only compute if rational passed
    let alg_norm_big = eval_homogeneous_abs(&poly.coefficients, a, b);
    let alg_norm: u128 = match alg_norm_big.to_u128() {
        Some(v) if v > 0 => v,
        _ => return None,
    };

    // Trial divide algebraic side
    let (alg_exps, alg_cofactor) = trial_divide_u128(alg_norm, &data.primes);

    // Check algebraic cofactor
    if alg_cofactor > 1 && alg_cofactor > data.lpb_bound as u128 {
        return None;
    }

    // Build factor lists
    let rat_factors: Vec<(u64, u32)> = data
        .primes
        .iter()
        .zip(rat_exps.iter())
        .filter(|(_, &e)| e > 0)
        .map(|(&p, &e)| (p, e))
        .collect();

    let alg_factors: Vec<(u64, u32)> = data
        .primes
        .iter()
        .zip(alg_exps.iter())
        .filter(|(_, &e)| e > 0)
        .map(|(&p, &e)| (p, e))
        .collect();

    Some(Relation {
        a,
        b,
        rat_factors,
        alg_factors,
        rat_cofactor: rat_cofactor as u64,
        alg_cofactor: alg_cofactor as u64,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use classical_nfs::polynomial::select_polynomial;

    #[test]
    fn test_compute_roots_fast() {
        // f(x) = x^2 + 1: roots mod 5 are {2, 3}
        let coeffs = vec![
            BigUint::from(1u64),
            BigUint::from(0u64),
            BigUint::from(1u64),
        ];
        let roots = compute_roots_fast(&coeffs, &[5]);
        assert_eq!(roots.len(), 1);
        assert_eq!(roots[0].prime, 5);
        let mut r: Vec<u64> = roots[0].roots.clone();
        r.sort();
        assert_eq!(r, vec![2, 3]);
    }

    #[test]
    fn test_trial_divide_u128() {
        let (exps, cofactor) = trial_divide_u128(360, &[2, 3, 5, 7]);
        assert_eq!(exps, vec![3, 2, 1, 0]);
        assert_eq!(cofactor, 1);

        // Large value that doesn't fit u64
        let big: u128 = (1u128 << 100) * 3;
        let (exps2, cofactor2) = trial_divide_u128(big, &[2, 3]);
        assert_eq!(exps2[0], 100);
        assert_eq!(exps2[1], 1);
        assert_eq!(cofactor2, 1);
    }

    #[test]
    fn test_sieve_finds_relations_32bit() {
        let n = BigUint::from(2214461131u64);
        let poly = select_polynomial(&n, 3);
        let m = poly.m.to_u64().unwrap();
        let config = SieveMcmcConfig::for_bits(32);

        let (relations, timings) = collect_relations(&poly, m, &config);

        eprintln!(
            "32-bit: {} relations ({} full, {} partial), {} candidates",
            timings.relations_found,
            timings.full_relations,
            timings.partial_relations,
            timings.candidates_found
        );
        eprintln!(
            "  Time: {:.1}ms (setup: {:.1}, sieve: {:.1}, scan: {:.1}, cofactor: {:.1})",
            timings.total_ms,
            timings.setup_ms,
            timings.sieve_ms,
            timings.mcmc_ms,
            timings.cofactor_ms
        );

        assert!(!relations.is_empty(), "Should find relations for 32-bit");

        // Verify a relation
        let rel = &relations[0];
        let rat_norm = (rel.a as i128 - (rel.b as i128) * (m as i128)).unsigned_abs() as u64;
        let mut reconstructed = rel.rat_cofactor;
        for &(p, e) in &rel.rat_factors {
            for _ in 0..e {
                reconstructed *= p;
            }
        }
        assert_eq!(
            reconstructed, rat_norm,
            "Rational factorization must be correct"
        );
    }

    #[test]
    fn test_sieve_finds_relations_64bit() {
        let n = BigUint::from(13763568394002235027u64);
        let poly = select_polynomial(&n, 3);
        let m = poly.m.to_u64().unwrap();
        let config = SieveMcmcConfig::for_bits(64);

        let (relations, timings) = collect_relations(&poly, m, &config);

        eprintln!(
            "64-bit: {} relations ({} full, {} partial), {} candidates in {:.1}ms",
            timings.relations_found,
            timings.full_relations,
            timings.partial_relations,
            timings.candidates_found,
            timings.total_ms
        );

        assert!(
            !relations.is_empty(),
            "Should find relations for 64-bit semiprime"
        );
    }

    #[test]
    fn test_sieve_96bit() {
        let n = BigUint::from(65351508052009705689102957167u128);
        let poly = select_polynomial(&n, 3);
        let m = poly.m.to_u64().unwrap();
        let config = SieveMcmcConfig::for_bits(96);

        let (relations, timings) = collect_relations(&poly, m, &config);

        let rels_per_sec = if timings.total_ms > 0.0 {
            relations.len() as f64 / (timings.total_ms / 1000.0)
        } else {
            0.0
        };
        eprintln!(
            "96-bit: {} relations ({} full, {} partial) in {:.1}ms = {:.0} rels/sec",
            relations.len(),
            timings.full_relations,
            timings.partial_relations,
            timings.total_ms,
            rels_per_sec
        );
        eprintln!(
            "  setup: {:.1}ms, sieve: {:.1}ms, scan: {:.1}ms, cofactor: {:.1}ms, candidates: {}",
            timings.setup_ms,
            timings.sieve_ms,
            timings.mcmc_ms,
            timings.cofactor_ms,
            timings.candidates_found
        );
    }
}
